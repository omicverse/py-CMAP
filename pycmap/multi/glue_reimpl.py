"""
Faithful reimplementation of GLUE (no scglue dependency).

Replicates GLUE's exact architecture and training procedure:
1. SEPARATE encoders per modality (not shared)
2. Graph convolution on FULL graph, loss on SAMPLED edges + negative sampling
3. Adversarial discriminator with burnin noise
4. NB decoder with softmax normalization
5. Pretrain → balancing weights → finetune
6. Edge normalization: keepvar (in-degree^-0.5)

Then we can add spatial loss and compare.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np
import logging
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)
EPS = 1e-7
TOTAL_COUNT = 1e4


# ============================================================
# Exact GLUE components
# ============================================================

class DataEncoder(nn.Module):
    """GLUE's NBDataEncoder: MLP encoder with log-normalization."""

    def __init__(self, in_features, out_features, h_depth=2, h_dim=256, dropout=0.2):
        super().__init__()
        self.h_depth = h_depth
        ptr_dim = in_features
        for layer in range(h_depth):
            setattr(self, f'linear_{layer}', nn.Linear(ptr_dim, h_dim))
            setattr(self, f'act_{layer}', nn.LeakyReLU(0.2))
            setattr(self, f'bn_{layer}', nn.BatchNorm1d(h_dim))
            setattr(self, f'dropout_{layer}', nn.Dropout(dropout))
            ptr_dim = h_dim
        self.loc = nn.Linear(ptr_dim, out_features)
        self.std_lin = nn.Linear(ptr_dim, out_features)

    def compute_l(self, x):
        return x.sum(dim=1, keepdim=True)

    def normalize(self, x, l):
        return (x * (TOTAL_COUNT / l)).log1p()

    def forward(self, x, xrep, lazy_normalizer=True):
        if xrep.numel():
            l = None if lazy_normalizer else self.compute_l(x)
            ptr = xrep
        else:
            l = self.compute_l(x)
            ptr = self.normalize(x, l)
        for layer in range(self.h_depth):
            ptr = getattr(self, f'linear_{layer}')(ptr)
            ptr = getattr(self, f'act_{layer}')(ptr)
            ptr = getattr(self, f'bn_{layer}')(ptr)
            ptr = getattr(self, f'dropout_{layer}')(ptr)
        loc = self.loc(ptr)
        std = F.softplus(self.std_lin(ptr)) + EPS
        return D.Normal(loc, std), l


class DataDecoder(nn.Module):
    """GLUE's NBDataDecoder."""

    def __init__(self, n_features, n_batches=1):
        super().__init__()
        self.scale_lin = nn.Parameter(torch.zeros(n_batches, n_features))
        self.bias = nn.Parameter(torch.zeros(n_batches, n_features))
        self.log_theta = nn.Parameter(torch.zeros(n_batches, n_features))

    def forward(self, u, v, b, l):
        scale = F.softplus(self.scale_lin[b])
        logit_mu = scale * (u @ v.T) + self.bias[b]
        mu = F.softmax(logit_mu, dim=1) * l
        log_theta = self.log_theta[b]
        return D.NegativeBinomial(log_theta.exp(), logits=(mu + EPS).log() - log_theta)


class GraphEncoder(nn.Module):
    """GLUE's GraphEncoder: learnable vertex embeddings + graph conv."""

    def __init__(self, n_vertices, latent_dim):
        super().__init__()
        self.vrepr = nn.Parameter(torch.zeros(n_vertices, latent_dim))
        nn.init.normal_(self.vrepr, std=0.01)
        self.loc = nn.Linear(latent_dim, latent_dim)
        self.std_lin = nn.Linear(latent_dim, latent_dim)

    def forward(self, eidx, enorm, esgn):
        """Graph convolution on the full graph."""
        src, tgt = eidx[0], eidx[1]
        messages = self.vrepr[src] * (esgn * enorm).unsqueeze(1)
        agg = torch.zeros_like(self.vrepr)
        agg.scatter_add_(0, tgt.unsqueeze(1).expand_as(messages), messages)
        loc = self.loc(agg)
        std = F.softplus(self.std_lin(agg)) + EPS
        return D.Normal(loc, std)


class GraphDecoder(nn.Module):
    """GLUE's GraphDecoder: dot-product edge prediction."""

    def forward(self, v, eidx, esgn):
        src, tgt = eidx[0], eidx[1]
        logits = esgn * (v[src] * v[tgt]).sum(dim=1)
        return D.Bernoulli(logits=logits)


class Discriminator(nn.Module):
    """GLUE's Discriminator."""

    def __init__(self, latent_dim, h_dim=256, h_depth=2, dropout=0.2):
        super().__init__()
        layers = []
        dim = latent_dim
        for _ in range(h_depth):
            layers += [nn.Linear(dim, h_dim), nn.LeakyReLU(0.2), nn.Dropout(dropout)]
            dim = h_dim
        layers.append(nn.Linear(h_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, u):
        return self.net(u)


# ============================================================
# Edge normalization (keepvar, like GLUE)
# ============================================================

def normalize_edges_keepvar(edge_index, edge_weight, n_vertices):
    """Keepvar normalization: w * in_degree^(-0.5)."""
    from scipy.sparse import coo_matrix
    src, tgt = edge_index[0], edge_index[1]
    wt = np.abs(edge_weight)
    adj = coo_matrix((wt, (src, tgt)), shape=(n_vertices, n_vertices))
    in_deg = np.array(adj.sum(axis=0)).ravel()
    normalizer = np.where(in_deg > 0, in_deg ** (-0.5), 0.0)
    return (edge_weight * normalizer[tgt]).astype(np.float32)


# ============================================================
# Graph edge sampler with negative sampling
# ============================================================

class GraphEdgeSampler:
    def __init__(self, edge_index, edge_weight, edge_sign, n_vertices,
                 batch_size=10000, neg_samples=10):
        mask = (edge_index[0] != edge_index[1]) & (np.abs(edge_weight) > 0.01)
        self.pos_src = edge_index[0, mask]
        self.pos_tgt = edge_index[1, mask]
        self.pos_wt = edge_weight[mask]
        self.pos_sgn = edge_sign[mask]
        self.n_pos = len(self.pos_src)
        self.n_vertices = n_vertices
        self.batch_size = batch_size
        self.neg_samples = neg_samples

    def sample(self, device):
        n_pos = min(self.batch_size, self.n_pos)
        idx = np.random.choice(self.n_pos, n_pos, replace=False)
        n_neg = n_pos * self.neg_samples
        neg_src = np.random.randint(0, self.n_vertices, n_neg)
        neg_tgt = np.random.randint(0, self.n_vertices, n_neg)

        all_src = np.concatenate([self.pos_src[idx], neg_src])
        all_tgt = np.concatenate([self.pos_tgt[idx], neg_tgt])
        all_wt = np.concatenate([self.pos_wt[idx], np.zeros(n_neg, dtype=np.float32)])
        all_sgn = np.concatenate([self.pos_sgn[idx], np.ones(n_neg, dtype=np.int8)])

        return (torch.tensor(np.stack([all_src, all_tgt]), dtype=torch.long, device=device),
                torch.tensor(all_wt, dtype=torch.float32, device=device),
                torch.tensor(all_sgn, dtype=torch.float32, device=device))


# ============================================================
# Training
# ============================================================

def train_glue_reimpl(adata_rna, adata_atac, graph,
                      spatial_coords=None,
                      latent_dim=50, n_epochs=1500, batch_size=256,
                      lr=2e-3, sample_ratio=0.2,
                      lam_data=1.0, lam_kl=1.0, lam_graph=0.02,
                      lam_align=0.05, lam_spatial=0.0,
                      burnin_frac=0.3, neg_samples=10,
                      device='cuda'):
    """
    Train GLUE reimplementation.

    Set lam_spatial=0 for pure GLUE replica.
    Set lam_spatial>0 to add spatial coherence (our innovation).
    """
    from scipy.sparse import issparse

    # ---- Prepare data ----
    rna_hvg = graph['rna_hvg_mask']
    atac_hvg = graph['atac_hvg_mask']
    rna_names, atac_names = graph['rna_names'], graph['atac_names']

    rna_hvg_names = [rna_names[i] for i in range(len(rna_names)) if rna_hvg[i]]
    rna_mask = np.isin(adata_rna.var_names, rna_hvg_names)
    X_rna = adata_rna.X[:, rna_mask]
    if issparse(X_rna): X_rna = X_rna.toarray()
    X_rna = torch.tensor(X_rna, dtype=torch.float32, device=device)
    rep_rna = torch.tensor(adata_rna.obsm['X_pca'].astype(np.float32), device=device)

    atac_hvg_names = [atac_names[i] for i in range(len(atac_names)) if atac_hvg[i]]
    atac_mask = np.isin(adata_atac.var_names, atac_hvg_names)
    X_atac = adata_atac.X[:, atac_mask]
    if issparse(X_atac): X_atac = X_atac.toarray()
    X_atac = torch.tensor(X_atac, dtype=torch.float32, device=device)
    rep_atac = torch.tensor(adata_atac.obsm['X_lsi'].astype(np.float32), device=device)

    n_st, n_cells = X_rna.shape[0], X_atac.shape[0]
    n_feat_rna, n_feat_atac = X_rna.shape[1], X_atac.shape[1]
    rep_dim_rna = rep_rna.shape[1]
    rep_dim_atac = rep_atac.shape[1]

    logger.info(f"Data: {n_st} ST ({n_feat_rna} HVG, rep={rep_dim_rna}), "
                f"{n_cells} ATAC ({n_feat_atac} peaks, rep={rep_dim_atac})")

    # ---- Spatial graph ----
    sp_adj_t, sp_wt_t = None, None
    if spatial_coords is not None:
        from .spatial_encoder import build_spatial_graph
        sp_adj, sp_wt = build_spatial_graph(spatial_coords, k=6)
        sp_adj_t = torch.tensor(sp_adj, dtype=torch.long, device=device)
        sp_wt_t = torch.tensor(sp_wt, dtype=torch.float32, device=device)

    # ---- Graph: full edges (for convolution) + sampler (for loss) ----
    n_vertices = graph['n_rna'] + graph['n_atac']
    rna_hvg_idx = np.where(rna_hvg)[0]
    atac_hvg_idx = np.where(atac_hvg)[0]

    # Normalize edges (keepvar, like GLUE)
    enorm = normalize_edges_keepvar(graph['edge_index'], graph['edge_weight'], n_vertices)
    full_eidx = torch.tensor(graph['edge_index'], dtype=torch.long, device=device)
    full_enorm = torch.tensor(enorm, dtype=torch.float32, device=device)
    full_esgn = torch.tensor(graph['edge_sign'], dtype=torch.float32, device=device)

    edge_sampler = GraphEdgeSampler(
        graph['edge_index'], graph['edge_weight'], graph['edge_sign'],
        n_vertices, batch_size=min(10000, graph['edge_index'].shape[1] // 3),
        neg_samples=neg_samples)

    rna_fidx = torch.tensor(rna_hvg_idx, dtype=torch.long, device=device)
    atac_fidx = torch.tensor(graph['n_rna'] + atac_hvg_idx, dtype=torch.long, device=device)

    # ---- Build model (SEPARATE encoders, like GLUE) ----
    enc_rna = DataEncoder(rep_dim_rna, latent_dim).to(device)
    enc_atac = DataEncoder(rep_dim_atac, latent_dim).to(device)
    g2v = GraphEncoder(n_vertices, latent_dim).to(device)
    v2g = GraphDecoder()
    dec_rna = DataDecoder(n_feat_rna).to(device)
    dec_atac = DataDecoder(n_feat_atac).to(device)
    disc = Discriminator(latent_dim).to(device)

    prior = D.Normal(torch.zeros(latent_dim, device=device),
                     torch.ones(latent_dim, device=device))

    all_params = (list(enc_rna.parameters()) + list(enc_atac.parameters()) +
                  list(g2v.parameters()) + list(dec_rna.parameters()) +
                  list(dec_atac.parameters()))
    dsc_params = list(disc.parameters())

    BURNIN_NOISE_EXAG = 1.5

    # ---- Compute losses (exact GLUE logic) ----
    def compute_losses(epoch, burnin, dsc_only=False, dsc_wt_rna=None, dsc_wt_atac=None,
                       ri=None, ai=None):
        b0_r = torch.zeros(len(ri), dtype=torch.long, device=device)
        b0_a = torch.zeros(len(ai), dtype=torch.long, device=device)

        # Encode
        u_rna, l_rna = enc_rna(X_rna[ri], rep_rna[ri], lazy_normalizer=dsc_only)
        u_atac, l_atac = enc_atac(X_atac[ai], rep_atac[ai], lazy_normalizer=dsc_only)
        z_rna = u_rna.rsample()
        z_atac = u_atac.rsample()

        # Adversarial
        u_cat = torch.cat([u_rna.mean, u_atac.mean])
        anneal = max(1 - (epoch - 1) / burnin, 0) if burnin > 0 else 0
        if anneal > 0:
            noise = D.Normal(torch.zeros_like(u_cat[0]), u_cat.std(dim=0)).sample((u_cat.shape[0],))
            u_cat_n = u_cat + anneal * BURNIN_NOISE_EXAG * noise
        else:
            u_cat_n = u_cat

        labels = torch.cat([torch.zeros(len(ri), dtype=torch.long, device=device),
                             torch.ones(len(ai), dtype=torch.long, device=device)])

        if dsc_wt_rna is not None:
            xdwt = torch.cat([dsc_wt_rna[ri], dsc_wt_atac[ai]])
            dsc_loss = (F.cross_entropy(disc(u_cat_n), labels, reduction='none') * xdwt).sum() / xdwt.numel()
        else:
            xdwt = torch.ones(len(u_cat), device=device)
            dsc_loss = F.cross_entropy(disc(u_cat_n), labels)

        if dsc_only:
            return {'dsc_loss': lam_align * dsc_loss}

        # Graph encoder (full graph for convolution)
        v = g2v(full_eidx, full_enorm, full_esgn)
        vsamp = v.rsample()

        # Graph loss (SAMPLED edges with negative sampling)
        s_eidx, s_ewt, s_esgn = edge_sampler.sample(device)
        # GLUE's graph decoder uses Bernoulli — binarize weights
        s_ewt_bin = (s_ewt.abs() > 0).float()
        g_nll_raw = -v2g(vsamp, s_eidx, s_esgn).log_prob(s_ewt_bin)
        pos_mask = (s_ewt_bin != 0).to(torch.int64)
        n_pos = pos_mask.sum().item()
        n_neg = pos_mask.numel() - n_pos
        g_nll_pn = torch.zeros(2, dtype=g_nll_raw.dtype, device=device)
        g_nll_pn.scatter_add_(0, pos_mask, g_nll_raw)
        avgc = (n_pos > 0) + (n_neg > 0)
        g_nll = (g_nll_pn[0] / max(n_neg, 1) + g_nll_pn[1] / max(n_pos, 1)) / avgc
        g_kl = D.kl_divergence(v, prior).sum(dim=1).mean() / vsamp.shape[0]
        g_elbo = g_nll + lam_kl * g_kl

        # Reconstruction
        v_rna = vsamp[rna_fidx]
        v_atac = vsamp[atac_fidx]

        if l_rna is None:
            l_rna = enc_rna.compute_l(X_rna[ri])
        if l_atac is None:
            l_atac = enc_atac.compute_l(X_atac[ai])

        rna_nll = -dec_rna(z_rna, v_rna, b0_r, l_rna).log_prob(X_rna[ri]).nanmean()

        atac_lp = dec_atac(z_atac, v_atac, b0_a, l_atac).log_prob(X_atac[ai])
        if sample_ratio and n_feat_atac > 2000:
            n_s = max(int(n_feat_atac * sample_ratio), 500)
            si = torch.randperm(n_feat_atac, device=device)[:n_s]
            atac_nll = -(atac_lp[:, si].nanmean()) * (n_feat_atac / n_s)
        else:
            atac_nll = -atac_lp.nanmean()

        rna_kl = D.kl_divergence(u_rna, prior).sum(1).mean() / X_rna[ri].shape[1]
        atac_kl = D.kl_divergence(u_atac, prior).sum(1).mean() / X_atac[ai].shape[1]
        rna_elbo = rna_nll + lam_kl * rna_kl
        atac_elbo = atac_nll + lam_kl * atac_kl

        x_elbo = 0.5 * (rna_elbo + atac_elbo)
        vae_loss = lam_data * x_elbo + lam_graph * 2 * g_elbo

        # Spatial coherence (our addition)
        sp_val = torch.tensor(0.0, device=device)
        if sp_adj_t is not None:
            u_all_st, _ = enc_rna(X_rna, rep_rna, lazy_normalizer=True)
            diff = u_all_st.mean[sp_adj_t[0]] - u_all_st.mean[sp_adj_t[1]]
            sp_val = (sp_wt_t.unsqueeze(1) * diff.pow(2)).mean()

        gen_loss = vae_loss - lam_align * dsc_loss + _lam_sp[0] * sp_val

        return {'gen_loss': gen_loss, 'dsc_loss': dsc_loss, 'vae_loss': vae_loss,
                'rna_nll': rna_nll, 'atac_nll': atac_nll, 'g_nll': g_nll, 'sp': sp_val}

    # ---- Phase runner ----
    def run_phase(name, max_ep, burnin, dsc_wt_rna=None, dsc_wt_atac=None):
        gen_opt = torch.optim.RMSprop(all_params, lr=lr)
        dsc_opt = torch.optim.RMSprop(dsc_params, lr=lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            gen_opt, patience=max(max_ep // 10, 15), factor=0.5, min_lr=1e-5)
        best, pat = float('inf'), 0
        patience = max(max_ep // 8, 20)

        for ep in range(max_ep):
            for m in [enc_rna, enc_atac, g2v, dec_rna, dec_atac, disc]:
                m.train()

            ri = torch.randperm(n_st, device=device)[:batch_size]
            ai = torch.randperm(n_cells, device=device)[:batch_size]

            # Discriminator step
            dl = compute_losses(ep, burnin, dsc_only=True,
                                dsc_wt_rna=dsc_wt_rna, dsc_wt_atac=dsc_wt_atac, ri=ri, ai=ai)
            dsc_opt.zero_grad()
            dl['dsc_loss'].backward()
            dsc_opt.step()

            # Generator step
            losses = compute_losses(ep, burnin, dsc_only=False,
                                    dsc_wt_rna=dsc_wt_rna, dsc_wt_atac=dsc_wt_atac, ri=ri, ai=ai)
            gen_opt.zero_grad()
            losses['gen_loss'].backward()
            nn.utils.clip_grad_norm_(all_params, 1.0)
            gen_opt.step()

            vae_val = losses['vae_loss'].item()
            sched.step(vae_val)
            if vae_val < best: best = vae_val; pat = 0
            else: pat += 1
            if pat >= patience and ep > burnin:
                logger.info(f"  {name} early stop at epoch {ep}")
                break

            if ep % 50 == 0:
                logger.info(f"  {name} {ep:4d}: vae={vae_val:.4f} "
                            f"rna={losses['rna_nll'].item():.3f} atac={losses['atac_nll'].item():.3f} "
                            f"dsc={losses['dsc_loss'].item():.3f} g={losses['g_nll'].item():.3f} "
                            f"sp={losses['sp'].item():.4f}")

    # ---- Balancing weights ----
    def estimate_weights():
        import scanpy as sc, anndata as ad
        for m in [enc_rna, enc_atac]: m.eval()
        with torch.no_grad():
            z_r = enc_rna(X_rna, rep_rna, lazy_normalizer=True)[0].mean.cpu().numpy()
            z_a = enc_atac(X_atac, rep_atac, lazy_normalizer=True)[0].mean.cpu().numpy()

        from sklearn.preprocessing import normalize as sk_norm
        result = {}
        centroids_all = {}
        for label, z in [('rna', z_r), ('atac', z_a)]:
            tmp = ad.AnnData(obsm={'X': z})
            sc.pp.neighbors(tmp, use_rep='X', metric='cosine')
            sc.tl.leiden(tmp, resolution=1.0)
            cl = tmp.obs['leiden'].values.astype(int)
            cent = np.array([z[cl == c].mean(0) for c in range(cl.max() + 1)])
            centroids_all[label] = (cl, sk_norm(cent, norm='l2'))

        cos = centroids_all['rna'][1] @ centroids_all['atac'][1].T
        cos[cos < 0.5] = 0; cos = cos ** 4

        for side, (cl, _), axis in [('rna', centroids_all['rna'], 1),
                                     ('atac', centroids_all['atac'], 0)]:
            n_per = np.array([(cl == c).sum() for c in range(cl.max() + 1)])
            bal = cos.sum(axis=axis) / n_per
            # Safety: if no clusters matched (all cosine < 0.5), use uniform weights
            if bal.sum() == 0 or np.isnan(bal.sum()):
                logger.warning(f"  {side}: no cluster matches, using uniform weights")
                bal = np.ones_like(bal)
            bal = bal / (bal.sum() / len(bal))
            result[side] = torch.tensor(bal[cl], dtype=torch.float32, device=device)

        logger.info(f"Balancing: RNA [{result['rna'].min():.2f},{result['rna'].max():.2f}], "
                    f"ATAC [{result['atac'].min():.2f},{result['atac'].max():.2f}]")
        return result['rna'], result['atac']

    # ==== EXECUTE ====
    burnin_ep = int(n_epochs * burnin_frac)

    # Use mutable container so compute_losses closure sees updates
    _lam_sp = [lam_spatial]

    # Phase 1: Pretrain WITHOUT spatial loss
    _lam_sp[0] = 0.0
    logger.info(f"Phase 1: Pretrain (burnin={burnin_ep}, no spatial)")
    run_phase("Pretrain", n_epochs, burnin_ep)

    dwt_r, dwt_a = estimate_weights()

    # Phase 2: Finetune WITH spatial loss + balancing weights
    _lam_sp[0] = lam_spatial
    ft_burnin = max(burnin_ep // 8, 30)
    logger.info(f"Phase 2: Finetune (burnin={ft_burnin}, lam_spatial={lam_spatial})")
    run_phase("Finetune", n_epochs, ft_burnin, dwt_r, dwt_a)

    # Extract embeddings
    for m in [enc_rna, enc_atac]: m.eval()
    with torch.no_grad():
        z_rna_out = enc_rna(X_rna, rep_rna, lazy_normalizer=True)[0].mean.cpu().numpy()
        z_atac_out = enc_atac(X_atac, rep_atac, lazy_normalizer=True)[0].mean.cpu().numpy()

    logger.info(f"Embeddings: RNA {z_rna_out.shape}, ATAC {z_atac_out.shape}")
    return z_rna_out, z_atac_out
