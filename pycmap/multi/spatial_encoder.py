"""
Spatial-Guided Cross-Modal Encoder (v4).

Our architecture + GLUE's training engineering:
  Architecture (ours):
    - Shared encoder body (natural mixing)
    - Spatial coherence loss (ST spatial prior)
    - Sampled reconstruction (ATAC regularization)
  Training (GLUE's):
    - Graph edge batching with negative sampling
    - Pretrain → balance weights → finetune (two-phase)
    - Adversarial alignment with burnin noise
    - Validation-based early stopping
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


# ============================================================
# Network
# ============================================================

class SharedEncoder(nn.Module):
    def __init__(self, rep_dim_rna, rep_dim_atac, latent_dim=50,
                 proj_dim=128, h_dim=256, h_depth=2, dropout=0.2):
        super().__init__()
        self.proj_rna = nn.Sequential(
            nn.Linear(rep_dim_rna, proj_dim), nn.LeakyReLU(0.2), nn.BatchNorm1d(proj_dim))
        self.proj_atac = nn.Sequential(
            nn.Linear(rep_dim_atac, proj_dim), nn.LeakyReLU(0.2), nn.BatchNorm1d(proj_dim))
        layers = []
        dim = proj_dim
        for _ in range(h_depth):
            layers += [nn.Linear(dim, h_dim), nn.LeakyReLU(0.2),
                       nn.BatchNorm1d(h_dim), nn.Dropout(dropout)]
            dim = h_dim
        self.body = nn.Sequential(*layers)
        self.loc = nn.Linear(h_dim, latent_dim)
        self.log_std = nn.Linear(h_dim, latent_dim)

    def forward(self, x_rep, modality):
        h = self.proj_rna(x_rep) if modality == 'rna' else self.proj_atac(x_rep)
        h = self.body(h)
        return D.Normal(self.loc(h), F.softplus(self.log_std(h)) + EPS)


class GraphEncoder(nn.Module):
    def __init__(self, n_vertices, latent_dim=50):
        super().__init__()
        self.vrepr = nn.Parameter(torch.randn(n_vertices, latent_dim) * 0.01)
        self.loc = nn.Linear(latent_dim, latent_dim)
        self.std_lin = nn.Linear(latent_dim, latent_dim)

    def forward(self, edge_index, edge_norm, edge_sign):
        src, tgt = edge_index[0], edge_index[1]
        messages = self.vrepr[src] * (edge_sign * edge_norm).unsqueeze(1)
        agg = torch.zeros_like(self.vrepr)
        agg.scatter_add_(0, tgt.unsqueeze(1).expand_as(messages), messages)
        return D.Normal(self.loc(agg), F.softplus(self.std_lin(agg)) + EPS)


class NBDecoder(nn.Module):
    def __init__(self, latent_dim, n_features, n_batches=1):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(n_batches, n_features))
        self.bias = nn.Parameter(torch.zeros(n_batches, n_features))
        self.log_theta = nn.Parameter(torch.zeros(n_batches, n_features))

    def forward(self, u, v, batch_idx, lib_size):
        scale = F.softplus(self.scale[batch_idx])
        logit_mu = scale * (u @ v.T) + self.bias[batch_idx]
        mu = F.softmax(logit_mu, dim=1) * lib_size
        theta = self.log_theta[batch_idx].exp()
        return D.NegativeBinomial(theta, logits=(mu + EPS).log() - torch.log(theta + EPS))


class Discriminator(nn.Module):
    def __init__(self, latent_dim, h_dim=256, h_depth=2, dropout=0.2):
        super().__init__()
        layers = []
        dim = latent_dim
        for _ in range(h_depth):
            layers += [nn.Linear(dim, h_dim), nn.LeakyReLU(0.2), nn.Dropout(dropout)]
            dim = h_dim
        layers.append(nn.Linear(h_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


def spatial_coherence_loss(z_st, spatial_adj, spatial_wt):
    src, tgt = spatial_adj[0], spatial_adj[1]
    diff = z_st[src] - z_st[tgt]
    return (spatial_wt.unsqueeze(1) * diff.pow(2)).mean()


# ============================================================
# Graph edge sampling (like GLUE)
# ============================================================

class GraphEdgeSampler:
    """Sample graph edges with negative sampling per batch."""

    def __init__(self, edge_index, edge_weight, edge_sign, n_vertices,
                 batch_size=10000, neg_samples=10, device='cpu'):
        self.n_vertices = n_vertices
        self.batch_size = batch_size
        self.neg_samples = neg_samples
        self.device = device

        # Only keep positive edges (non-self-loop, non-zero weight)
        mask = (edge_index[0] != edge_index[1]) & (np.abs(edge_weight) > 0.01)
        self.pos_src = edge_index[0, mask]
        self.pos_tgt = edge_index[1, mask]
        self.pos_wt = edge_weight[mask]
        self.pos_sgn = edge_sign[mask]
        self.n_pos = len(self.pos_src)

        # Self-loops (always included)
        self.loop_idx = np.arange(n_vertices)

    def sample(self):
        """Sample a batch of edges: positive edges + negative edges + self-loops."""
        # Sample positive edges
        n_pos = min(self.batch_size, self.n_pos)
        idx = np.random.choice(self.n_pos, n_pos, replace=False)

        # Negative edges: random pairs
        n_neg = n_pos * self.neg_samples
        neg_src = np.random.randint(0, self.n_vertices, n_neg)
        neg_tgt = np.random.randint(0, self.n_vertices, n_neg)

        # Combine: positive + negative + self-loops
        all_src = np.concatenate([self.pos_src[idx], neg_src, self.loop_idx])
        all_tgt = np.concatenate([self.pos_tgt[idx], neg_tgt, self.loop_idx])
        all_wt = np.concatenate([self.pos_wt[idx],
                                 np.zeros(n_neg, dtype=np.float32),
                                 np.ones(self.n_vertices, dtype=np.float32)])
        all_sgn = np.concatenate([self.pos_sgn[idx],
                                  np.ones(n_neg, dtype=np.int8),
                                  np.ones(self.n_vertices, dtype=np.int8)])

        return (torch.tensor(np.stack([all_src, all_tgt]), dtype=torch.long, device=self.device),
                torch.tensor(all_wt, dtype=torch.float32, device=self.device),
                torch.tensor(all_sgn, dtype=torch.float32, device=self.device))


# ============================================================
# Training
# ============================================================

def build_spatial_graph(spatial_coords, k=6):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(spatial_coords)
    dists, indices = nn.kneighbors(spatial_coords)
    median_dist = np.median(dists[:, 1])
    src, tgt, wts = [], [], []
    for i in range(len(spatial_coords)):
        for j in range(1, k + 1):
            src.append(i); tgt.append(indices[i, j])
            wts.append(np.exp(-dists[i, j]**2 / (2 * median_dist**2)))
    return np.array([src, tgt], dtype=np.int64), np.array(wts, dtype=np.float32)


def train_spatial_guided(adata_rna, adata_atac, graph,
                         spatial_coords=None,
                         latent_dim=50, n_epochs=2000, batch_size=256,
                         lr=2e-3, sample_ratio=0.2,
                         lam_align=0.05, lam_spatial=0.1, lam_graph=0.02,
                         align_burnin_frac=0.3, neg_samples=10,
                         device='cuda'):
    from scipy.sparse import issparse
    import scanpy as sc
    import anndata as ad

    # ---- Prepare data (HVG-filtered) ----
    rna_hvg = graph['rna_hvg_mask']
    atac_hvg = graph['atac_hvg_mask']
    rna_names, atac_names = graph['rna_names'], graph['atac_names']

    rna_hvg_names = [rna_names[i] for i in range(len(rna_names)) if rna_hvg[i]]
    rna_mask = np.isin(adata_rna.var_names, rna_hvg_names)
    X_rna = adata_rna.X[:, rna_mask]
    if issparse(X_rna): X_rna = X_rna.toarray()
    X_rna = torch.tensor(X_rna, dtype=torch.float32, device=device)
    rep_rna = torch.tensor(adata_rna.obsm['X_pca'].astype(np.float32), device=device)
    lib_rna = X_rna.sum(dim=1, keepdim=True).clamp(min=1)

    atac_hvg_names = [atac_names[i] for i in range(len(atac_names)) if atac_hvg[i]]
    atac_mask = np.isin(adata_atac.var_names, atac_hvg_names)
    X_atac = adata_atac.X[:, atac_mask]
    if issparse(X_atac): X_atac = X_atac.toarray()
    X_atac = torch.tensor(X_atac, dtype=torch.float32, device=device)
    rep_atac = torch.tensor(adata_atac.obsm['X_lsi'].astype(np.float32), device=device)
    lib_atac = X_atac.sum(dim=1, keepdim=True).clamp(min=1)

    n_st, n_cells = X_rna.shape[0], X_atac.shape[0]
    n_feat_rna, n_feat_atac = X_rna.shape[1], X_atac.shape[1]
    logger.info(f"Data: {n_st} ST ({n_feat_rna} HVG), {n_cells} ATAC ({n_feat_atac} peaks)")

    # ---- Spatial graph ----
    sp_adj_t, sp_wt_t = None, None
    if spatial_coords is not None:
        sp_adj, sp_wt = build_spatial_graph(spatial_coords, k=6)
        sp_adj_t = torch.tensor(sp_adj, dtype=torch.long, device=device)
        sp_wt_t = torch.tensor(sp_wt, dtype=torch.float32, device=device)

    # ---- Graph edge sampler (negative sampling like GLUE) ----
    n_vertices = graph['n_rna'] + graph['n_atac']
    rna_hvg_idx = np.where(rna_hvg)[0]
    atac_hvg_idx = np.where(atac_hvg)[0]

    edge_sampler = GraphEdgeSampler(
        graph['edge_index'], graph['edge_weight'], graph['edge_sign'],
        n_vertices, batch_size=min(10000, graph['edge_index'].shape[1] // 3),
        neg_samples=neg_samples, device=device)

    # ---- Build model ----
    model_kwargs = dict(latent_dim=latent_dim)

    encoder = SharedEncoder(rep_rna.shape[1], rep_atac.shape[1], latent_dim).to(device)
    graph_enc = GraphEncoder(n_vertices, latent_dim).to(device)
    dec_rna = NBDecoder(latent_dim, n_feat_rna).to(device)
    dec_atac = NBDecoder(latent_dim, n_feat_atac).to(device)
    discriminator = Discriminator(latent_dim).to(device)

    rna_graph_idx = torch.tensor(rna_hvg_idx, dtype=torch.long, device=device)
    atac_graph_idx = torch.tensor(graph['n_rna'] + atac_hvg_idx, dtype=torch.long, device=device)

    # Full graph edges for graph convolution (normalized, kept on device)
    full_eidx = torch.tensor(graph['edge_index'], dtype=torch.long, device=device)
    full_ewt = torch.tensor(graph['edge_weight'], dtype=torch.float32, device=device)
    full_esgn = torch.tensor(graph['edge_sign'], dtype=torch.float32, device=device)

    all_params = (list(encoder.parameters()) + list(graph_enc.parameters()) +
                  list(dec_rna.parameters()) + list(dec_atac.parameters()))
    dsc_params = list(discriminator.parameters())
    prior = D.Normal(0., 1.)

    # ---- Training step ----
    def train_step(epoch, gen_optim, dsc_optim, burnin, use_weights=False, dsc_wt_rna=None, dsc_wt_atac=None):
        encoder.train(); graph_enc.train(); dec_rna.train(); dec_atac.train(); discriminator.train()

        # Sample data batches
        ri = torch.randperm(n_st, device=device)[:batch_size]
        ai = torch.randperm(n_cells, device=device)[:batch_size]
        b0_r = torch.zeros(len(ri), dtype=torch.long, device=device)
        b0_a = torch.zeros(len(ai), dtype=torch.long, device=device)

        # Encode
        u_rna = encoder(rep_rna[ri], 'rna')
        u_atac = encoder(rep_atac[ai], 'atac')

        # Discriminator with burnin noise
        u_cat = torch.cat([u_rna.mean, u_atac.mean])
        anneal = max(1 - epoch / burnin, 0) if burnin > 0 else 0
        if anneal > 0:
            noise = torch.randn_like(u_cat) * u_cat.std(dim=0, keepdim=True)
            u_cat_noisy = u_cat + anneal * 1.5 * noise
        else:
            u_cat_noisy = u_cat

        labels = torch.cat([torch.zeros(len(ri), dtype=torch.long, device=device),
                             torch.ones(len(ai), dtype=torch.long, device=device)])

        # Discriminator step
        dsc_logits = discriminator(u_cat_noisy.detach())
        if use_weights and dsc_wt_rna is not None:
            w = torch.cat([dsc_wt_rna[ri], dsc_wt_atac[ai]])
            dsc_loss = (F.cross_entropy(dsc_logits, labels, reduction='none') * w).sum() / w.numel()
        else:
            dsc_loss = F.cross_entropy(dsc_logits, labels)

        dsc_optim.zero_grad()
        (lam_align * dsc_loss).backward()
        dsc_optim.step()

        # Re-encode for generator step (fresh computation graph)
        u_rna = encoder(rep_rna[ri], 'rna')
        u_atac = encoder(rep_atac[ai], 'atac')
        z_rna = u_rna.rsample()
        z_atac = u_atac.rsample()

        # Graph encoder (full graph for convolution, sampled edges for loss)
        v_dist = graph_enc(full_eidx, full_ewt, full_esgn)
        v = v_dist.rsample()

        # Reconstruction
        v_rna = v[rna_graph_idx]
        v_atac = v[atac_graph_idx]

        rna_nll = -dec_rna(z_rna, v_rna, b0_r, lib_rna[ri]).log_prob(X_rna[ri]).nanmean()
        atac_lp = dec_atac(z_atac, v_atac, b0_a, lib_atac[ai]).log_prob(X_atac[ai])
        if n_feat_atac > 2000:
            n_s = max(int(n_feat_atac * sample_ratio), 500)
            si = torch.randperm(n_feat_atac, device=device)[:n_s]
            atac_nll = -(atac_lp[:, si].nanmean()) * (n_feat_atac / n_s)
        else:
            atac_nll = -atac_lp.nanmean()

        rna_kl = D.kl_divergence(u_rna, prior).sum(1).mean() / n_feat_rna
        atac_kl = D.kl_divergence(u_atac, prior).sum(1).mean() / n_feat_atac

        # Graph loss (sampled edges with negative sampling)
        s_eidx, s_ewt, s_esgn = edge_sampler.sample()
        g_logits = s_esgn * (v[s_eidx[0]] * v[s_eidx[1]]).sum(dim=1)
        g_tgt = s_ewt.abs().clamp(0, 1)
        g_nll_all = F.binary_cross_entropy_with_logits(g_logits, g_tgt, reduction='none')
        pm = (g_tgt > 0).long()
        np_ = pm.sum().item(); nn_ = pm.numel() - np_
        g_pn = torch.zeros(2, dtype=g_nll_all.dtype, device=device)
        g_pn.scatter_add_(0, pm, g_nll_all)
        ac = (np_ > 0) + (nn_ > 0)
        g_nll = (g_pn[0] / max(nn_, 1) + g_pn[1] / max(np_, 1)) / ac
        g_kl = D.kl_divergence(v_dist, prior).sum(1).mean() / v.shape[0]

        # Adversarial (generator wants to fool discriminator)
        u_cat2 = torch.cat([u_rna.mean, u_atac.mean])
        if anneal > 0:
            noise2 = torch.randn_like(u_cat2) * u_cat2.std(dim=0, keepdim=True)
            u_cat2 = u_cat2 + anneal * 1.5 * noise2
        dsc_logits2 = discriminator(u_cat2)
        if use_weights and dsc_wt_rna is not None:
            w2 = torch.cat([dsc_wt_rna[ri], dsc_wt_atac[ai]])
            dsc_loss2 = (F.cross_entropy(dsc_logits2, labels, reduction='none') * w2).sum() / w2.numel()
        else:
            dsc_loss2 = F.cross_entropy(dsc_logits2, labels)

        # Spatial coherence
        sp_val = torch.tensor(0.0, device=device)
        if sp_adj_t is not None:
            u_all_st = encoder(rep_rna, 'rna')
            sp_val = spatial_coherence_loss(u_all_st.mean, sp_adj_t, sp_wt_t)

        vae_loss = 0.5 * ((rna_nll + rna_kl) + (atac_nll + atac_kl)) + lam_graph * 2 * (g_nll + g_kl)
        gen_loss = vae_loss - lam_align * dsc_loss2 + lam_spatial * sp_val

        gen_optim.zero_grad()
        gen_loss.backward()
        nn.utils.clip_grad_norm_(all_params, 1.0)
        gen_optim.step()

        return {'vae': vae_loss.item(), 'rna': rna_nll.item(), 'atac': atac_nll.item(),
                'dsc': dsc_loss.item(), 'sp': sp_val.item(), 'g': g_nll.item()}

    # ---- Phase runner ----
    def run_phase(name, max_ep, burnin, use_weights=False, dwt_r=None, dwt_a=None):
        _gen_opt = torch.optim.RMSprop(all_params, lr=lr)
        _dsc_opt = torch.optim.RMSprop(dsc_params, lr=lr)
        _sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            _gen_opt, patience=max(max_ep // 10, 15), factor=0.5, min_lr=1e-5)
        best, pat = float('inf'), 0
        patience = max(max_ep // 8, 20)

        for ep in range(max_ep):
            m = train_step(ep, _gen_opt, _dsc_opt, burnin, use_weights, dwt_r, dwt_a)
            _sched.step(m['vae'])
            if m['vae'] < best: best = m['vae']; pat = 0
            else: pat += 1
            if pat >= patience and ep > burnin:
                logger.info(f"  {name} early stop at epoch {ep}")
                break
            if ep % 50 == 0:
                logger.info(f"  {name} {ep:4d}: vae={m['vae']:.4f} rna={m['rna']:.3f} "
                            f"atac={m['atac']:.3f} dsc={m['dsc']:.3f} sp={m['sp']:.4f}")

    # ---- Estimate balancing weights ----
    def estimate_weights():
        encoder.eval()
        with torch.no_grad():
            z_r = encoder(rep_rna, 'rna').mean.cpu().numpy()
            z_a = encoder(rep_atac, 'atac').mean.cpu().numpy()

        from sklearn.preprocessing import normalize as sk_norm
        weights = {}
        for label, z in [('rna', z_r), ('atac', z_a)]:
            tmp = ad.AnnData(obsm={'X': z})
            sc.pp.neighbors(tmp, use_rep='X', metric='cosine')
            sc.tl.leiden(tmp, resolution=1.0)
            clusters = tmp.obs['leiden'].values.astype(int)
            centroids = np.array([z[clusters == c].mean(0) for c in range(clusters.max() + 1)])
            weights[label] = (clusters, sk_norm(centroids, norm='l2'))

        cos = weights['rna'][1] @ weights['atac'][1].T
        cos[cos < 0.5] = 0; cos = cos ** 4

        result = {}
        for side, (clusters, _), axis in [('rna', weights['rna'], 1), ('atac', weights['atac'], 0)]:
            n_per = np.array([(clusters == c).sum() for c in range(clusters.max() + 1)])
            bal = cos.sum(axis=axis) / n_per
            bal = bal / (bal.sum() / len(bal))
            w_per_cell = bal[clusters]
            result[side] = torch.tensor(w_per_cell, dtype=torch.float32, device=device)

        logger.info(f"Balancing: RNA [{result['rna'].min():.2f},{result['rna'].max():.2f}], "
                     f"ATAC [{result['atac'].min():.2f},{result['atac'].max():.2f}]")
        return result['rna'], result['atac']

    # ==== EXECUTE ====
    burnin_ep = int(n_epochs * align_burnin_frac)

    # Phase 1: Pretrain
    logger.info(f"Phase 1: Pretrain (burnin={burnin_ep})")
    run_phase("Pretrain", n_epochs, burnin_ep)

    # Estimate balancing weights
    dwt_r, dwt_a = estimate_weights()

    # Phase 2: Finetune with balancing weights
    ft_burnin = max(burnin_ep // 8, 30)
    logger.info(f"Phase 2: Finetune (burnin={ft_burnin}, with balancing weights)")
    run_phase("Finetune", n_epochs, ft_burnin, use_weights=True, dwt_r=dwt_r, dwt_a=dwt_a)

    # Extract embeddings
    encoder.eval()
    with torch.no_grad():
        z_rna_out = encoder(rep_rna, 'rna').mean.cpu().numpy()
        z_atac_out = encoder(rep_atac, 'atac').mean.cpu().numpy()
    logger.info(f"Embeddings: RNA {z_rna_out.shape}, ATAC {z_atac_out.shape}")
    return None, z_rna_out, z_atac_out
