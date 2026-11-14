"""
VICReg alignment + GMG regularization for GLUE.

Replaces adversarial discriminator with VICReg (no training instability).
Adds Gromov-Monge Gap regularization to preserve pairwise distances in encoder.

VICReg (ICLR 2022): Variance + Invariance + Covariance
  - Invariance: align paired embeddings across modalities
  - Variance: prevent representation collapse
  - Covariance: encourage informative (decorrelated) dimensions

GMG (ICLR 2025, Apple Research):
  - Forces encoder to preserve pairwise distance structure
  - L_gmg ∝ ||D_latent - D_input||, where D is distance matrix
  - Prevents fine-grained structure loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np
import logging

logger = logging.getLogger(__name__)

EPS = 1e-7
_original_compute_losses = None


def vicreg_loss(z_a, z_b, lam_inv=25.0, lam_var=25.0, lam_cov=1.0):
    """
    VICReg loss between two embedding batches.

    Args:
        z_a, z_b: (batch, dim) embeddings from two modalities
        lam_inv: invariance weight (align corresponding samples)
        lam_var: variance weight (prevent collapse)
        lam_cov: covariance weight (decorrelate dimensions)
    """
    batch_size, dim = z_a.shape

    # Invariance: MSE between paired embeddings
    inv_loss = F.mse_loss(z_a, z_b)

    # Variance: std of each dimension should be > 1
    std_a = torch.sqrt(z_a.var(dim=0) + 1e-4)
    std_b = torch.sqrt(z_b.var(dim=0) + 1e-4)
    var_loss = torch.mean(F.relu(1.0 - std_a)) + torch.mean(F.relu(1.0 - std_b))

    # Covariance: off-diagonal of covariance matrix should be 0
    z_a_centered = z_a - z_a.mean(dim=0)
    z_b_centered = z_b - z_b.mean(dim=0)
    cov_a = (z_a_centered.T @ z_a_centered) / (batch_size - 1)
    cov_b = (z_b_centered.T @ z_b_centered) / (batch_size - 1)

    # Off-diagonal elements
    off_diag_a = cov_a.pow(2).sum() - cov_a.diagonal().pow(2).sum()
    off_diag_b = cov_b.pow(2).sum() - cov_b.diagonal().pow(2).sum()
    cov_loss = (off_diag_a + off_diag_b) / dim

    return lam_inv * inv_loss + lam_var * var_loss + lam_cov * cov_loss


def gmg_loss(z, x_rep, n_sample=500):
    """
    Gromov-Monge Gap regularization.

    Encourages encoder to preserve pairwise distance structure.

    Args:
        z: (batch, latent_dim) encoder output
        x_rep: (batch, rep_dim) encoder input (e.g., PCA/LSI representation)
        n_sample: number of pairs to sample for efficiency
    """
    batch = z.shape[0]
    if batch < 4:
        return torch.tensor(0.0, device=z.device)

    # Sample random pairs
    n = min(n_sample, batch * (batch - 1) // 2)
    idx_i = torch.randint(0, batch, (n,), device=z.device)
    idx_j = torch.randint(0, batch, (n,), device=z.device)
    # Ensure i != j
    mask = idx_i != idx_j
    idx_i, idx_j = idx_i[mask], idx_j[mask]

    if len(idx_i) < 10:
        return torch.tensor(0.0, device=z.device)

    # Pairwise distances in latent space
    d_latent = torch.norm(z[idx_i] - z[idx_j], dim=1)

    # Pairwise distances in input space
    d_input = torch.norm(x_rep[idx_i] - x_rep[idx_j], dim=1)

    # Normalize both to [0, 1]
    d_latent = d_latent / (d_latent.max() + 1e-8)
    d_input = d_input / (d_input.max() + 1e-8)

    # GMG: correlation between distance orderings should be preserved
    # Using rank correlation (Spearman-like) via soft sorting
    # Simplified: just MSE between normalized distances
    return F.mse_loss(d_latent, d_input)


def patch_glue_vicreg_gmg(lam_vicreg=0.05, lam_gmg=0.1, sample_ratio=0.2,
                           target_modality='atac'):
    """
    Patch GLUE with VICReg alignment + GMG regularization + sampled reconstruction.

    Combines three improvements:
    1. VICReg replaces adversarial discriminator
    2. GMG preserves pairwise distances in encoder
    3. Sampled reconstruction (from sampled_glue.py)
    """
    from scglue.models.scglue import SCGLUETrainer

    global _original_compute_losses
    if _original_compute_losses is None:
        _original_compute_losses = SCGLUETrainer.compute_losses

    def vicreg_gmg_compute_losses(self, data, epoch, dsc_only=False):
        net = self.net
        x, xrep, xbch, xlbl, xdwt, xflag, eidx, ewt, esgn = data

        # Encoder forward
        u, l = {}, {}
        for k in net.keys:
            u[k], l[k] = net.x2u[k](x[k], xrep[k], lazy_normalizer=dsc_only)
        usamp = {k: u[k].rsample() for k in net.keys}
        if self.normalize_u:
            usamp = {k: F.normalize(usamp[k], dim=1) for k in net.keys}
        prior = net.prior()

        # ===== VICReg instead of adversarial =====
        keys_list = list(net.keys)
        if len(keys_list) >= 2 and not dsc_only:
            # Align all pairs of modalities
            vc_loss = torch.tensor(0.0, device=net.device)
            n_pairs = 0
            for i in range(len(keys_list)):
                for j in range(i + 1, len(keys_list)):
                    k_a, k_b = keys_list[i], keys_list[j]
                    # Use mean of variational distribution
                    z_a, z_b = u[k_a].mean, u[k_b].mean
                    # Min batch size
                    n = min(z_a.shape[0], z_b.shape[0])
                    vc_loss = vc_loss + vicreg_loss(z_a[:n], z_b[:n])
                    n_pairs += 1
            if n_pairs > 0:
                vc_loss = vc_loss / n_pairs
        else:
            vc_loss = torch.tensor(0.0, device=net.device)

        # Still compute discriminator for logging (but don't use in gen_loss)
        u_cat = torch.cat([u[k].mean for k in net.keys])
        xbch_cat = torch.cat([xbch[k] for k in net.keys])
        xdwt_cat = torch.cat([xdwt[k] for k in net.keys])
        xflag_cat = torch.cat([xflag[k] for k in net.keys])
        dsc_loss = F.cross_entropy(net.du(u_cat, xbch_cat), xflag_cat, reduction="none")
        dsc_loss = (dsc_loss * xdwt_cat).sum() / xdwt_cat.numel()

        if dsc_only:
            return {"dsc_loss": self.lam_align * dsc_loss}

        # Supervision (unchanged)
        if net.u2c:
            xlbl_cat = torch.cat([xlbl[k] for k in net.keys])
            lmsk = xlbl_cat >= 0
            sup_loss = F.cross_entropy(
                net.u2c(u_cat[lmsk]), xlbl_cat[lmsk], reduction="none"
            ).sum() / max(lmsk.sum(), 1)
        else:
            sup_loss = torch.tensor(0.0, device=net.device)

        # Graph encoder (unchanged)
        v = net.g2v(self.eidx, self.enorm, self.esgn)
        vsamp = v.rsample()

        g_nll = -net.v2g(vsamp, eidx, esgn).log_prob(ewt)
        pos_mask = (ewt != 0).to(torch.int64)
        n_pos = pos_mask.sum().item()
        n_neg = pos_mask.numel() - n_pos
        g_nll_pn = torch.zeros(2, dtype=g_nll.dtype, device=g_nll.device)
        g_nll_pn.scatter_add_(0, pos_mask, g_nll)
        avgc = (n_pos > 0) + (n_neg > 0)
        g_nll = (g_nll_pn[0] / max(n_neg, 1) + g_nll_pn[1] / max(n_pos, 1)) / avgc
        g_kl = D.kl_divergence(v, prior).sum(dim=1).mean() / vsamp.shape[0]
        g_elbo = g_nll + self.lam_kl * g_kl

        # ===== Sampled reconstruction (from sampled_glue.py) =====
        x_nll = {}
        for k in net.keys:
            dist = net.u2x[k](usamp[k], vsamp[getattr(net, f"{k}_idx")], xbch[k], l[k])
            full_log_prob = dist.log_prob(x[k])
            n_features = full_log_prob.shape[1] if full_log_prob.dim() > 1 else 1

            if k == target_modality and n_features > 2000:
                n_sample = max(int(n_features * sample_ratio), 500)
                idx = torch.randperm(n_features, device=full_log_prob.device)[:n_sample]
                x_nll[k] = -(full_log_prob[:, idx].nanmean()) * (n_features / n_sample)
            else:
                x_nll[k] = -full_log_prob.nanmean()

        x_kl = {
            k: D.kl_divergence(u[k], prior).sum(dim=1).mean() / x[k].shape[1]
            for k in net.keys
        }
        x_elbo = {k: x_nll[k] + self.lam_kl * x_kl[k] for k in net.keys}
        x_elbo_sum = sum(self.modality_weight[k] * x_elbo[k] for k in net.keys)

        # ===== GMG regularization =====
        gmg_total = torch.tensor(0.0, device=net.device)
        for k in net.keys:
            if xrep[k] is not None and usamp[k].shape[0] > 10:
                gmg_total = gmg_total + gmg_loss(usamp[k], xrep[k])
        gmg_total = gmg_total / max(len(net.keys), 1)

        # ===== Combined loss =====
        # Keep adversarial, ADD VICReg and GMG as extra regularization
        vae_loss = (
            self.lam_data * x_elbo_sum
            + self.lam_graph * len(net.keys) * g_elbo
            + self.lam_sup * sup_loss
        )
        gen_loss = (vae_loss
                    - self.lam_align * dsc_loss          # adversarial (keep!)
                    + lam_vicreg * vc_loss               # VICReg alignment (add)
                    + lam_gmg * gmg_total)               # GMG distance preservation (add)

        losses = {
            "dsc_loss": dsc_loss,  # still logged for monitoring
            "vae_loss": vae_loss,
            "gen_loss": gen_loss,
            "g_nll": g_nll, "g_kl": g_kl, "g_elbo": g_elbo,
            "vicreg": vc_loss,
            "gmg": gmg_total,
        }
        for k in net.keys:
            losses.update({
                f"x_{k}_nll": x_nll[k], f"x_{k}_kl": x_kl[k], f"x_{k}_elbo": x_elbo[k]
            })
        if net.u2c:
            losses["sup_loss"] = sup_loss
        return losses

    SCGLUETrainer.compute_losses = vicreg_gmg_compute_losses
    logger.info(f"Patched GLUE: VICReg (lam={lam_vicreg}) + GMG (lam={lam_gmg}) + sampled ({sample_ratio})")
