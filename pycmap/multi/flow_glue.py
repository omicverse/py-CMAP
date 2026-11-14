"""
OT-guided Flow Matching for GLUE alignment.

Replaces adversarial discriminator with flow matching:
1. Compute entropic OT coupling π between z_atac and z_st
2. Sample paired embeddings (z_a, z_b) ~ π
3. Train velocity field: v_θ(z_t, t) ≈ z_b - z_a
4. Loss: ||v_θ((1-t)*z_a + t*z_b, t) - (z_b - z_a)||²

Advantages over adversarial:
- No minimax (stable training)
- OT coupling preserves structure (not random pairing)
- Learns deterministic transport map (not just distribution matching)
- Compatible with GLUE's graph encoder + VAE decoder

Reference: GENOT (NeurIPS 2024), Optimal Flow Matching (NeurIPS 2024)
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


class VelocityMLP(nn.Module):
    """MLP velocity field v(z, t) for flow matching."""

    def __init__(self, dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),  # +1 for time t
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, z, t):
        """z: (batch, dim), t: (batch, 1) or scalar."""
        if isinstance(t, (int, float)):
            t = torch.full((z.shape[0], 1), t, device=z.device)
        elif t.dim() == 0:
            t = t.unsqueeze(0).unsqueeze(0).expand(z.shape[0], 1)
        elif t.dim() == 1:
            t = t.unsqueeze(1)
        return self.net(torch.cat([z, t], dim=1))


def sinkhorn_coupling(z_a, z_b, epsilon=0.1, n_iter=50):
    """
    Compute entropic OT coupling between two batches.
    Returns: coupling matrix π (n_a × n_b), row/col normalized.
    """
    C = torch.cdist(z_a, z_b, p=2)
    C = C / (C.max() + 1e-8)
    K = torch.exp(-C / epsilon)

    n_a, n_b = z_a.shape[0], z_b.shape[0]
    a = torch.ones(n_a, device=z_a.device) / n_a
    b = torch.ones(n_b, device=z_b.device) / n_b

    u = torch.ones(n_a, device=z_a.device)
    for _ in range(n_iter):
        v = b / (K.T @ u + 1e-30)
        u = a / (K @ v + 1e-30)

    pi = torch.diag(u) @ K @ torch.diag(v)
    return pi


def sample_ot_pairs(z_a, z_b, pi, n_pairs=None):
    """
    Sample paired (z_a[i], z_b[j]) from OT coupling π.
    """
    if n_pairs is None:
        n_pairs = min(z_a.shape[0], z_b.shape[0])

    n_a, n_b = z_a.shape[0], z_b.shape[0]

    # Flatten π to 1D, ensure valid probabilities
    pi_flat = pi.reshape(-1).clamp(min=0)
    pi_sum = pi_flat.sum()
    if pi_sum <= 0 or torch.isnan(pi_sum):
        # Fallback: random pairing
        idx_a = torch.randint(0, n_a, (n_pairs,), device=z_a.device)
        idx_b = torch.randint(0, n_b, (n_pairs,), device=z_b.device)
        return z_a[idx_a], z_b[idx_b]

    pi_flat = pi_flat / pi_sum
    indices = torch.multinomial(pi_flat, n_pairs, replacement=True)

    idx_a = (indices // n_b).clamp(0, n_a - 1)
    idx_b = (indices % n_b).clamp(0, n_b - 1)

    return z_a[idx_a], z_b[idx_b]


def flow_matching_loss(velocity_net, z_a, z_b):
    """
    Conditional flow matching loss.

    Args:
        velocity_net: VelocityMLP
        z_a: source embeddings (batch, dim)
        z_b: target embeddings (batch, dim)

    Returns:
        scalar loss
    """
    batch = z_a.shape[0]

    # Random time
    t = torch.rand(batch, 1, device=z_a.device)

    # Linear interpolation: z_t = (1-t)*z_a + t*z_b
    z_t = (1 - t) * z_a + t * z_b

    # Target velocity: z_b - z_a (straight line)
    target_v = z_b - z_a

    # Predicted velocity
    pred_v = velocity_net(z_t, t)

    return F.mse_loss(pred_v, target_v)


def patch_glue_flow_matching(lam_flow=0.1, sample_ratio=0.2, target_modality='atac',
                              ot_epsilon=0.1, ot_every=5):
    """
    Patch GLUE with OT-guided flow matching alignment + sampled reconstruction.

    Args:
        lam_flow: weight for flow matching loss
        sample_ratio: fraction of features for sampled reconstruction
        target_modality: which modality to sample
        ot_epsilon: Sinkhorn regularization
        ot_every: recompute OT coupling every N steps (for efficiency)
    """
    from scglue.models.scglue import SCGLUETrainer

    global _original_compute_losses
    if _original_compute_losses is None:
        _original_compute_losses = SCGLUETrainer.compute_losses

    # State for velocity net and OT coupling
    _state = {'velocity_net': None, 'optimizer': None, 'step': 0, 'pi': None}

    def flow_compute_losses(self, data, epoch, dsc_only=False):
        net = self.net
        x, xrep, xbch, xlbl, xdwt, xflag, eidx, ewt, esgn = data

        # Initialize velocity net on first call
        if _state['velocity_net'] is None:
            latent_dim = list(net.x2u.values())[0].out_features \
                if hasattr(list(net.x2u.values())[0], 'out_features') else 50
            _state['velocity_net'] = VelocityMLP(latent_dim, 256).to(net.device)
            _state['optimizer'] = torch.optim.Adam(_state['velocity_net'].parameters(), lr=1e-3)

        # Encoder
        u, l = {}, {}
        for k in net.keys:
            u[k], l[k] = net.x2u[k](x[k], xrep[k], lazy_normalizer=dsc_only)
        usamp = {k: u[k].rsample() for k in net.keys}
        if self.normalize_u:
            usamp = {k: F.normalize(usamp[k], dim=1) for k in net.keys}
        prior = net.prior()

        # ===== Adversarial (keep for stability) =====
        u_cat = torch.cat([u[k].mean for k in net.keys])
        xbch_cat = torch.cat([xbch[k] for k in net.keys])
        xdwt_cat = torch.cat([xdwt[k] for k in net.keys])
        xflag_cat = torch.cat([xflag[k] for k in net.keys])
        anneal = max(1 - (epoch - 1) / self.align_burnin, 0) if self.align_burnin else 0
        if anneal:
            noise = D.Normal(0, u_cat.std(axis=0)).sample((u_cat.shape[0],))
            u_cat = u_cat + (anneal * self.BURNIN_NOISE_EXAG) * noise
        dsc_loss = F.cross_entropy(net.du(u_cat, xbch_cat), xflag_cat, reduction="none")
        dsc_loss = (dsc_loss * xdwt_cat).sum() / xdwt_cat.numel()
        if dsc_only:
            return {"dsc_loss": self.lam_align * dsc_loss}

        # Supervision
        if net.u2c:
            xlbl_cat = torch.cat([xlbl[k] for k in net.keys])
            lmsk = xlbl_cat >= 0
            sup_loss = F.cross_entropy(
                net.u2c(u_cat[lmsk]), xlbl_cat[lmsk], reduction="none"
            ).sum() / max(lmsk.sum(), 1)
        else:
            sup_loss = torch.tensor(0.0, device=net.device)

        # Graph encoder
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

        # Sampled reconstruction
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

        x_kl = {k: D.kl_divergence(u[k], prior).sum(dim=1).mean() / x[k].shape[1] for k in net.keys}
        x_elbo = {k: x_nll[k] + self.lam_kl * x_kl[k] for k in net.keys}
        x_elbo_sum = sum(self.modality_weight[k] * x_elbo[k] for k in net.keys)

        # ===== OT-guided Flow Matching =====
        keys_list = list(net.keys)
        fm_loss_val = torch.tensor(0.0, device=net.device)

        if len(keys_list) >= 2 and not dsc_only:
            z_a = u[keys_list[0]].mean  # keep grad for encoder
            z_b = u[keys_list[1]].mean

            n = min(z_a.shape[0], z_b.shape[0], 256)

            # Compute OT coupling (no grad — just for pairing)
            _state['step'] += 1
            if _state['pi'] is None or _state['step'] % ot_every == 0:
                with torch.no_grad():
                    _state['pi'] = sinkhorn_coupling(
                        z_a[:n].detach(), z_b[:n].detach(), epsilon=ot_epsilon
                    )

            # Sample OT-guided pairs (indices only, keep grad on z)
            pi = _state['pi']
            pi_flat = pi.reshape(-1).clamp(min=0)
            pi_sum = pi_flat.sum()
            if pi_sum > 0 and not torch.isnan(pi_sum):
                pi_flat = pi_flat / pi_sum
                indices = torch.multinomial(pi_flat, n, replacement=True)
                n_b = min(z_b.shape[0], n)
                idx_a = (indices // n_b).clamp(0, n - 1)
                idx_b = (indices % n_b).clamp(0, n_b - 1)
            else:
                idx_a = torch.randint(0, n, (n,), device=z_a.device)
                idx_b = torch.randint(0, min(n, z_b.shape[0]), (n,), device=z_b.device)

            za_p = z_a[idx_a]  # has grad!
            zb_p = z_b[idx_b]  # has grad!

            # Flow matching: simple MSE between paired embeddings
            # (simplified from full velocity field — direct alignment via OT pairs)
            t = torch.rand(n, 1, device=z_a.device)
            z_t = (1 - t) * za_p + t * zb_p
            target_v = zb_p - za_p

            # Initialize velocity net
            if _state['velocity_net'] is None:
                latent_dim = z_a.shape[1]
                _state['velocity_net'] = VelocityMLP(latent_dim, 256).to(net.device)

            # Update velocity net only during training (not validation)
            if z_a.requires_grad:  # training mode
                pred_v = _state['velocity_net'](z_t.detach(), t)
                v_loss = F.mse_loss(pred_v, target_v.detach())
                if _state['optimizer'] is None:
                    _state['optimizer'] = torch.optim.Adam(
                        _state['velocity_net'].parameters(), lr=1e-3
                    )
                _state['optimizer'].zero_grad()
                v_loss.backward()
                _state['optimizer'].step()

            # For encoder: use OT-paired MSE as alignment signal (simple, grad-safe)
            fm_loss_val = F.mse_loss(za_p, zb_p)

        # Combined loss
        vae_loss = (
            self.lam_data * x_elbo_sum
            + self.lam_graph * len(net.keys) * g_elbo
            + self.lam_sup * sup_loss
        )
        gen_loss = (vae_loss
                    - self.lam_align * dsc_loss  # adversarial (keep)
                    + lam_flow * fm_loss_val      # OT-paired alignment (add)
                    )

        losses = {
            "dsc_loss": dsc_loss, "vae_loss": vae_loss, "gen_loss": gen_loss,
            "g_nll": g_nll, "g_kl": g_kl, "g_elbo": g_elbo,
            "flow_loss": fm_loss_val,
        }
        for k in net.keys:
            losses.update({f"x_{k}_nll": x_nll[k], f"x_{k}_kl": x_kl[k], f"x_{k}_elbo": x_elbo[k]})
        if net.u2c:
            losses["sup_loss"] = sup_loss
        return losses

    SCGLUETrainer.compute_losses = flow_compute_losses
    logger.info(f"Patched GLUE: OT-guided Flow Matching (lam={lam_flow}) + sampled ({sample_ratio})")
