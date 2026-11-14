"""
Sampled reconstruction loss for GLUE — solves the decoder bottleneck.

Instead of reconstructing ALL 10k+ ATAC peaks every batch,
randomly sample a subset (e.g., 20%) and scale the loss.
Unbiased gradient estimator (SDGD theory).

Usage:
    from pycmap.multi.sampled_glue import patch_glue_sampled_loss
    patch_glue_sampled_loss(sample_ratio=0.2)  # before calling fit_SCGLUE
"""

import torch
import logging

logger = logging.getLogger(__name__)

_original_compute_losses = None


def patch_glue_sampled_loss(sample_ratio=0.2, target_modality='atac'):
    """
    Monkey-patch GLUE's compute_losses to use sampled reconstruction.

    Args:
        sample_ratio: Fraction of features to reconstruct per batch (0.2 = 20%)
        target_modality: Which modality to sample ('atac' or apply to all)
    """
    from scglue.models.scglue import SCGLUETrainer
    import torch.distributions as D
    import torch.nn.functional as F
    from typing import Mapping

    global _original_compute_losses
    if _original_compute_losses is None:
        _original_compute_losses = SCGLUETrainer.compute_losses

    def sampled_compute_losses(self, data, epoch, dsc_only=False):
        net = self.net
        x, xrep, xbch, xlbl, xdwt, xflag, eidx, ewt, esgn = data

        # Encoder forward (unchanged)
        u, l = {}, {}
        for k in net.keys:
            u[k], l[k] = net.x2u[k](x[k], xrep[k], lazy_normalizer=dsc_only)
        usamp = {k: u[k].rsample() for k in net.keys}
        if self.normalize_u:
            usamp = {k: F.normalize(usamp[k], dim=1) for k in net.keys}
        prior = net.prior()

        # Discriminator (unchanged)
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

        # Supervision (unchanged)
        if net.u2c:
            xlbl_cat = torch.cat([xlbl[k] for k in net.keys])
            lmsk = xlbl_cat >= 0
            sup_loss = F.cross_entropy(
                net.u2c(u_cat[lmsk]), xlbl_cat[lmsk], reduction="none"
            ).sum() / max(lmsk.sum(), 1)
        else:
            sup_loss = torch.tensor(0.0, device=self.net.device)

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

        # ===== SAMPLED RECONSTRUCTION (the key change) =====
        x_nll = {}
        for k in net.keys:
            # Decoder forward: get the full distribution
            dist = net.u2x[k](usamp[k], vsamp[getattr(net, f"{k}_idx")], xbch[k], l[k])

            # Full log_prob for all features
            full_log_prob = dist.log_prob(x[k])  # (batch, n_features)

            n_features = full_log_prob.shape[1] if full_log_prob.dim() > 1 else 1

            # Only sample for the target modality with many features
            if k == target_modality and n_features > 2000:
                # Random sample of feature dimensions
                n_sample = max(int(n_features * sample_ratio), 500)
                idx = torch.randperm(n_features, device=full_log_prob.device)[:n_sample]
                sampled_log_prob = full_log_prob[:, idx]
                # Scale by D/k for unbiased estimate
                x_nll[k] = -(sampled_log_prob.nanmean()) * (n_features / n_sample)
            else:
                x_nll[k] = -full_log_prob.nanmean()

        x_kl = {
            k: D.kl_divergence(u[k], prior).sum(dim=1).mean() / x[k].shape[1]
            for k in net.keys
        }
        x_elbo = {k: x_nll[k] + self.lam_kl * x_kl[k] for k in net.keys}
        x_elbo_sum = sum(self.modality_weight[k] * x_elbo[k] for k in net.keys)

        vae_loss = (
            self.lam_data * x_elbo_sum
            + self.lam_graph * len(net.keys) * g_elbo
            + self.lam_sup * sup_loss
        )
        gen_loss = vae_loss - self.lam_align * dsc_loss

        losses = {
            "dsc_loss": dsc_loss, "vae_loss": vae_loss, "gen_loss": gen_loss,
            "g_nll": g_nll, "g_kl": g_kl, "g_elbo": g_elbo,
        }
        for k in net.keys:
            losses.update({
                f"x_{k}_nll": x_nll[k], f"x_{k}_kl": x_kl[k], f"x_{k}_elbo": x_elbo[k]
            })
        if net.u2c:
            losses["sup_loss"] = sup_loss
        return losses

    SCGLUETrainer.compute_losses = sampled_compute_losses
    logger.info(f"Patched GLUE: sampled reconstruction (ratio={sample_ratio}, target={target_modality})")


def unpatch_glue():
    """Restore original GLUE compute_losses."""
    global _original_compute_losses
    if _original_compute_losses is not None:
        from scglue.models.scglue import SCGLUETrainer
        SCGLUETrainer.compute_losses = _original_compute_losses
        _original_compute_losses = None
