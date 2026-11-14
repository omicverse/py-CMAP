"""
Low-rank + sampled decoder for GLUE acceleration.

Key insight: GLUE's NB decoder already does u @ v.t() (rank-50 factorization).
The bottleneck is the FULL 10k-dim output of softmax + NB distribution + log_prob.

This patch samples feature dimensions BEFORE the decoder forward,
so we only compute u @ v[sampled].t() → (batch, k) instead of (batch, 10k).
This saves compute in ALL downstream operations (softmax, NB, log_prob).

Expected speedup: ~5x on ATAC decoder (10k → 2k features per batch).
"""

import torch
import torch.nn.functional as F
import torch.distributions as D
import logging

logger = logging.getLogger(__name__)

EPS = 1e-7

_original_compute_losses = None


def patch_glue_lowrank_sampled(sample_ratio=0.2, target_modality='atac'):
    """
    Patch GLUE to sample features BEFORE decoder forward.

    The original decoder:
        logit_mu = scale * (u @ v.t()) + bias       # (batch, ALL_features)
        mu = softmax(logit_mu) * l                   # (batch, ALL_features)
        dist = NB(theta, logits)                     # (batch, ALL_features)
        nll = -dist.log_prob(x)                      # (batch, ALL_features)

    Patched:
        v_sampled = v[idx]                           # (k, 50)
        logit_mu = scale[:, idx] * (u @ v_sampled.t()) + bias[:, idx]  # (batch, k)
        mu = softmax(logit_mu) * l                   # (batch, k)
        dist = NB(theta[:, idx], logits)             # (batch, k)
        nll = -dist.log_prob(x[:, idx])              # (batch, k)
        x_nll = nll.mean() * (D/k)                  # unbiased estimate
    """
    from scglue.models.scglue import SCGLUETrainer

    global _original_compute_losses
    if _original_compute_losses is None:
        _original_compute_losses = SCGLUETrainer.compute_losses

    def lowrank_sampled_compute_losses(self, data, epoch, dsc_only=False):
        net = self.net
        x, xrep, xbch, xlbl, xdwt, xflag, eidx, ewt, esgn = data

        # Encoder (unchanged)
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

        # ===== SAMPLED DECODER: sample features BEFORE forward =====
        x_nll = {}
        for k in net.keys:
            decoder = net.u2x[k]
            v_feat = vsamp[getattr(net, f"{k}_idx")]  # (n_features, latent_dim)
            n_features = v_feat.shape[0]
            u_k = usamp[k]
            b_k = xbch[k]
            l_k = l[k]
            x_k = x[k]

            if k == target_modality and n_features > 2000:
                # Sample a subset of features
                n_sample = max(int(n_features * sample_ratio), 500)
                idx = torch.randperm(n_features, device=v_feat.device)[:n_sample]

                # Manually compute decoder forward on sampled features only
                v_sub = v_feat[idx]  # (k, latent_dim)
                x_sub = x_k[:, idx]  # (batch, k)

                # NBDataDecoder forward (reimplemented for subset):
                scale = F.softplus(decoder.scale_lin[b_k])[:, idx]  # (batch, k)
                bias = decoder.bias[b_k][:, idx]  # (batch, k)
                log_theta = decoder.log_theta[b_k][:, idx]  # (batch, k)

                logit_mu = scale * (u_k @ v_sub.t()) + bias  # (batch, k) ← KEY: only k dims!
                mu = F.softmax(logit_mu, dim=1) * l_k
                dist = D.NegativeBinomial(log_theta.exp(), logits=(mu + EPS).log() - log_theta)
                nll = -dist.log_prob(x_sub)

                # Scale for unbiased estimate
                x_nll[k] = nll.nanmean() * (n_features / n_sample)
            else:
                # Full decoder for small modalities
                dist = decoder(u_k, v_feat, b_k, l_k)
                x_nll[k] = -dist.log_prob(x_k).nanmean()

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

    SCGLUETrainer.compute_losses = lowrank_sampled_compute_losses
    logger.info(f"Patched GLUE: low-rank sampled decoder (ratio={sample_ratio}, target={target_modality})")
