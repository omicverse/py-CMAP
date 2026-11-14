"""
Spatial-aware GLUE: add spatial coherence loss during GLUE training.

Monkey-patches SCGLUETrainer.compute_losses to add:
  L_spatial = Σ_{(i,j) ∈ spatial_neighbors} ||z_i - z_j||² * w_ij

Usage:
    from pycmap.multi.spatial_glue import patch_glue_spatial
    patch_glue_spatial(adata_st, lam_spatial=0.1)
    glue = scglue.models.fit_SCGLUE(...)  # training now includes spatial loss
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)

_original_compute_losses = None
_spatial_state = {}  # Module-level state shared between patch and training


def patch_glue_spatial(adata_st, spatial_key='spatial', k=6,
                       lam_spatial=0.1, st_key='st',
                       lam_sample=None, sample_target='atac'):
    """
    Patch GLUE to add spatial coherence loss.

    Args:
        adata_st: AnnData with spatial coordinates
        spatial_key: obsm key for coordinates
        k: spatial neighbors
        lam_spatial: spatial loss weight
        st_key: modality key in GLUE for ST data
        lam_sample: if set, also add sampled reconstruction (e.g., 0.2)
        sample_target: which modality to sample ('atac' or all)
    """
    from scglue.models.scglue import SCGLUETrainer
    from scipy.sparse import issparse

    global _original_compute_losses, _spatial_state
    if _original_compute_losses is None:
        _original_compute_losses = SCGLUETrainer.compute_losses

    # Pre-compute spatial graph
    coords = adata_st.obsm[spatial_key].astype(float)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    dists, indices = nn.kneighbors(coords)
    median_dist = np.median(dists[:, 1])

    src, tgt, wts = [], [], []
    for i in range(len(coords)):
        for j in range(1, k + 1):
            src.append(i)
            tgt.append(indices[i, j])
            wts.append(np.exp(-dists[i, j]**2 / (2 * median_dist**2)))

    # Pre-compute full ST representation for spatial loss
    config = adata_st.uns.get('_scglue', {})
    use_rep = config.get('use_rep', None)
    use_layer = config.get('use_layer', None)
    features = config.get('features', None)

    if features is not None:
        mask = np.isin(adata_st.var_names, features)
    else:
        mask = np.ones(adata_st.n_vars, dtype=bool)

    if use_layer and use_layer in adata_st.layers:
        X_st = adata_st.layers[use_layer][:, mask]
    else:
        X_st = adata_st.X[:, mask]
    if issparse(X_st):
        X_st = X_st.toarray()

    xrep_st = adata_st.obsm[use_rep].astype(np.float32) if use_rep else np.zeros((adata_st.n_obs, 0), dtype=np.float32)

    _spatial_state = {
        'sp_src': np.array(src, dtype=np.int64),
        'sp_tgt': np.array(tgt, dtype=np.int64),
        'sp_wt': np.array(wts, dtype=np.float32),
        'X_st': X_st.astype(np.float32),
        'xrep_st': xrep_st,
        'st_key': st_key,
        'lam_spatial': lam_spatial,
        'lam_sample': lam_sample,
        'sample_target': sample_target,
        'device_tensors': None,  # Will be created on first use
    }

    def spatial_compute_losses(self, data, epoch, dsc_only=False):
        losses = _original_compute_losses(self, data, epoch, dsc_only)
        if dsc_only:
            return losses

        state = _spatial_state
        net = self.net
        device = net.device

        # Lazy-init device tensors
        if state['device_tensors'] is None:
            state['device_tensors'] = {
                'sp_src': torch.tensor(state['sp_src'], device=device),
                'sp_tgt': torch.tensor(state['sp_tgt'], device=device),
                'sp_wt': torch.tensor(state['sp_wt'], device=device),
                'X_st': torch.tensor(state['X_st'], device=device),
                'xrep_st': torch.tensor(state['xrep_st'], device=device),
            }

        dt = state['device_tensors']
        sk = state['st_key']

        if sk in net.keys:
            # Encode ALL ST spots (ST is small: 2-3K spots)
            u_full, _ = net.x2u[sk](dt['X_st'], dt['xrep_st'], lazy_normalizer=True)
            z_full = u_full.mean  # (n_spots, latent_dim)

            # Sample a subset of spatial edges for efficiency
            n_edges = dt['sp_src'].shape[0]
            n_sample = min(n_edges, 2000)
            edge_idx = torch.randperm(n_edges, device=device)[:n_sample]

            diff = z_full[dt['sp_src'][edge_idx]] - z_full[dt['sp_tgt'][edge_idx]]
            sp_loss = (dt['sp_wt'][edge_idx].unsqueeze(1) * diff.pow(2)).mean()

            losses['gen_loss'] = losses['gen_loss'] + state['lam_spatial'] * sp_loss
            losses['spatial'] = sp_loss

        # Optional sampled reconstruction
        if state['lam_sample'] is not None:
            # Already handled in sampled_glue.py if needed
            pass

        return losses

    SCGLUETrainer.compute_losses = spatial_compute_losses
    logger.info(f"Patched GLUE: spatial coherence (lam={lam_spatial}, k={k}, "
                f"{len(src)} edges)")


def unpatch_glue():
    """Restore original GLUE compute_losses."""
    global _original_compute_losses
    if _original_compute_losses is not None:
        from scglue.models.scglue import SCGLUETrainer
        SCGLUETrainer.compute_losses = _original_compute_losses
        _original_compute_losses = None
        logger.info("Restored original GLUE compute_losses")
