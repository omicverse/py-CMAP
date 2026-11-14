"""
Fused Gromov-Wasserstein (FGW) optimizer for cross-modal spatial mapping.

Replaces SSIM optimizer in CMAP Level 2.
Pure PyTorch implementation — fully GPU-accelerated.

FGW jointly optimizes:
  - Wasserstein term: feature similarity in embedding space
  - Gromov-Wasserstein term: structure preservation (cell-cell ↔ spot-spot distances)
  - Entropic regularization: uniform cell distribution (same as CMAP's entropy)

The transport plan T (cells × spots) is directly equivalent to CMAP's mapping matrix.
"""

import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


def fgw_mapping(cell_emb, spot_emb, spot_spatial=None,
                alpha=0.5, epsilon=0.05, n_iter_outer=50, n_iter_sinkhorn=20,
                device='cpu'):
    """
    Compute FGW transport plan mapping cells to spots.

    Args:
        cell_emb: (n_cells, d) cell embeddings (GLUE or Harmony)
        spot_emb: (n_spots, d) spot embeddings
        spot_spatial: (n_spots, 2) spatial coordinates. If provided, used as
                      additional structure constraint for spots.
        alpha: GW vs W weight. 0 = pure Wasserstein, 1 = pure GW.
               0.5 = balanced (recommended).
        epsilon: Entropic regularization strength.
                 Smaller = sharper mapping, larger = smoother.
        n_iter_outer: Number of FGW iterations (alternating T update + cost recompute).
        n_iter_sinkhorn: Number of Sinkhorn iterations per outer step.
        device: 'cpu' or 'cuda'.

    Returns:
        T: (n_cells, n_spots) transport plan (probability matrix, rows sum to 1/n).
    """
    cell_emb = torch.tensor(cell_emb, device=device, dtype=torch.float32)
    spot_emb = torch.tensor(spot_emb, device=device, dtype=torch.float32)

    n, m = cell_emb.shape[0], spot_emb.shape[0]

    # ===== Cost matrices =====
    # Wasserstein: pairwise distance in embedding space
    M = torch.cdist(cell_emb, spot_emb, p=2)  # (n, m)
    # Normalize
    M = M / (M.max() + 1e-8)

    # GW: structure matrices
    # Cell-cell distance in embedding space
    C1 = torch.cdist(cell_emb, cell_emb, p=2)  # (n, n)
    C1 = C1 / (C1.max() + 1e-8)

    # Spot-spot distance: combine embedding + spatial
    C2 = torch.cdist(spot_emb, spot_emb, p=2)  # (m, m)
    if spot_spatial is not None:
        spot_sp = torch.tensor(spot_spatial, device=device, dtype=torch.float32)
        C2_spatial = torch.cdist(spot_sp, spot_sp, p=2)
        C2_spatial = C2_spatial / (C2_spatial.max() + 1e-8)
        C2 = C2 / (C2.max() + 1e-8)
        # Combine: embedding distance + spatial distance
        C2 = 0.5 * C2 + 0.5 * C2_spatial
    else:
        C2 = C2 / (C2.max() + 1e-8)

    # Uniform marginals
    a = torch.ones(n, device=device) / n
    b = torch.ones(m, device=device) / m

    # Initialize transport plan (uniform)
    T = torch.outer(a, b)

    # Precompute constant terms for correct GW gradient
    # Full GW gradient: constC - 2*C1@T@C2
    # where constC = C1²@a ⊗ 1' + 1 ⊗ b'@C2² (constant w.r.t. T)
    C1_sq = C1 ** 2
    C2_sq = C2 ** 2
    f1 = C1_sq @ a  # (n,)
    f2 = C2_sq @ b  # (m,)
    constC = f1.unsqueeze(1) + f2.unsqueeze(0)  # (n, m)

    for outer in range(n_iter_outer):
        # ===== Correct FGW gradient =====
        # GW_cost[i,j] = sum_kl (C1[i,k] - C2[j,l])^2 * T[k,l]
        #              = constC[i,j] - 2 * (C1 @ T @ C2)[i,j]
        GW_cost = constC - 2.0 * C1 @ T @ C2  # (n, m)

        # ===== Combined cost =====
        cost = alpha * GW_cost + (1.0 - alpha) * M

        # ===== Sinkhorn iterations =====
        K = torch.exp(-cost / epsilon)
        K = K.clamp(min=1e-30)  # numerical stability

        u = torch.ones(n, device=device)
        v = torch.ones(m, device=device)

        for _ in range(n_iter_sinkhorn):
            u = a / (K @ v + 1e-30)
            v = b / (K.T @ u + 1e-30)

        T = torch.diag(u) @ K @ torch.diag(v)

        if outer % 10 == 0 and logger.isEnabledFor(logging.INFO):
            w_cost = (M * T).sum().item()
            gw_cost = (GW_cost * T).sum().item()
            logger.info(f"FGW iter {outer}: W={w_cost:.4f}, GW={gw_cost:.4f}")

    # Normalize rows to probabilities (each cell sums to 1/n, rescale)
    T_np = T.cpu().numpy()
    # Scale so each row sums to 1 (probability of cell i → spot j)
    row_sums = T_np.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    T_prob = T_np / row_sums

    return T_prob


def fgw_map_cell_to_spot(cell_emb, spot_emb, spot_spatial=None,
                         alpha=0.5, epsilon=0.05,
                         n_iter_outer=50, n_iter_sinkhorn=20,
                         device='cpu'):
    """
    High-level API matching CMAP's map_cell_to_spot_optimize interface.

    Returns:
        mapping: (n_cells, n_spots) probability matrix.
    """
    mapping = fgw_mapping(
        cell_emb, spot_emb, spot_spatial=spot_spatial,
        alpha=alpha, epsilon=epsilon,
        n_iter_outer=n_iter_outer, n_iter_sinkhorn=n_iter_sinkhorn,
        device=device
    )
    return mapping
