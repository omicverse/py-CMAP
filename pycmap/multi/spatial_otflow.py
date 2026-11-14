"""
OT-Flow: Optimal Transport + Flow Matching for spatial mapping.

Replaces CMAP's 3-level (SVM→SSIM→interpolation) pipeline with:
  Stage 1: Entropic OT  → soft coupling matrix π(cell, spot)
  Stage 2: Flow Matching → learn embedding→coords velocity field
  Stage 3: Spatial Smoothing → cell-type co-localization prior

No SVM, no domain partitioning, no threshold filtering → 100% mapping.

Usage:
    mapper = OTFlowMapper(z_dim=50, coord_dim=2)
    mapper.fit(z_st, coords_st, z_atac, cell_types=ct)
    coords_atac = mapper.transform()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


# ============================================================
# Stage 1: Entropic Optimal Transport
# ============================================================

class EntropicOT:
    """
    Sinkhorn-based entropic OT for cell→spot soft coupling.

    Solves: min_π <C, π> - ε·H(π)
            s.t. π·1 = a  (cell marginal)
                 π^T·1 = b (spot marginal)

    Cost: squared Euclidean distance in GLUE embedding space.
    """

    @staticmethod
    def solve(z_cells, z_spots, eps=0.05, max_iter=100,
              capacity_ratio=None, device='cuda'):
        """
        Args:
            z_cells: (n_cells, d) cell embeddings
            z_spots: (n_spots, d) spot embeddings
            eps: entropic regularization (smaller = sharper assignment)
            max_iter: Sinkhorn iterations
            capacity_ratio: max cells per spot / avg cells per spot.
                None = uniform marginal (balanced OT).

        Returns:
            pi: (n_cells, n_spots) coupling matrix, rows sum to 1
            cost_matrix: (n_cells, n_spots) pairwise distances
        """
        n_cells, n_spots = len(z_cells), len(z_spots)

        zc = torch.tensor(z_cells, dtype=torch.float32, device=device)
        zs = torch.tensor(z_spots, dtype=torch.float32, device=device)

        # Cost matrix: squared L2 normalized by dimension
        # Normalize embeddings for cosine-like distance
        zc_n = F.normalize(zc, dim=1)
        zs_n = F.normalize(zs, dim=1)
        C = 1.0 - zc_n @ zs_n.T  # cosine distance ∈ [0, 2]

        # Marginals
        a = torch.ones(n_cells, device=device) / n_cells
        if capacity_ratio is not None:
            # Allow spots to receive varying numbers of cells
            # Base capacity proportional to n_cells/n_spots
            b = torch.ones(n_spots, device=device) / n_spots
        else:
            b = torch.ones(n_spots, device=device) / n_spots

        # Log-domain Sinkhorn for numerical stability
        log_a = torch.log(a)
        log_b = torch.log(b)
        log_K = -C / eps  # log of Gibbs kernel

        u = torch.zeros(n_cells, device=device)  # dual variable (cells)
        v = torch.zeros(n_spots, device=device)  # dual variable (spots)

        for it in range(max_iter):
            # Update u: row normalization
            u = log_a - torch.logsumexp(log_K + v[None, :], dim=1)
            # Update v: column normalization
            v = log_b - torch.logsumexp(log_K + u[:, None], dim=0)

        # Coupling matrix
        log_pi = u[:, None] + log_K + v[None, :]
        pi = torch.exp(log_pi)

        # Normalize rows to sum to 1 (cell marginal)
        pi = pi / (pi.sum(dim=1, keepdim=True) + 1e-10)

        return pi.cpu().numpy(), C.cpu().numpy()


# ============================================================
# Stage 2: Conditional Flow Matching
# ============================================================

class VelocityNet(nn.Module):
    """
    MLP velocity field v_θ(x_t, z, t) for flow matching.

    Learns to map noise → spatial coordinates, conditioned on embedding.
    Input: x_t (coord_dim) + z (z_dim) + t_emb (t_dim)
    Output: velocity (coord_dim)
    """

    def __init__(self, z_dim, coord_dim, h_dim=256, h_depth=3, t_dim=32):
        super().__init__()
        self.coord_dim = coord_dim

        # Time embedding
        self.t_embed = nn.Sequential(
            nn.Linear(1, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )

        in_dim = coord_dim + z_dim + t_dim
        layers = [nn.Linear(in_dim, h_dim), nn.SiLU()]
        for _ in range(h_depth - 1):
            layers += [nn.Linear(h_dim, h_dim), nn.SiLU()]
        layers.append(nn.Linear(h_dim, coord_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t, z, t):
        """
        Args:
            x_t: (B, coord_dim) current position
            z:   (B, z_dim) embedding condition
            t:   (B, 1) time ∈ [0, 1]
        """
        t_emb = self.t_embed(t)
        h = torch.cat([x_t, z, t_emb], dim=-1)
        return self.net(h)


class ConditionalFlowMatcher:
    """
    Conditional Flow Matching for spatial coordinate refinement.

    Training: learns velocity field from OT-paired (noise, target_coords).
    Inference: integrate velocity to generate coordinates from noise.

    The key innovation: training pairs come from OT coupling, not strict
    pairing. Each cell is paired with its OT-weighted centroid ± spatial jitter.
    """

    def __init__(self, z_dim, coord_dim, h_dim=256, h_depth=3, device='cuda'):
        self.device = device
        self.coord_dim = coord_dim
        self.model = VelocityNet(z_dim, coord_dim, h_dim, h_depth).to(device)
        self.coord_mean = None
        self.coord_std = None

    def _normalize(self, coords):
        return (coords - self.coord_mean) / self.coord_std

    def _denormalize(self, coords):
        return coords * self.coord_std + self.coord_mean

    def train(self, z_cells, target_coords, sigma_noise=0.1,
              n_epochs=2000, batch_size=512, lr=1e-3):
        """
        Train flow matching model.

        Args:
            z_cells: (N, z_dim) cell embeddings
            target_coords: (N, coord_dim) OT-weighted target coordinates
            sigma_noise: noise scale for conditional OT path (σ_min)
            n_epochs: training epochs
            batch_size: batch size
            lr: learning rate
        """
        z_t = torch.tensor(z_cells, dtype=torch.float32, device=self.device)
        x1_all = torch.tensor(target_coords, dtype=torch.float32, device=self.device)

        # Normalize coordinates
        self.coord_mean = x1_all.mean(dim=0, keepdim=True)
        self.coord_std = x1_all.std(dim=0, keepdim=True) + 1e-6
        x1_all = self._normalize(x1_all)

        N = len(z_t)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, eta_min=1e-5)

        self.model.train()
        losses = []

        for epoch in range(n_epochs):
            idx = torch.randperm(N, device=self.device)[:batch_size]
            x1 = x1_all[idx]
            z = z_t[idx]

            # Sample time t ~ U(0,1)
            t = torch.rand(len(idx), 1, device=self.device)

            # Sample x0 ~ N(0, 1)  (noise)
            x0 = torch.randn_like(x1)

            # Conditional OT path: x_t = (1-t)*x0 + t*x1
            # (Lipman et al., 2023: linear interpolation path)
            x_t = (1 - t) * x0 + t * x1

            # Target velocity: dx/dt = x1 - x0
            u_t = x1 - x0

            # Predict velocity
            v_t = self.model(x_t, z, t)

            # Flow matching loss
            loss = F.mse_loss(v_t, u_t)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())

            if epoch % 500 == 0 or epoch == n_epochs - 1:
                avg = np.mean(losses[-500:])
                logger.info(f"  FlowMatch epoch {epoch:4d}: loss={avg:.6f}")

        return losses

    @torch.no_grad()
    def generate(self, z_cells, n_steps=50, n_samples=1):
        """
        Generate coordinates by integrating the learned velocity field.

        Args:
            z_cells: (N, z_dim) embeddings
            n_steps: ODE integration steps (Euler)
            n_samples: samples per cell for uncertainty

        Returns:
            coords: (N, n_samples, coord_dim)
        """
        self.model.eval()
        N = len(z_cells)
        z = torch.tensor(z_cells, dtype=torch.float32, device=self.device)
        dt = 1.0 / n_steps

        all_samples = []
        for _ in range(n_samples):
            x = torch.randn(N, self.coord_dim, device=self.device)

            for step in range(n_steps):
                t = torch.full((N, 1), step * dt, device=self.device)
                v = self.model(x, z, t)
                x = x + v * dt

            coords = self._denormalize(x).cpu().numpy()
            all_samples.append(coords)

        return np.stack(all_samples, axis=1)


# ============================================================
# Stage 3: Spatial Smoothing
# ============================================================

def spatial_smooth(coords, z_cells, cell_types=None,
                   k_neighbors=15, alpha=0.3, n_iter=3):
    """
    Smooth mapped coordinates using embedding-space neighbors.

    For each cell, its position is adjusted toward the centroid of
    its embedding-space neighbors' positions. This enforces:
    - Cells with similar chromatin → nearby spatial positions
    - Same cell type cells cluster spatially

    Args:
        coords: (N, D) mapped coordinates
        z_cells: (N, z_dim) embeddings
        cell_types: (N,) optional cell type labels for type-aware smoothing
        k_neighbors: number of neighbors
        alpha: smoothing strength (0=no change, 1=full average)
        n_iter: number of smoothing iterations

    Returns:
        smoothed_coords: (N, D)
    """
    coords = coords.copy()
    N = len(coords)

    for it in range(n_iter):
        if cell_types is not None:
            # Type-aware smoothing: only average within same cell type
            smoothed = coords.copy()
            for ct in np.unique(cell_types):
                mask = cell_types == ct
                if mask.sum() < k_neighbors + 1:
                    continue
                z_ct = z_cells[mask]
                c_ct = coords[mask]
                k = min(k_neighbors, len(z_ct) - 1)
                nn = NearestNeighbors(n_neighbors=k + 1, metric='cosine').fit(z_ct)
                _, idx = nn.kneighbors(z_ct)
                # Weighted average: closer in embedding → higher weight
                for i in range(len(z_ct)):
                    neigh_coords = c_ct[idx[i, 1:]]
                    smoothed[np.where(mask)[0][i]] = (
                        (1 - alpha) * c_ct[i] + alpha * neigh_coords.mean(axis=0)
                    )
            coords = smoothed
        else:
            nn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric='cosine').fit(z_cells)
            _, idx = nn.kneighbors(z_cells)
            neigh_mean = np.array([coords[idx[i, 1:]].mean(axis=0) for i in range(N)])
            coords = (1 - alpha) * coords + alpha * neigh_mean

    return coords


# ============================================================
# Full Pipeline
# ============================================================

class OTFlowMapper:
    """
    OT-Flow spatial mapper: maps cells to spatial coordinates using
    optimal transport + flow matching + spatial smoothing.

    No SVM, no domain partitioning, 100% mapping rate.
    """

    def __init__(self, z_dim=50, coord_dim=2, device='cuda'):
        self.z_dim = z_dim
        self.coord_dim = coord_dim
        self.device = device
        self.flow_matcher = None
        self.pi = None  # OT coupling
        self.ot_coords = None  # OT centroid coordinates

    def fit(self, z_st, coords_st, z_atac,
            cell_types=None,
            # Stage 1 params
            ot_eps=0.05, ot_max_iter=100,
            # Stage 2 params
            flow_epochs=2000, flow_batch=512, flow_lr=1e-3,
            flow_h_dim=256, flow_h_depth=3,
            # Stage 3 params
            smooth_k=15, smooth_alpha=0.3, smooth_iter=3,
            # Field params (optional Stage 2b)
            use_field=False, field_bandwidth=None,
            field_lam_confidence=0.5, field_n_refine=2,
            field_refine_weight=0.3,
            verbose=True):
        """
        Fit the OT-Flow mapper.

        Args:
            z_st: (n_spots, z_dim) ST embeddings
            coords_st: (n_spots, coord_dim) spatial coordinates
            z_atac: (n_cells, z_dim) ATAC embeddings
            cell_types: (n_cells,) optional cell type labels
            use_field: if True, apply kernel field refinement after OT dispersion
            field_bandwidth: kernel bandwidth (None=auto)
            field_lam_confidence: confidence weight in field cost
            field_n_refine: iterative refinement rounds for field
            field_refine_weight: weight of mapped cells as field anchors
        """
        import time
        t_start = time.time()

        n_cells, n_spots = len(z_atac), len(z_st)
        self.coords_st = coords_st
        self.z_st = z_st
        self.z_atac = z_atac
        if verbose:
            print(f"[OT-Flow] {n_cells} cells → {n_spots} spots, "
                  f"{self.coord_dim}D, eps={ot_eps}")

        # ── Stage 1: Entropic OT → hard assignment ──
        t = time.time()
        self.pi, cost = EntropicOT.solve(
            z_atac, z_st, eps=ot_eps, max_iter=ot_max_iter, device=self.device)

        # Hard assignment: each cell → its best spot
        self.assigned_spot = np.argmax(self.pi, axis=1)
        self.ot_coords = coords_st[self.assigned_spot].copy()  # (n_cells, coord_dim)

        # Also store soft centroid for comparison
        self.ot_soft_coords = self.pi @ coords_st

        # Coupling statistics
        entropy = -np.sum(self.pi * np.log(self.pi + 1e-10), axis=1)
        max_weight = self.pi.max(axis=1)

        if verbose:
            print(f"[Stage 1] OT coupling: {time.time()-t:.1f}s")
            print(f"  Coupling entropy: {entropy.mean():.2f} ± {entropy.std():.2f}")
            print(f"  Max weight per cell: {max_weight.mean():.4f} ± {max_weight.std():.4f}")

        # ── Stage 2: Sub-spot dispersion using coupling ──
        t = time.time()
        self.subspot_disperse(k_neighbors=6)

        if verbose:
            # Count cells per spot
            from collections import Counter
            spot_counts = Counter(self.assigned_spot)
            multi = sum(1 for v in spot_counts.values() if v > 1)
            max_per = max(spot_counts.values())
            print(f"[Stage 2] Disperse: {time.time()-t:.1f}s | "
                  f"{multi}/{len(spot_counts)} spots have >1 cell, max={max_per}")

        # ── Stage 2b (optional): Kernel field refinement ──
        self.field = None
        self.field_coords = None
        if use_field:
            t = time.time()
            from .spatial_field import KernelSpatialField
            self.field = KernelSpatialField(
                bandwidth=field_bandwidth,
                coord_dim=self.coord_dim,
                z_dim=self.z_dim)
            self.field_coords = self.field.iterative_map(
                z_st, coords_st, z_atac,
                n_refine=field_n_refine,
                refine_weight=field_refine_weight,
                lam_confidence=field_lam_confidence,
                device=self.device,
                verbose=verbose)
            if verbose:
                print(f"[Stage 2b] Field: {time.time()-t:.1f}s")

        # ── Stage 3: Spatial Smoothing ──
        t = time.time()
        base_coords = self.field_coords if use_field else self.dispersed_coords
        self.smoothed_coords = spatial_smooth(
            base_coords, z_atac, cell_types=cell_types,
            k_neighbors=smooth_k, alpha=smooth_alpha, n_iter=smooth_iter)

        # Flow matching is optional — store None
        self.flow_matcher = None
        self.mapped_coords = base_coords.copy()

        if verbose:
            print(f"[Stage 3] Smoothing: {time.time()-t:.1f}s")
            print(f"[Total] {time.time()-t_start:.1f}s, "
                  f"{n_cells} cells mapped")

        return self

    def get_coords(self, stage='dispersed'):
        """
        Get mapped coordinates.

        Args:
            stage: 'ot' (hard argmax), 'ot_soft' (weighted centroid),
                   'dispersed' (sub-spot dispersed), 'field' (kernel field refined),
                   'flow' (flow matching), 'smoothed' (spatial smoothed)
        """
        if stage == 'ot':
            return self.ot_coords
        elif stage == 'ot_soft':
            return self.ot_soft_coords
        elif stage == 'dispersed':
            return self.dispersed_coords
        elif stage == 'field':
            return self.field_coords if self.field_coords is not None else self.dispersed_coords
        elif stage == 'flow':
            return self.mapped_coords
        elif stage == 'smoothed':
            return self.smoothed_coords
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def get_coupling(self):
        """Get OT coupling matrix π (n_cells, n_spots)."""
        return self.pi

    def get_confidence(self):
        """Per-cell mapping confidence (max coupling weight)."""
        return self.pi.max(axis=1)

    def get_uncertainty(self):
        """Per-cell mapping uncertainty (normalized coupling entropy)."""
        entropy = -np.sum(self.pi * np.log(self.pi + 1e-10), axis=1)
        max_ent = np.log(self.pi.shape[1])
        return entropy / max_ent  # 0=certain, 1=uniform

    def filter_by_confidence(self, coords, min_confidence=None, max_uncertainty=None):
        """
        Set unmapped cells to NaN based on coupling quality.

        Cells with weak coupling (no good spot match) are likely
        not present in the spatial tissue and should not be mapped.

        Args:
            coords: (n_cells, D) coordinates to filter
            min_confidence: minimum max coupling weight (e.g. 0.05)
            max_uncertainty: maximum normalized entropy (e.g. 0.55)

        Returns:
            filtered_coords: copy with NaN for low-confidence cells
        """
        coords = coords.copy()
        if min_confidence is not None:
            low_conf = self.get_confidence() < min_confidence
            coords[low_conf] = np.nan
        if max_uncertainty is not None:
            high_unc = self.get_uncertainty() > max_uncertainty
            coords[high_unc] = np.nan
        n_valid = np.isfinite(coords[:, 0]).sum()
        logger.info(f"Filtered: {n_valid}/{len(coords)} cells kept")
        return coords

    def subspot_disperse(self, k_neighbors=6):
        """
        Disperse cells within their assigned spot using OT coupling.

        Cells assigned to the same spot get different positions based on
        their secondary affinities to neighboring spots. No interpolation,
        no generation — just using information already in the coupling matrix.

        For cell i assigned to spot j:
          offset_i = Σ_{k ∈ neighbors(j)} π[i,k] * (coords[k] - coords[j])
          position_i = coords[j] + offset_i / Σ_k π[i,k]

        Same spot, different cells → different π → different offsets → dispersed.
        """
        coords_st = self.coords_st
        n_cells = len(self.assigned_spot)
        n_spots = len(coords_st)

        # Build spot neighbor graph
        k = min(k_neighbors, n_spots - 1)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(coords_st)
        _, spot_neighbors = nn.kneighbors(coords_st)  # (n_spots, k+1)
        # spot_neighbors[:, 0] is self, 1: are neighbors

        dispersed = np.zeros((n_cells, self.coord_dim))

        for i in range(n_cells):
            j = self.assigned_spot[i]
            center = coords_st[j]
            neigh_idx = spot_neighbors[j, 1:]  # k neighbors of spot j

            # Cell i's coupling weights to these neighbors
            w = self.pi[i, neigh_idx]
            w_center = self.pi[i, j]

            # Offset: weighted direction toward neighbors
            offsets = coords_st[neigh_idx] - center  # (k, D)
            w_total = w.sum() + w_center + 1e-10

            # Position = center + coupling-weighted offset
            dispersed[i] = center + (w @ offsets) / w_total

        self.dispersed_coords = dispersed
        return dispersed

    def generate_samples(self, z_cells, n_samples=10, n_steps=50):
        """Generate multiple coordinate samples per cell for uncertainty."""
        return self.flow_matcher.generate(z_cells, n_steps=n_steps, n_samples=n_samples)
