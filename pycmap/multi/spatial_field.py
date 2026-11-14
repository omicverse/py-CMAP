"""
Spatial Fields: continuous position→embedding for super-resolution mapping.

Two approaches:
  1. NeuralSpatialField: MLP with Fourier positional encoding (NeRF-style).
     Good for smooth fields, but extrapolates outside tissue boundary.
  2. KernelSpatialField: Gaussian kernel regression (non-parametric).
     Naturally decays outside tissue, preserves tissue shape.

Pipeline:
  1. Train field F on ST: (coords_spot → z_spot)
  2. Initialize ATAC positions from OT hard assignment
  3. Optimize: (x*,y*) = argmin ||F(x,y) - z_atac||² + λ·(1-conf(x,y))
  4. Refine field with ATAC anchors → repeat 3-4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


class SpatialFieldNet(nn.Module):
    """
    MLP that maps spatial coordinates → embedding.

    Uses positional encoding (Fourier features) for high-frequency detail,
    similar to NeRF but for tissue embedding fields.
    """

    def __init__(self, coord_dim=2, z_dim=50, h_dim=256, h_depth=4,
                 n_freq=10):
        super().__init__()
        self.n_freq = n_freq
        # Positional encoding: [x, sin(2^k x), cos(2^k x) for k=0..n_freq-1]
        pe_dim = coord_dim * (1 + 2 * n_freq)

        layers = [nn.Linear(pe_dim, h_dim), nn.SiLU()]
        for _ in range(h_depth - 1):
            layers += [nn.Linear(h_dim, h_dim), nn.SiLU()]
        layers.append(nn.Linear(h_dim, z_dim))
        self.net = nn.Sequential(*layers)

        # Frequency bands
        self.register_buffer(
            'freq_bands',
            2.0 ** torch.linspace(0, n_freq - 1, n_freq)
        )

    def positional_encoding(self, coords):
        """Fourier feature encoding for continuous positions."""
        # coords: (B, D), normalize to [-1, 1] first (done externally)
        pe = [coords]
        for freq in self.freq_bands:
            pe.append(torch.sin(freq * np.pi * coords))
            pe.append(torch.cos(freq * np.pi * coords))
        return torch.cat(pe, dim=-1)

    def forward(self, coords):
        """
        Args:
            coords: (B, coord_dim) normalized spatial coordinates
        Returns:
            z_pred: (B, z_dim) predicted embedding
        """
        pe = self.positional_encoding(coords)
        return self.net(pe)


class NeuralSpatialField:
    """
    Full pipeline: field training + gradient-based cell positioning + iterative refinement.
    """

    def __init__(self, coord_dim=2, z_dim=50, h_dim=256, h_depth=4,
                 n_freq=10, device='cuda'):
        self.coord_dim = coord_dim
        self.z_dim = z_dim
        self.device = device

        self.field = SpatialFieldNet(
            coord_dim, z_dim, h_dim, h_depth, n_freq
        ).to(device)

        # Normalization params (fit to ST coords)
        self.coord_center = None
        self.coord_scale = None

    def _normalize_coords(self, coords):
        """Normalize to [-1, 1] range."""
        return (coords - self.coord_center) / self.coord_scale

    def _denormalize_coords(self, coords):
        return coords * self.coord_scale + self.coord_center

    def train_field(self, coords, z_targets, n_epochs=2000, lr=1e-3,
                    batch_size=512, weights=None):
        """
        Train the spatial field on anchor points (ST spots or ATAC cells).

        Args:
            coords: (N, D) spatial coordinates
            z_targets: (N, z_dim) target embeddings
            n_epochs: training epochs
            lr: learning rate
            weights: (N,) optional per-sample weights (for iterative refinement)
        """
        # Fit normalization on first call
        coords_t = torch.tensor(coords, dtype=torch.float32, device=self.device)
        if self.coord_center is None:
            self.coord_center = coords_t.mean(dim=0, keepdim=True)
            self.coord_scale = (coords_t.max(dim=0).values -
                                coords_t.min(dim=0).values).unsqueeze(0) / 2 + 1e-6

        coords_n = self._normalize_coords(coords_t)
        z_t = torch.tensor(z_targets, dtype=torch.float32, device=self.device)

        if weights is not None:
            w_t = torch.tensor(weights, dtype=torch.float32, device=self.device)
        else:
            w_t = torch.ones(len(coords), device=self.device)

        N = len(coords)
        optimizer = torch.optim.AdamW(self.field.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, eta_min=1e-5)

        self.field.train()
        for epoch in range(n_epochs):
            idx = torch.randperm(N, device=self.device)[:batch_size]
            z_pred = self.field(coords_n[idx])
            # Weighted MSE
            loss = (w_t[idx].unsqueeze(1) * (z_pred - z_t[idx]).pow(2)).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.field.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if epoch % 500 == 0 or epoch == n_epochs - 1:
                with torch.no_grad():
                    z_all = self.field(coords_n)
                    full_loss = ((z_all - z_t).pow(2)).mean().item()
                logger.info(f"  Field epoch {epoch:4d}: loss={full_loss:.6f}")

    @torch.no_grad()
    def query(self, coords):
        """Query the field at arbitrary continuous positions."""
        self.field.eval()
        coords_t = torch.tensor(coords, dtype=torch.float32, device=self.device)
        coords_n = self._normalize_coords(coords_t)
        return self.field(coords_n).cpu().numpy()

    def optimize_positions(self, z_targets, init_coords, n_steps=200, lr=0.01,
                           max_displacement=None):
        """
        Local refinement: optimize positions WITHIN a neighborhood of init_coords.

        NOT global optimization — cells stay near their OT-assigned spot.
        The field gradient determines WHERE within the local neighborhood
        the cell best matches.

        Args:
            z_targets: (N, z_dim) target embeddings (ATAC cells)
            init_coords: (N, D) initial positions (from OT)
            n_steps: gradient descent steps
            lr: position learning rate
            max_displacement: maximum distance from init (in original coord space).
                If None, auto-set to median spot spacing.

        Returns:
            optimized_coords: (N, D) locally refined positions
        """
        self.field.eval()
        z_t = torch.tensor(z_targets, dtype=torch.float32, device=self.device)
        init_t = torch.tensor(init_coords, dtype=torch.float32, device=self.device)
        init_n = self._normalize_coords(init_t)

        # Auto-detect max displacement from spot spacing
        # Use UNIQUE positions to avoid zero distances from overlapping cells
        if max_displacement is None:
            from sklearn.neighbors import NearestNeighbors
            unique_coords = np.unique(np.round(init_coords, 2), axis=0)
            nn = NearestNeighbors(n_neighbors=2).fit(unique_coords)
            d, _ = nn.kneighbors(unique_coords)
            max_displacement = np.median(d[:, 1]) * 0.5  # half spot spacing
            logger.info(f"  Max displacement: {max_displacement:.1f} "
                        f"(spot spacing ~{np.median(d[:,1]):.1f})")

        # Convert max_displacement to normalized space
        max_disp_n = max_displacement / self.coord_scale.cpu().numpy().mean()

        pos = init_n.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([pos], lr=lr)

        for step in range(n_steps):
            z_pred = self.field(pos)
            loss = F.mse_loss(z_pred, z_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Leash: clamp displacement from init
            with torch.no_grad():
                disp = pos - init_n
                disp_norm = disp.norm(dim=1, keepdim=True)
                exceed = disp_norm > max_disp_n
                if exceed.any():
                    disp = torch.where(exceed, disp * max_disp_n / (disp_norm + 1e-8), disp)
                    pos.copy_(init_n + disp)

        optimized = self._denormalize_coords(pos.detach()).cpu().numpy()
        return optimized

    def fit(self, z_st, coords_st, z_atac, init_coords_atac,
            field_epochs=2000, field_lr=1e-3,
            opt_steps=200, opt_lr=0.01,
            n_refine=2, refine_weight=0.3,
            verbose=True):
        """
        Full pipeline: train field → optimize → refine → optimize.

        Args:
            z_st: (n_spots, z_dim) ST embeddings
            coords_st: (n_spots, D) ST coordinates
            z_atac: (n_cells, z_dim) ATAC embeddings
            init_coords_atac: (n_cells, D) initial positions (from OT)
            field_epochs: epochs for field training
            field_lr: field learning rate
            opt_steps: gradient descent steps per cell
            opt_lr: position optimization learning rate
            n_refine: number of iterative refinement rounds
            refine_weight: weight of ATAC anchors relative to ST in refinement

        Returns:
            final_coords: (n_cells, D) super-resolution positions
        """
        import time
        t_start = time.time()
        n_cells, n_spots = len(z_atac), len(z_st)

        if verbose:
            print(f"[NeuralField] {n_cells} cells, {n_spots} spots, {self.coord_dim}D")

        # ── Round 0: Train field on ST spots ──
        t = time.time()
        if verbose:
            print(f"[Round 0] Training field on {n_spots} ST spots...")
        self.train_field(coords_st, z_st, n_epochs=field_epochs, lr=field_lr)
        if verbose:
            print(f"  Field trained: {time.time()-t:.1f}s")

        # ── Round 0: Optimize ATAC positions ──
        t = time.time()
        current_coords = self.optimize_positions(
            z_atac, init_coords_atac, n_steps=opt_steps, lr=opt_lr)
        if verbose:
            displacement = np.sqrt(((current_coords - init_coords_atac)**2).sum(axis=1))
            print(f"  Optimized: {time.time()-t:.1f}s, "
                  f"displacement mean={displacement.mean():.1f}, "
                  f"median={np.median(displacement):.1f}")

        # ── Iterative refinement ──
        for r in range(n_refine):
            t = time.time()
            if verbose:
                print(f"[Round {r+1}] Refining field with {n_cells} ATAC anchors "
                      f"(weight={refine_weight})...")

            # Combine ST spots (weight=1) and ATAC cells (weight=refine_weight)
            all_coords = np.vstack([coords_st, current_coords])
            all_z = np.vstack([z_st, z_atac])
            all_w = np.concatenate([
                np.ones(n_spots),
                np.full(n_cells, refine_weight)
            ])

            # Retrain field on combined data
            self.train_field(all_coords, all_z,
                             n_epochs=max(field_epochs // 2, 500),
                             lr=field_lr * 0.5, weights=all_w)

            # Re-optimize positions
            current_coords = self.optimize_positions(
                z_atac, current_coords, n_steps=opt_steps, lr=opt_lr * 0.5)

            if verbose:
                displacement = np.sqrt(((current_coords - init_coords_atac)**2).sum(axis=1))
                print(f"  Refined: {time.time()-t:.1f}s, "
                      f"displacement mean={displacement.mean():.1f}")

        self.final_coords = current_coords
        if verbose:
            print(f"[Total] {time.time()-t_start:.1f}s")

        return current_coords


# ============================================================
# Kernel-based Spatial Field (non-parametric)
# ============================================================

class KernelSpatialField:
    """
    Gaussian kernel regression field: F(x) = Σ K(x, x_j) z_j / Σ K(x, x_j).

    Advantages over MLP field:
      - No training needed (closed-form)
      - Naturally decays outside tissue boundary (no extrapolation artifacts)
      - Confidence = Σ K(x, x_j) indicates how well-covered a position is
      - Fast update: just add new anchor points

    For mapping, uses confidence-weighted OT cost:
      C(cell, pos) = cosine_dist(z_cell, F(pos)) + λ · (1 - confidence(pos))
    This prevents cells from being assigned to poorly-covered regions.
    """

    def __init__(self, bandwidth=None, coord_dim=2, z_dim=50):
        """
        Args:
            bandwidth: Gaussian kernel bandwidth (σ). If None, auto-set from
                       median nearest-neighbor distance of anchor points.
            coord_dim: spatial dimensions (2 or 3)
            z_dim: embedding dimension
        """
        self.bandwidth = bandwidth
        self.coord_dim = coord_dim
        self.z_dim = z_dim

        # Anchor points
        self.anchor_coords = None  # (M, D)
        self.anchor_z = None       # (M, z_dim)
        self.anchor_weights = None # (M,) per-anchor confidence weights
        self._nn_index = None      # cached NN index for fast queries

    def fit(self, coords, z, weights=None):
        """
        Set anchor points for the field.

        Args:
            coords: (M, D) spatial coordinates of anchors (ST spots or mapped cells)
            z: (M, z_dim) embeddings at anchor positions
            weights: (M,) optional per-anchor weights (e.g., 1.0 for ST, 0.3 for ATAC)
        """
        self.anchor_coords = np.asarray(coords, dtype=np.float64)
        self.anchor_z = np.asarray(z, dtype=np.float32)

        if weights is not None:
            self.anchor_weights = np.asarray(weights, dtype=np.float64)
        else:
            self.anchor_weights = np.ones(len(coords), dtype=np.float64)

        # Auto bandwidth from median NN distance
        self._nn_index = None  # reset cached index

        if self.bandwidth is None:
            nn = NearestNeighbors(n_neighbors=2).fit(self.anchor_coords)
            d, _ = nn.kneighbors(self.anchor_coords)
            self.bandwidth = np.median(d[:, 1])
            logger.info(f"  Kernel bandwidth (auto): {self.bandwidth:.2f}")

        return self

    def query(self, coords, return_confidence=False, k_approx=None):
        """
        Query the field at arbitrary positions.

        Args:
            coords: (N, D) query positions
            return_confidence: if True, also return per-position confidence
            k_approx: if set, only use k nearest anchors per query (fast approx).
                      None = use all anchors (exact but slow for large M).

        Returns:
            z_pred: (N, z_dim) predicted embeddings
            confidence: (N,) if return_confidence=True, sum of kernel weights
        """
        coords = np.asarray(coords, dtype=np.float64)
        N = len(coords)
        M = len(self.anchor_coords)
        sigma = self.bandwidth

        # Auto k_approx for large anchor sets
        if k_approx is None and M > 5000:
            k_approx = min(100, M)

        if k_approx is not None and k_approx < M:
            # Fast approximate: only use k nearest anchors
            if not hasattr(self, '_nn_index') or self._nn_index is None:
                self._nn_index = NearestNeighbors(
                    n_neighbors=k_approx, algorithm='ball_tree'
                ).fit(self.anchor_coords)

            z_pred = np.zeros((N, self.z_dim), dtype=np.float32)
            conf = np.zeros(N, dtype=np.float64)

            chunk_size = 5000
            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                c = coords[start:end]
                dists, idx = self._nn_index.kneighbors(c)

                # Gaussian kernel on k nearest
                K = self.anchor_weights[idx] * np.exp(-dists**2 / (2 * sigma**2))
                K_sum = K.sum(axis=1, keepdims=True) + 1e-10

                # Weighted embedding: gather anchor embeddings for each query
                z_neigh = self.anchor_z[idx]  # (chunk, k, z_dim)
                z_pred[start:end] = (K[:, :, None] * z_neigh).sum(axis=1) / K_sum
                conf[start:end] = K_sum.squeeze()
        else:
            # Exact: use all anchors
            chunk_size = 2000
            z_pred = np.zeros((N, self.z_dim), dtype=np.float32)
            conf = np.zeros(N, dtype=np.float64)

            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                c = coords[start:end]
                dist2 = ((c[:, None, :] - self.anchor_coords[None, :, :]) ** 2).sum(axis=2)
                K = self.anchor_weights[None, :] * np.exp(-dist2 / (2 * sigma ** 2))
                K_sum = K.sum(axis=1, keepdims=True) + 1e-10
                z_pred[start:end] = (K @ self.anchor_z) / K_sum
                conf[start:end] = K_sum.squeeze()

        if return_confidence:
            conf_norm = conf / (conf.max() + 1e-10)
            return z_pred, conf_norm
        return z_pred

    def confidence_map(self, grid_resolution=100):
        """
        Generate a 2D confidence map over the bounding box of anchors.

        Returns:
            xx, yy: meshgrid arrays
            conf: (grid_resolution, grid_resolution) confidence values
        """
        assert self.coord_dim == 2, "confidence_map only for 2D"
        margin = self.bandwidth * 2
        x_min, y_min = self.anchor_coords.min(axis=0) - margin
        x_max, y_max = self.anchor_coords.max(axis=0) + margin

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, grid_resolution),
            np.linspace(y_min, y_max, grid_resolution)
        )
        grid = np.column_stack([xx.ravel(), yy.ravel()])
        _, conf = self.query(grid, return_confidence=True)
        return xx, yy, conf.reshape(grid_resolution, grid_resolution)

    def map_cells(self, z_cells, init_coords=None,
                  lam_confidence=0.5, n_sample=80000, max_candidates=25000,
                  boundary_factor=1.2, ot_eps=0.01, ot_iter=200,
                  device='cuda', verbose=True):
        """
        Map cells to continuous positions via Sinkhorn OT on dense candidates.

        1. Generate dense random candidate positions within tissue boundary
        2. Query field at candidates → embeddings + confidence
        3. Build cost: C = cosine_dist(cells, field_at_candidates) + λ*(1-conf)
        4. Sinkhorn OT between cells and candidates
        5. Each cell → argmax candidate position

        This produces truly continuous coordinates (not grid-snapped).

        Args:
            z_cells: (N, z_dim) cell embeddings
            init_coords: not used (kept for API compat), positions come from OT on field
            lam_confidence: weight of confidence penalty in cost
            n_sample: number of random candidate points to generate
            max_candidates: max candidates after tissue boundary filtering
            boundary_factor: keep candidates within this × spot_spacing of anchors
            ot_eps: Sinkhorn regularization
            ot_iter: Sinkhorn iterations
            device: 'cuda' or 'cpu'

        Returns:
            mapped_coords: (N, D) continuous positions
        """
        import torch
        import time
        t0 = time.time()

        N = len(z_cells)
        z_cells = np.asarray(z_cells, dtype=np.float32)

        # ── 1. Generate dense candidate positions within tissue ──
        coords_min = self.anchor_coords.min(axis=0) - self.bandwidth * 2
        coords_max = self.anchor_coords.max(axis=0) + self.bandwidth * 2

        rng = np.random.RandomState(42)
        sample_pts = np.column_stack([
            rng.uniform(coords_min[d], coords_max[d], n_sample)
            for d in range(self.coord_dim)
        ])

        # Filter: keep only points near tissue (within boundary_factor × spot_spacing)
        nn_anchor = NearestNeighbors(n_neighbors=1).fit(self.anchor_coords)
        dists, _ = nn_anchor.kneighbors(sample_pts)
        spot_spacing = self.bandwidth  # bandwidth ≈ median NN distance
        in_tissue = dists[:, 0] < spot_spacing * boundary_factor
        sample_pts = sample_pts[in_tissue]

        if len(sample_pts) > max_candidates:
            idx = rng.permutation(len(sample_pts))[:max_candidates]
            sample_pts = sample_pts[idx]

        n_pts = len(sample_pts)
        if verbose:
            logger.info(f"  Candidates: {n_pts} (from {n_sample} samples, "
                        f"boundary={boundary_factor}×)")

        # ── 2. Query field at candidate positions ──
        z_field, conf = self.query(sample_pts, return_confidence=True)

        # ── 3. Build cost matrix on GPU ──
        zc = torch.tensor(z_cells, dtype=torch.float32, device=device)
        zf = torch.tensor(z_field, dtype=torch.float32, device=device)
        conf_t = torch.tensor(conf, dtype=torch.float32, device=device)

        # Normalize for cosine
        zc = zc / (zc.norm(dim=1, keepdim=True) + 1e-8)
        zf = zf / (zf.norm(dim=1, keepdim=True) + 1e-8)

        C_cos = 1.0 - zc @ zf.T  # (N, n_pts)
        conf_norm = conf_t / (conf_t.max() + 1e-10)
        C = C_cos + lam_confidence * (1.0 - conf_norm).unsqueeze(0)

        # ── 4. Sinkhorn OT ──
        n_c, n_f = C.shape
        log_K = -C / ot_eps
        log_a = torch.log(torch.ones(n_c, device=device) / n_c)
        log_b = torch.log(torch.ones(n_f, device=device) / n_f)
        u = torch.zeros(n_c, device=device)
        v = torch.zeros(n_f, device=device)

        for _ in range(ot_iter):
            u = log_a - torch.logsumexp(log_K + v[None, :], dim=1)
            v = log_b - torch.logsumexp(log_K + u[:, None], dim=0)

        log_pi = u[:, None] + log_K + v[None, :]
        pi = torch.exp(log_pi)
        pi = (pi / (pi.sum(1, keepdim=True) + 1e-10)).cpu().numpy()

        # ── 5. Argmax → continuous position ──
        mapped_coords = sample_pts[np.argmax(pi, axis=1)]

        if verbose:
            # Quality metrics
            cos_at_mapped = 1.0 - C_cos[
                torch.arange(n_c, device=device),
                torch.tensor(np.argmax(pi, axis=1), device=device)
            ].cpu().numpy()
            conf_at_mapped = conf[np.argmax(pi, axis=1)]
            logger.info(f"  Field mapped: {time.time()-t0:.1f}s, "
                        f"cos={cos_at_mapped.mean():.3f}, "
                        f"conf={conf_at_mapped.mean():.3f}")

        return mapped_coords

    def add_anchors(self, coords, z, weights=None):
        """Add new anchor points (e.g., mapped ATAC cells for refinement)."""
        coords = np.asarray(coords, dtype=np.float64)
        z = np.asarray(z, dtype=np.float32)
        if weights is None:
            weights = np.ones(len(coords), dtype=np.float64)
        else:
            weights = np.asarray(weights, dtype=np.float64)

        self.anchor_coords = np.vstack([self.anchor_coords, coords])
        self.anchor_z = np.vstack([self.anchor_z, z])
        self.anchor_weights = np.concatenate([self.anchor_weights, weights])
        self._nn_index = None  # reset cached index

    def iterative_map(self, z_st, coords_st, z_cells, init_coords=None,
                      n_refine=2, refine_weight=0.3,
                      lam_confidence=0.5, device='cuda', verbose=True):
        """
        Full pipeline: fit field → OT map on candidates → refine field → re-map.

        Args:
            z_st: (n_spots, z_dim) ST embeddings
            coords_st: (n_spots, D) ST coordinates
            z_cells: (n_cells, z_dim) cell embeddings
            init_coords: not used (positions come from OT on field candidates)
            n_refine: refinement rounds
            refine_weight: weight of mapped cells as field anchors (ST = 1.0)
            lam_confidence: confidence penalty in cost
            device: 'cuda' or 'cpu'
            verbose: print progress

        Returns:
            final_coords: (n_cells, D) continuous positions
        """
        import time
        t_start = time.time()

        # Round 0: Fit field on ST spots
        if verbose:
            print(f"[KernelField] {len(z_cells)} cells, {len(z_st)} spots")
            print(f"[Round 0] Fitting field on ST spots...")
        self.fit(coords_st, z_st)

        # Round 0: Map cells via OT on dense candidates
        current_coords = self.map_cells(
            z_cells, lam_confidence=lam_confidence,
            device=device, verbose=verbose)

        # Iterative refinement
        for r in range(n_refine):
            if verbose:
                print(f"[Round {r+1}] Refining with {len(z_cells)} mapped cells "
                      f"(weight={refine_weight})...")

            # Re-fit field with ST + mapped cells
            all_coords = np.vstack([coords_st, current_coords])
            all_z = np.vstack([z_st, z_cells])
            all_w = np.concatenate([
                np.ones(len(z_st)),
                np.full(len(z_cells), refine_weight)
            ])
            self.fit(all_coords, all_z, weights=all_w)

            # Re-map on updated field
            current_coords = self.map_cells(
                z_cells, lam_confidence=lam_confidence,
                device=device, verbose=verbose)

        if verbose:
            print(f"[Total] {time.time()-t_start:.1f}s")

        return current_coords
