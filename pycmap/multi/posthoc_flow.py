"""
Post-hoc Flow Matching on frozen GLUE embeddings.

Two-stage approach:
  Stage 1: Train GLUE with sampled reconstruction → high-quality embeddings
  Stage 2: Freeze embeddings, train velocity field v(z,t) on OT-coupled pairs

The velocity field captures the continuous transport from ATAC → ST
in the shared embedding space — modeling chromatin-to-expression dynamics.

Biological applications:
  - Regulatory potential: ||v(z,0)|| = distance between chromatin and expression states
  - Regulatory trajectories: path from ATAC→RNA per cell type
  - Cross-modal intermediate states: z(t=0.5) fuses ATAC and RNA information
  - Velocity arrows on UMAP: which cells are "activating" vs "silencing"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)


class VelocityField(nn.Module):
    """MLP velocity field v(z, t) for OT flow matching."""

    def __init__(self, dim, hidden_dim=256, n_layers=3):
        super().__init__()
        layers = [nn.Linear(dim + 1, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z, t):
        if isinstance(t, (int, float)):
            t = torch.full((z.shape[0], 1), t, device=z.device, dtype=z.dtype)
        elif t.dim() == 0:
            t = t.unsqueeze(0).unsqueeze(0).expand(z.shape[0], 1)
        elif t.dim() == 1:
            t = t.unsqueeze(1)
        return self.net(torch.cat([z, t], dim=1))


def sinkhorn_plan(z_a, z_b, epsilon=0.1, n_iter=50):
    """Compute entropic OT coupling."""
    C = torch.cdist(z_a, z_b, p=2)
    C = C / (C.max() + 1e-8)
    K = torch.exp(-C / epsilon)
    n, m = z_a.shape[0], z_b.shape[0]
    a, b = torch.ones(n, device=z_a.device) / n, torch.ones(m, device=z_b.device) / m
    u = torch.ones(n, device=z_a.device)
    for _ in range(n_iter):
        v = b / (K.T @ u + 1e-30)
        u = a / (K @ v + 1e-30)
    return torch.diag(u) @ K @ torch.diag(v)


def train_posthoc_flow(z_atac, z_st, n_epochs=5000, batch_size=512,
                       hidden_dim=512, n_layers=4, lr=5e-4,
                       k_neighbors=5, device='cuda'):
    """
    Train velocity field on frozen GLUE embeddings using NN pairing.

    Uses nearest-neighbor pairing instead of OT coupling: for each ATAC cell,
    pairs it with one of its K nearest ST spots. This gives cleaner training
    signal since GLUE already aligns the embedding spaces.

    Args:
        z_atac: (n_cells, dim) ATAC GLUE embeddings (frozen)
        z_st: (n_spots, dim) ST GLUE embeddings (frozen)
        n_epochs: training iterations
        batch_size: pairs per batch
        hidden_dim: velocity net width
        n_layers: velocity net depth
        lr: learning rate
        k_neighbors: number of nearest ST spots to sample from per ATAC cell
        device: 'cuda' or 'cpu'

    Returns:
        Trained VelocityField model
    """
    from sklearn.neighbors import NearestNeighbors

    z_a = torch.tensor(z_atac, device=device, dtype=torch.float32)
    z_b = torch.tensor(z_st, device=device, dtype=torch.float32)
    dim = z_a.shape[1]

    # NN pairing: for each ATAC cell, find K nearest ST spots
    nn = NearestNeighbors(n_neighbors=k_neighbors).fit(z_st)
    nn_dists, nn_indices = nn.kneighbors(z_atac)
    nn_indices_t = torch.tensor(nn_indices, device=device, dtype=torch.long)

    logger.info(f"NN pairing: K={k_neighbors}, mean dist={nn_dists[:, 0].mean():.3f}")

    # Train velocity field
    vf = VelocityField(dim, hidden_dim, n_layers).to(device)
    optimizer = torch.optim.Adam(vf.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    logger.info(f"Training velocity field ({n_epochs} epochs, {hidden_dim}x{n_layers})...")
    for epoch in range(n_epochs):
        # Sample ATAC cells, pair with random NN ST spot
        batch_idx = torch.randint(0, z_a.shape[0], (batch_size,), device=device)
        k_choice = torch.randint(0, k_neighbors, (batch_size,), device=device)
        st_idx = nn_indices_t[batch_idx, k_choice]

        za, zb = z_a[batch_idx], z_b[st_idx]

        # Random time
        t = torch.rand(batch_size, 1, device=device)

        # Interpolation and target
        z_t = (1 - t) * za + t * zb
        target = zb - za  # straight-line velocity

        # Predict and loss
        pred = vf(z_t, t)
        loss = F.mse_loss(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 1000 == 0:
            logger.info(f"  Epoch {epoch}: loss={loss.item():.6f}")

    logger.info(f"Training complete. Final loss: {loss.item():.6f}")
    return vf


def compute_regulatory_potential(vf, z_atac, device='cuda'):
    """
    Compute regulatory potential for each ATAC cell.

    ||v(z, t=0)|| = magnitude of velocity at chromatin state
    Large = chromatin is "ready" but expression hasn't followed yet (priming)
    Small = chromatin and expression are in sync (steady state)

    Returns:
        (n_cells,) array of regulatory potential scores
    """
    z = torch.tensor(z_atac, device=device, dtype=torch.float32)
    with torch.no_grad():
        v = vf(z, 0.0)  # velocity at t=0
        potential = torch.norm(v, dim=1).cpu().numpy()
    return potential


def compute_flow_trajectory(vf, z_atac, n_steps=10, device='cuda'):
    """
    Integrate the velocity field to get ATAC→ST trajectories.

    Returns:
        (n_cells, n_steps+1, dim) array of trajectory points
        trajectory[:, 0, :] = ATAC state (t=0)
        trajectory[:, -1, :] = predicted ST state (t=1)
    """
    z = torch.tensor(z_atac, device=device, dtype=torch.float32)
    dt = 1.0 / n_steps

    trajectory = [z.cpu().numpy()]
    with torch.no_grad():
        for step in range(n_steps):
            t = step * dt
            v = vf(z, t)
            z = z + v * dt  # Euler integration
            trajectory.append(z.cpu().numpy())

    return np.stack(trajectory, axis=1)  # (n_cells, n_steps+1, dim)


def compute_intermediate_embedding(vf, z_atac, t=0.5, device='cuda'):
    """
    Compute cross-modal intermediate embedding at time t.

    t=0: pure ATAC state
    t=0.5: fused ATAC+ST state
    t=1: predicted ST state

    Returns:
        (n_cells, dim) intermediate embedding
    """
    z = torch.tensor(z_atac, device=device, dtype=torch.float32)
    n_steps = max(int(t * 20), 1)
    dt = t / n_steps

    with torch.no_grad():
        for step in range(n_steps):
            t_cur = step * dt
            v = vf(z, t_cur)
            z = z + v * dt

    return z.cpu().numpy()
