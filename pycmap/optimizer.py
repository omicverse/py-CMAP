"""
Core optimization engine for Level 2 (OptimalSpot) mapping.

Adapted from CMAP/inst/python/cmap_optimizer.py, cmap_utils.py, and mapping01.py.
Merged into a single module with proper imports.

The Mapping class optimizes a cell-to-spot mapping matrix using:
- SSIM loss: measures expression pattern similarity
- Entropy regularization: encourages uniform cell distribution across spots
"""

import logging
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import softmax
from scipy.sparse import issparse

from . import ssim as pytorch_msssim

logging.getLogger(__name__).setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def entropy(data):
    """Compute Shannon entropy of discrete data (vectorized)."""
    if isinstance(data, np.ndarray):
        _, counts = np.unique(data, return_counts=True)
        probs = counts / len(data)
        return float(-np.sum(probs * np.log2(probs)))
    else:
        # Torch tensor — stay on GPU
        _, counts = data.unique(return_counts=True)
        probs = counts.float() / len(data)
        return float((-probs * probs.log2()).sum().item())


class Mapping:
    """
    Optimizes a cell-to-spot mapping matrix M (cells x spots).

    The mapping is trained via gradient descent on:
        loss = para_distance * (1 - SSIM) - para_density * H(assignments)

    where SSIM measures structural similarity between predicted and observed
    spot expression, and H is the Shannon entropy of cell-to-spot assignments.
    """

    def __init__(self, C, S, device="cpu", random_seed_set=None):
        """
        Args:
            C: Cell expression matrix (n_cells x n_genes), numpy array.
            S: Spot expression matrix (n_spots x n_genes), numpy array.
            device: 'cpu' or torch.device for GPU.
            random_seed_set: Int seed for reproducibility.
        """
        self.C = torch.tensor(C, device=device, dtype=torch.float32)
        self.S = torch.tensor(S, device=device, dtype=torch.float32)

        self.random_seed_set = random_seed_set
        if self.random_seed_set:
            np.random.seed(seed=self.random_seed_set)
        self.M = np.random.normal(0, 1, (C.shape[0], S.shape[0]))
        self.M = torch.tensor(
            self.M, device=device, requires_grad=True, dtype=torch.float32
        )

        # Pre-normalize spot expression to [0, 255] for SSIM (4D: B,C,H,W)
        denom = self.S.max() - self.S.min()
        denom = denom if denom > 0 else torch.tensor(1.0, device=device)
        self.S_n = (255 * (self.S - self.S.min()) / denom).unsqueeze(0).unsqueeze(0)

    def _loss_fn(self, para_distance, para_density, verbose=True):
        """Compute combined SSIM + entropy loss."""
        M_probs = softmax(self.M, dim=1)

        # Predicted spot expression: weighted sum of cell expressions
        S_pred_t = torch.matmul(M_probs.t(), self.C)

        # Normalize predicted expression to [0, 255] (4D for SSIM)
        denom = S_pred_t.max() - S_pred_t.min()
        denom = denom if denom > 0 else torch.tensor(1.0, device=S_pred_t.device)
        S_pred_n = (255 * (S_pred_t - S_pred_t.min()) / denom).unsqueeze(0).unsqueeze(0)

        # Entropy of cell assignments (stay on GPU)
        map_spot_index = torch.max(M_probs, 1)[1]
        H_info = entropy(map_spot_index)

        # Combined loss: minimize (1-SSIM), maximize entropy
        total_loss = para_distance * (1 - pytorch_msssim.ssim(S_pred_n, self.S_n, window_size=2)) \
                     - para_density * H_info

        return total_loss

    def train(self, num_epochs, para_distance, para_density, learning_rate=0.1, print_each=100):
        """
        Optimize the mapping matrix.

        Args:
            num_epochs: Number of optimization iterations.
            para_distance: Weight for SSIM similarity term.
            para_density: Weight for entropy regularization term.
            learning_rate: NAdam learning rate.
            print_each: Print loss every N epochs (None to suppress).

        Returns:
            Probability matrix (n_cells x n_spots), numpy array.
        """
        if self.random_seed_set:
            torch.manual_seed(seed=self.random_seed_set)
        optimizer = torch.optim.NAdam([self.M], lr=learning_rate)

        if print_each:
            logger.info(f"Training for {num_epochs} epochs...")

        for t in range(num_epochs):
            if print_each is None or t % print_each != 0:
                run_loss = self._loss_fn(para_distance, para_density, verbose=False)
            else:
                run_loss = self._loss_fn(para_distance, para_density, verbose=True)

            if print_each and t % print_each == 0:
                logger.info(f"Epoch {t}: loss = {run_loss.item():.6f}")

            optimizer.zero_grad()
            run_loss.backward()
            optimizer.step()

        with torch.no_grad():
            output = softmax(self.M, dim=1).cpu().numpy()
            return output


def map_cell_to_spot_optimize(C, S, device="cpu", learning_rate=0.1, num_epochs=1000,
                              verbose=True, random_seed_set=None,
                              para_distance=1.0, para_density=1.0):
    """
    High-level API for cell-to-spot optimization.

    Args:
        C: Cell expression matrix (n_cells x n_genes). Can be numpy, sparse, or AnnData.X.
        S: Spot expression matrix (n_spots x n_genes). Can be numpy, sparse, or AnnData.X.
        device: 'cpu' or torch.device.
        learning_rate: NAdam learning rate (default 0.1).
        num_epochs: Number of training epochs (default 1000).
        verbose: Whether to print progress.
        random_seed_set: Int seed for reproducibility.
        para_distance: SSIM weight (default 1.0).
        para_density: Entropy weight (default 1.0).

    Returns:
        Mapping probability matrix (n_cells x n_spots), numpy array.
    """
    # Handle sparse matrices
    if issparse(C):
        C = C.toarray().astype(np.float32)
    else:
        C = np.asarray(C, dtype=np.float32)

    if issparse(S):
        S = S.toarray().astype(np.float32)
    else:
        S = np.asarray(S, dtype=np.float32)

    device = torch.device(device)
    print_each = 100 if verbose else None

    logger.info("Starting cell-to-spot optimization...")
    mapper = Mapping(C=C, S=S, device=device, random_seed_set=random_seed_set)

    mapping_matrix = mapper.train(
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        print_each=print_each,
        para_distance=para_distance,
        para_density=para_density,
    )

    logger.info("Optimization complete.")
    return mapping_matrix
