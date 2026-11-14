"""
Level 2: OptimalSpot — Cell-to-spot mapping via optimization.

Replicates R functions from CellMapToSpot.R:
- map_spot_genes(): Find variable genes per spatial domain
- map_cell_to_spot(): Orchestrate per-domain optimization
"""

import logging
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import torch

from .domain_prediction import data_to_transform
from .optimizer import map_cell_to_spot_optimize

logger = logging.getLogger(__name__)


def find_variable_features(st_norm, spatial_location, sc_norm=None, nfeatures=3000):
    """
    Find variable genes per spatial domain using VST method.

    Equivalent to R: CMAP::map_spot_genes()
    Uses scanpy's highly_variable_genes with seurat_v3 flavor.

    Args:
        st_norm: Spatial expression matrix (genes x spots), DataFrame.
        spatial_location: DataFrame with 'HMRF_cluster' column.
        sc_norm: Single-cell expression matrix (genes x cells), DataFrame.
                 Used to intersect genes. Optional.
        nfeatures: Number of top variable features per domain.

    Returns:
        List of gene lists, one per domain.
    """
    clusters = sorted(spatial_location['HMRF_cluster'].unique())
    st_gene_list = []

    for cluster_id in clusters:
        # Get spots in this cluster
        spot_ids = spatial_location[spatial_location['HMRF_cluster'] == cluster_id].index
        spot_ids = [s for s in spot_ids if s in st_norm.columns]

        if len(spot_ids) == 0:
            st_gene_list.append([])
            continue

        sub_data = st_norm[spot_ids].values.T  # spots x genes
        gene_names = np.array(st_norm.index)

        # Filter zero-variance genes
        gene_var = np.var(sub_data, axis=0)
        nonzero_var_mask = gene_var > 0
        sub_data = sub_data[:, nonzero_var_mask]
        gene_names_filtered = gene_names[nonzero_var_mask]

        if sub_data.shape[1] == 0:
            st_gene_list.append([])
            continue

        # Create AnnData and find variable features
        adata = ad.AnnData(
            X=sub_data,
            var=pd.DataFrame(index=gene_names_filtered)
        )

        n_top = min(nfeatures, adata.n_vars)
        try:
            sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=n_top)
            hvgs = adata.var_names[adata.var['highly_variable']].tolist()
        except Exception:
            # Fallback: use variance-based selection
            variances = np.var(sub_data, axis=0)
            top_idx = np.argsort(variances)[::-1][:n_top]
            hvgs = gene_names_filtered[top_idx].tolist()

        # Intersect with sc genes if provided
        if sc_norm is not None:
            hvgs = [g for g in hvgs if g in sc_norm.index]

        st_gene_list.append(hvgs)

    return st_gene_list


def map_cell_to_spot(sc_norm, sc_meta, st_norm, spatial_location,
                     pred_sc_svm, pred_st_svm, batch=True,
                     st_gene_list=None, pca_method='truncated_svd',
                     num_epochs=2000, para_distance=1.0, para_density=1.0,
                     device=None, random_seed_set=None):
    """
    Level 2 mapping: assign cells to optimal spots within each spatial domain.

    Equivalent to R: CMAP::map_cell_to_spot()

    For each spatial domain:
    1. Subset cells and spots to that domain
    2. Compute domain-specific embedding via data_to_transform
    3. Run optimization to find optimal cell-to-spot mapping
    4. Assign each cell to the spot with highest probability

    Args:
        sc_norm: Single-cell expression (genes x cells), DataFrame.
        sc_meta: Cell metadata DataFrame.
        st_norm: Spatial expression (genes x spots), DataFrame.
        spatial_location: Spot metadata DataFrame with 'HMRF_cluster'.
        pred_sc_svm: Series with predicted domain labels for cells (index = cell IDs).
        pred_st_svm: Series with predicted domain labels for spots (index = spot IDs).
        batch: Whether to apply batch correction.
        st_gene_list: Pre-computed variable gene lists per domain. If None, computed.
        pca_method: PCA method for data_to_transform.
        num_epochs: Number of optimization epochs.
        para_distance: SSIM weight.
        para_density: Entropy weight.
        device: 'cpu' or 'cuda'. Auto-detected if None.
        random_seed_set: Random seed for reproducibility.

    Returns:
        DataFrame with columns: Single_cell, Spot, Probability.
    """
    # Auto-detect device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Compute variable genes if not provided
    if st_gene_list is None:
        st_gene_list = find_variable_features(st_norm, spatial_location, sc_norm)

    clusters = sorted(pred_sc_svm.unique())
    cell_spot_map_list = []

    for j, cluster_id in enumerate(clusters):
        logger.info(f"Processing domain {cluster_id} ({j+1}/{len(clusters)})...")

        # Get cell and spot IDs for this domain
        single_cell_ids = pred_sc_svm[pred_sc_svm == cluster_id].index.tolist()
        spot_ids = pred_st_svm[pred_st_svm == cluster_id].index.tolist()

        if len(single_cell_ids) == 0 or len(spot_ids) == 0:
            logger.warning(f"Domain {cluster_id}: no cells or spots, skipping.")
            continue

        # Get genes for this domain (match by domain order in find_variable_features)
        hvg_clusters = sorted(spatial_location['HMRF_cluster'].unique())
        domain_pos = hvg_clusters.index(cluster_id) if cluster_id in hvg_clusters else 0
        gene_list_idx = min(domain_pos, len(st_gene_list) - 1)
        domain_genes = st_gene_list[gene_list_idx] if st_gene_list else []
        if len(domain_genes) == 0:
            # Fallback to all shared genes
            domain_genes = list(set(sc_norm.index) & set(st_norm.index))

        # Compute domain-specific embedding
        sub_embedding = data_to_transform(
            sc_norm=sc_norm[single_cell_ids],
            st_norm=st_norm[spot_ids],
            spatial_genes=domain_genes,
            batch=batch,
            pca_method=pca_method,
        )

        # Extract cell and spot embeddings (PCs x samples -> samples x PCs)
        cellxgene = sub_embedding[single_cell_ids].T.values  # cells x PCs
        spotxgene = sub_embedding[spot_ids].T.values          # spots x PCs

        # Run optimization
        cell_spot = map_cell_to_spot_optimize(
            C=cellxgene,
            S=spotxgene,
            device=device,
            num_epochs=num_epochs,
            learning_rate=0.1,
            random_seed_set=random_seed_set,
            para_distance=para_distance,
            para_density=para_density,
            verbose=True,
        )

        # Build mapping results with uncertainty metrics
        best_spot_idx = np.argmax(cell_spot, axis=1)
        best_prob = np.max(cell_spot, axis=1)

        # Second-best probability: indicates ambiguity
        sorted_probs = np.sort(cell_spot, axis=1)
        second_prob = sorted_probs[:, -2]

        # Per-cell entropy: higher = more uncertain mapping
        cell_spot_clipped = np.clip(cell_spot, 1e-10, 1.0)
        cell_entropy = -np.sum(cell_spot_clipped * np.log2(cell_spot_clipped), axis=1)

        # Max theoretical entropy for this domain (uniform distribution)
        max_entropy = max(np.log2(len(spot_ids)), 1e-10)

        sub_map = pd.DataFrame({
            'Single_cell': single_cell_ids,
            'Spot': [spot_ids[idx] for idx in best_spot_idx],
            'Probability': best_prob,
            'Second_prob': second_prob,
            'Entropy': cell_entropy,
            'Norm_entropy': cell_entropy / max_entropy,  # 0~1, higher = less certain
        })

        cell_spot_map_list.append(sub_map)

    cell_spot_map = pd.concat(cell_spot_map_list, ignore_index=True)

    # Attach gene list for reuse in Level 3 (avoids re-computation)
    cell_spot_map.attrs['st_gene_list'] = st_gene_list

    return cell_spot_map
