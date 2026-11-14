"""
CMAP end-to-end pipeline with AnnData input/output.

Accepts AnnData objects directly, stores results in adata.obs and adata.obsm.
No CSV intermediate files needed.
"""

import logging
import time
import numpy as np
import pandas as pd
import anndata as ad
import torch

from .utils import log_normalize, quantile_normalize
from .domain_prediction import data_to_transform, tune_parameter, predict_domain
from .spot_mapping import map_cell_to_spot, find_variable_features
from .exact_location import spatial_relation_all, cor_spot_cell, calculate_cell_location

logger = logging.getLogger(__name__)


def _adata_to_df(adata, layer=None, genes=None):
    """Convert AnnData to genes x cells DataFrame.

    Only densifies the selected genes subset to minimize memory usage.
    """
    if layer and layer in adata.layers:
        X = adata.layers[layer]
    else:
        X = adata.X

    if genes is not None:
        # Only extract the needed genes → much smaller dense matrix
        gene_mask = adata.var_names.isin(genes)
        X = X[:, gene_mask]
        var_names = adata.var_names[gene_mask]
    else:
        var_names = adata.var_names

    from scipy.sparse import issparse
    if issparse(X):
        X = X.toarray()
    return pd.DataFrame(X.T, index=var_names, columns=adata.obs_names)


def run_cmap(adata_sc, adata_st, spatial_genes, spatial_data_type='honeycomb',
             cluster_key='HMRF_cluster', celltype_key=None,
             svm_threshold=0.5, num_epochs=2000,
             para_distance=1.0, para_density=1.0,
             pca_method_l1='truncated_svd', pca_method_l2='pca',
             n_near_spot=None, dis_cut=None, radius=0.5,
             device=None, cross_para=None, verbose=True):
    """
    Run the complete CMAP pipeline on AnnData objects.

    Args:
        adata_sc: AnnData for scRNA-seq. Must contain raw or log-normalized counts.
                  If adata_sc.X is raw counts, log-normalization is applied automatically.
        adata_st: AnnData for spatial transcriptomics. Must contain:
                  - .obs[cluster_key]: HMRF spatial domain labels
                  - .obsm['spatial']: spatial coordinates (n_spots x 2)
        spatial_genes: List of spatial genes for Level 1.
        spatial_data_type: 'honeycomb' (Visium), 'square' (ST), 'singlecell_level' (MERFISH).
        cluster_key: Column name in adata_st.obs for domain labels.
        celltype_key: Column name in adata_sc.obs for cell type annotations (optional).
        svm_threshold: Confidence threshold for Level 1 filtering (0.5 default, 0.8 strict).
        num_epochs: Number of optimizer epochs for Level 2.
        para_distance: SSIM weight for Level 2.
        para_density: Entropy weight for Level 2.
        pca_method_l1: PCA method for Level 1 ('truncated_svd' or 'pca').
        pca_method_l2: PCA method for Level 2 ('pca' recommended).
        n_near_spot: Override number of neighbors for Level 3.
        dis_cut: Override distance cutoff for Level 3.
        radius: Random jitter radius for isolated spots in Level 3.
        device: 'cpu' or 'cuda'. Auto-detected if None.
        cross_para: Cross-validation folds for SVM tuning (default [4]).
        verbose: Print progress.

    Returns:
        adata_sc: Modified in-place with new fields:
            .obs['CMAP_domain']: Predicted spatial domain
            .obs['CMAP_svm_prob']: SVM prediction probability
            .obs['CMAP_spot']: Assigned spot ID
            .obs['CMAP_spot_prob']: Spot mapping probability
            .obs['CMAP_entropy']: Mapping entropy
            .obs['CMAP_norm_entropy']: Normalized entropy (0-1)
            .obs['CMAP_quality']: Composite quality score
            .obsm['spatial_CMAP']: Predicted (x, y) coordinates (n_cells x 2)

        Also returns cell_spot_map DataFrame for detailed inspection.
    """
    if cross_para is None:
        cross_para = [4]
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    t_start = time.time()

    # ========== Prepare data ==========
    if verbose:
        logger.info("Preparing data...")

    # Convert to genes x cells/spots DataFrames
    sc_df = _adata_to_df(adata_sc)
    st_df = _adata_to_df(adata_st)

    # Auto-detect if normalization is needed (max > 50 suggests raw counts)
    if sc_df.values.max() > 50:
        if verbose:
            logger.info("Auto-normalizing scRNA-seq (detected raw counts)")
        sc_df = sc_df.loc[sc_df.sum(axis=1) > 0]
        sc_df = log_normalize(sc_df)

    if st_df.values.max() > 50:
        if verbose:
            logger.info("Auto-normalizing spatial data (detected raw counts)")
        st_df = st_df.loc[st_df.sum(axis=1) > 0]
        st_df = log_normalize(st_df)

    # Build spatial_location DataFrame from adata_st
    spatial_location = adata_st.obs[[cluster_key]].copy()
    spatial_location.columns = ['HMRF_cluster']
    if 'spatial' in adata_st.obsm:
        coords = adata_st.obsm['spatial']
        spatial_location['x'] = coords[:, 0]
        spatial_location['y'] = coords[:, 1]
    elif 'x' in adata_st.obs.columns and 'y' in adata_st.obs.columns:
        spatial_location['x'] = adata_st.obs['x'].values
        spatial_location['y'] = adata_st.obs['y'].values
    else:
        raise ValueError("Spatial coordinates not found. Provide adata_st.obsm['spatial'] or adata_st.obs[['x','y']].")
    print(f"[DEBUG] spatial_location rows={len(spatial_location)}, "
          f"xy_nan={spatial_location[['x','y']].isna().any(axis=1).sum()}", flush=True)

    sc_meta = adata_sc.obs.copy()
    n_cells = sc_df.shape[1]
    n_spots = st_df.shape[1]

    if verbose:
        logger.info(f"Data: {n_cells} cells, {n_spots} spots, {sc_df.shape[0]} genes")

    # ========== Level 1: Domain Prediction ==========
    t = time.time()
    if verbose:
        logger.info("Level 1: Domain prediction...")

    matrix = data_to_transform(
        sc_df, st_df, spatial_genes, batch=True, pca_method=pca_method_l1
    )

    st_cols = [c for c in matrix.columns if c in st_df.columns]
    sc_cols = [c for c in matrix.columns if c in sc_df.columns]
    train_set = matrix[st_cols].T.copy()
    train_set['label'] = spatial_location.loc[st_cols, 'HMRF_cluster'].values
    test_set = matrix[sc_cols].T.copy()

    params = tune_parameter(train_set, test_set, cross_para=cross_para, verbose=False)
    best = params[f'cross_{cross_para[0]}']

    pred_st, _ = predict_domain(
        train_set, test_set, cost=best['cost'], gamma=best['gamma'], scale=True, st_svm=True)
    pred_sc, prob_sc = predict_domain(
        train_set, test_set, cost=best['cost'], gamma=best['gamma'], scale=True)

    max_prob = prob_sc.max(axis=1)

    # Store Level 1 results in adata
    adata_sc.obs['CMAP_domain'] = pd.NA
    adata_sc.obs.loc[pred_sc.index, 'CMAP_domain'] = pred_sc.values
    adata_sc.obs['CMAP_svm_prob'] = pd.NA
    adata_sc.obs.loc[max_prob.index, 'CMAP_svm_prob'] = max_prob.values

    # Filter
    high_conf = max_prob > svm_threshold
    sc_meta_f = sc_meta.loc[high_conf[high_conf].index]
    pred_sc_f = pred_sc[high_conf]
    sc_df_f = sc_df[sc_meta_f.index]

    if verbose:
        n_filtered = n_cells - len(sc_meta_f)
        logger.info(f"  Level 1 done: {time.time()-t:.1f}s, "
                    f"{len(sc_meta_f)} cells kept, {n_filtered} filtered (prob<={svm_threshold})")

    # Stop here, not forty lines downstream. With every cell filtered out,
    # Level 2 receives an empty set, skips every domain, and dies in
    # `pd.concat` with "No objects to concatenate" — which says nothing about
    # the cause. The cause is always one of two things, and both are visible
    # right here.
    if len(sc_meta_f) == 0:
        raise ValueError(
            f"Level 1 kept 0 of {n_cells} cells: no cell's domain probability "
            f"exceeded svm_threshold={svm_threshold} (highest observed "
            f"{float(max_prob.max()):.3f}, median {float(max_prob.median()):.3f}, "
            f"{len(set(pred_st))} domains, so chance is "
            f"{1 / max(len(set(pred_st)), 1):.3f}).\n"
            "Either the domain classifier is near chance — `spatial_genes` do "
            "not separate the domains in `cluster_key`, which is what to check "
            "first — or the threshold is too strict for this dataset. Inspect "
            "the probabilities before lowering it; a confident placement built "
            "on a chance-level classifier is worse than none."
        )

    # ========== Level 2: Spot Mapping ==========
    t = time.time()
    if verbose:
        logger.info("Level 2: Spot mapping...")

    cell_spot_map = map_cell_to_spot(
        sc_norm=sc_df_f, sc_meta=sc_meta_f,
        st_norm=st_df, spatial_location=spatial_location,
        pred_sc_svm=pred_sc_f, pred_st_svm=pred_st,
        batch=True, pca_method=pca_method_l2,
        num_epochs=num_epochs, para_distance=para_distance,
        para_density=para_density, device=device,
    )

    # Store Level 2 results
    csm = cell_spot_map.set_index('Single_cell')
    for col in ['Spot', 'Probability', 'Entropy', 'Norm_entropy']:
        adata_col = f'CMAP_{col.lower()}' if col != 'Spot' else 'CMAP_spot'
        if col == 'Spot':
            adata_col = 'CMAP_spot'
        elif col == 'Probability':
            adata_col = 'CMAP_spot_prob'
        elif col == 'Entropy':
            adata_col = 'CMAP_entropy'
        elif col == 'Norm_entropy':
            adata_col = 'CMAP_norm_entropy'
        adata_sc.obs[adata_col] = pd.NA
        adata_sc.obs.loc[csm.index.intersection(adata_sc.obs_names), adata_col] = \
            csm.loc[csm.index.intersection(adata_sc.obs_names), col].values

    if verbose:
        logger.info(f"  Level 2 done: {time.time()-t:.1f}s, {len(cell_spot_map)} cells mapped")

    # ========== Pre-compute correlation ==========
    t = time.time()
    st_gene_list = cell_spot_map.attrs.get('st_gene_list', None)
    shvgs = list(set(g for gl in st_gene_list for g in gl)) if st_gene_list else spatial_genes
    cell_spot_cor = cor_spot_cell(
        sc_df_f, st_df, shvgs, batch=True, pca_method=pca_method_l1
    )
    if verbose:
        logger.info(f"  Correlation computed: {time.time()-t:.1f}s")

    # ========== Level 3: Precise Location ==========
    t = time.time()
    if verbose:
        logger.info("Level 3: Precise location...")

    spot_neigh_list = spatial_relation_all(
        spatial_location, spatial_data_type=spatial_data_type,
        n_near_spot=n_near_spot, dis_cut=dis_cut,
    )

    sc_meta_coord = calculate_cell_location(
        cell_spot_map=cell_spot_map, st_meta=spatial_location,
        sc_meta=sc_meta_f, sc_norm=sc_df_f, st_norm=st_df,
        batch=True, spot_neigh_list=spot_neigh_list,
        radius=radius, cell_spot_cor=cell_spot_cor,
    )

    # Store coordinates in obsm['spatial_CMAP']
    coords_array = np.full((adata_sc.n_obs, 2), np.nan)
    mapped_cells = sc_meta_coord.index.intersection(adata_sc.obs_names)
    name_to_idx = {n: i for i, n in enumerate(adata_sc.obs_names)}
    cell_idx = [name_to_idx[c] for c in mapped_cells]
    coords_array[cell_idx, 0] = sc_meta_coord.loc[mapped_cells, 'pred_loc_x'].values.astype(float)
    coords_array[cell_idx, 1] = sc_meta_coord.loc[mapped_cells, 'pred_loc_y'].values.astype(float)
    adata_sc.obsm['spatial_CMAP'] = coords_array

    # Composite quality score
    svm_prob = pd.to_numeric(adata_sc.obs['CMAP_svm_prob'], errors='coerce')
    norm_ent = pd.to_numeric(adata_sc.obs['CMAP_norm_entropy'], errors='coerce')
    adata_sc.obs['CMAP_quality'] = svm_prob * (1.0 - norm_ent)

    if verbose:
        logger.info(f"  Level 3 done: {time.time()-t:.1f}s, {len(sc_meta_coord)} cells positioned")
        logger.info(f"Total pipeline: {time.time()-t_start:.1f}s")

    return cell_spot_map
