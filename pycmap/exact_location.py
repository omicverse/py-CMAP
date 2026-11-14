"""
Level 3: PreciseLocation — Sub-spot coordinate calculation.

Replicates R functions from ExactLocation.R:
- spatial_relation_all(): Build spatial neighborhood graph
- neighbor_spot(): Extract directional neighbors
- cor_spot_cell(): Compute spot-cell Spearman correlations
- calculate_cell_location(): Weighted coordinate interpolation
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed

from .domain_prediction import data_to_transform
from .spot_mapping import find_variable_features
from .utils import expand_diff


def spatial_relation_all(spatial_location, spatial_data_type='honeycomb',
                         n_near_spot=None, dis_cut=None):
    """
    Compute spatial neighborhood structure for spots.

    Equivalent to R: CMAP::spatial_relation_all()

    Args:
        spatial_location: DataFrame with 'x' and 'y' columns, indexed by spot IDs.
        spatial_data_type: 'honeycomb', 'square', or 'singlecell_level'.
        n_near_spot: Override number of neighbors.
        dis_cut: Override distance cutoff.

    Returns:
        Dict with keys:
        - 'all': dict of spot -> list of neighbor spots (including self)
        - 'left_right': dict of spot -> list of left/right neighbors
        - 'up_down': dict of spot -> list of up/down neighbors
    """
    # Default parameters by data type
    defaults = {
        'honeycomb': (6, 30),
        'square': (4, 30),
        'singlecell_level': (4, 10),
    }
    top_n, cutoff = defaults.get(spatial_data_type, (6, 30))

    if n_near_spot is not None:
        top_n = n_near_spot
    if dis_cut is not None:
        cutoff = dis_cut

    coords = spatial_location[['x', 'y']].dropna().values
    spot_ids = spatial_location[['x', 'y']].dropna().index.tolist()
    n_spots = len(spot_ids)

    # kNN search
    k = min(top_n, n_spots - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
    nn.fit(coords)
    distances, indices = nn.kneighbors(coords)

    # Build neighbor list (exclude self, which is index 0 with distance 0)
    df_j_all = {}
    for i in range(n_spots):
        neighbors = []
        for j_idx in range(1, k + 1):  # skip self (index 0)
            if distances[i, j_idx] < cutoff:
                neighbors.append(spot_ids[indices[i, j_idx]])
        neighbors.append(spot_ids[i])  # include self
        df_j_all[spot_ids[i]] = neighbors

    if spatial_data_type == 'singlecell_level':
        return {
            'all': df_j_all,
            'left_right': df_j_all.copy(),
            'up_down': df_j_all.copy(),
        }

    # Classify neighbors as left-right or up-down
    df_j_up_down = {}
    df_j_left_right = {}

    for spot_i in spot_ids:
        if len(df_j_all[spot_i]) <= 1:
            df_j_up_down[spot_i] = []
            df_j_left_right[spot_i] = []
            continue

        central_y = spatial_location.loc[spot_i, 'y']
        central_x = spatial_location.loc[spot_i, 'x']
        cent_s_y = {round(central_y), int(np.ceil(central_y)), int(np.floor(central_y))}
        # Note: R code uses central_spot$y for cent_s_x too (appears to be a bug in original)
        cent_s_x = {round(central_y), int(np.ceil(central_y)), int(np.floor(central_y))}

        neighbor_ids = [s for s in df_j_all[spot_i] if s != spot_i]

        # Check if any neighbor has same y as central (left-right neighbor)
        lr_neighbors = []
        ud_neighbors = []
        found_lr = False

        for n_id in neighbor_ids:
            n_y = spatial_location.loc[n_id, 'y']
            if round(n_y) in cent_s_y:
                lr_neighbors.append(n_id)
                found_lr = True

        if found_lr:
            ud_neighbors = [n for n in neighbor_ids if n not in lr_neighbors]
        else:
            # Check x alignment
            found_ud = False
            for n_id in neighbor_ids:
                n_x = spatial_location.loc[n_id, 'x']
                if round(n_x) in cent_s_x:
                    ud_neighbors.append(n_id)
                    found_ud = True

            if found_ud:
                lr_neighbors = [n for n in neighbor_ids if n not in ud_neighbors]
            else:
                # Fallback: all neighbors in both
                lr_neighbors = neighbor_ids
                ud_neighbors = neighbor_ids

        df_j_left_right[spot_i] = lr_neighbors
        df_j_up_down[spot_i] = ud_neighbors

    return {
        'all': df_j_all,
        'left_right': df_j_left_right,
        'up_down': df_j_up_down,
    }


def neighbor_spot(spot_neigh_list, top_sim_spot):
    """
    Extract directional neighbors for a given spot.

    Equivalent to R: CMAP::neighbor_spot()

    Args:
        spot_neigh_list: Dict from spatial_relation_all().
        top_sim_spot: Spot ID string.

    Returns:
        Dict with 'neigh_UD' and 'neigh_LR' lists.
    """
    ud = spot_neigh_list['up_down'].get(top_sim_spot, [])
    lr = spot_neigh_list['left_right'].get(top_sim_spot, [])

    if len(ud) > 0 and len(lr) > 0:
        neigh_UD = ud
        neigh_LR = lr
    elif len(ud) == 0 and len(lr) > 0:
        neigh_UD = [top_sim_spot]
        neigh_LR = lr
    elif len(ud) > 0 and len(lr) == 0:
        neigh_UD = ud
        neigh_LR = [top_sim_spot]
    else:
        neigh_LR = [top_sim_spot]
        neigh_UD = [top_sim_spot]

    return {'neigh_UD': neigh_UD, 'neigh_LR': neigh_LR}


def cor_spot_cell(sc_norm, st_norm, genes, batch=True, pca_method='truncated_svd',
                  parallel=False, n_jobs=-1):
    """
    Compute Spearman correlation between cells and spots.

    Equivalent to R: CMAP::cor_spot_cell()
    Optimized: uses rank-based vectorized computation.

    Args:
        sc_norm: Single-cell expression (genes x cells), DataFrame.
        st_norm: Spatial expression (genes x spots), DataFrame.
        genes: List of genes to use.
        batch: Whether to apply batch correction.
        pca_method: PCA method.
        parallel: Ignored (kept for API compatibility).
        n_jobs: Ignored.

    Returns:
        Correlation matrix (cells x spots), DataFrame.
    """
    matrix = data_to_transform(
        sc_norm=sc_norm, st_norm=st_norm,
        spatial_genes=genes, batch=batch, pca_method=pca_method,
    )

    sc_cols = [c for c in matrix.columns if c in sc_norm.columns]
    st_cols = [c for c in matrix.columns if c in st_norm.columns]

    sc_data = matrix[sc_cols].values  # PCs x cells
    st_data = matrix[st_cols].values  # PCs x spots

    # Vectorized Spearman: rank each column, then Pearson on ranks
    from scipy.stats import rankdata
    sc_ranks = np.apply_along_axis(rankdata, 0, sc_data)  # PCs x cells
    st_ranks = np.apply_along_axis(rankdata, 0, st_data)  # PCs x spots

    # Standardize ranks
    sc_ranks = (sc_ranks - sc_ranks.mean(axis=0, keepdims=True))
    sc_std = np.sqrt((sc_ranks ** 2).sum(axis=0, keepdims=True))
    sc_std[sc_std == 0] = 1
    sc_ranks = sc_ranks / sc_std

    st_ranks = (st_ranks - st_ranks.mean(axis=0, keepdims=True))
    st_std = np.sqrt((st_ranks ** 2).sum(axis=0, keepdims=True))
    st_std[st_std == 0] = 1
    st_ranks = st_ranks / st_std

    # Pearson on ranks = Spearman: (cells x PCs) @ (PCs x spots)
    cor_matrix = sc_ranks.T @ st_ranks  # cells x spots

    return pd.DataFrame(cor_matrix, index=sc_cols, columns=st_cols)


def calculate_cell_location(cell_spot_map, st_meta, sc_meta, sc_norm, st_norm,
                            batch=True, pca_method='truncated_svd', parallel=False,
                            st_gene_list=None, spot_neigh_list=None, radius=0.5,
                            cell_spot_cor=None):
    """
    Calculate precise sub-spot coordinates for each cell.

    Equivalent to R: CMAP::calculate_cell_location()

    For each mapped cell:
    1. Find its assigned spot and neighbors
    2. Compute Spearman correlation with neighbors
    3. Apply expand_diff() transformation
    4. Calculate weighted average position

    Args:
        cell_spot_map: DataFrame with Single_cell, Spot, Probability columns.
        st_meta: Spot metadata with 'x', 'y' columns.
        sc_meta: Cell metadata DataFrame.
        sc_norm: Single-cell expression (genes x cells), DataFrame.
        st_norm: Spatial expression (genes x spots), DataFrame.
        batch: Whether to apply batch correction.
        pca_method: PCA method.
        parallel: Whether to parallelize correlation computation.
        st_gene_list: Pre-computed variable gene lists per domain.
        spot_neigh_list: Pre-computed neighborhood structure.
        radius: Random jitter radius for isolated spots.
        cell_spot_cor: Pre-computed cell-spot correlation matrix (cells x spots).
                       If provided, skips the expensive cor_spot_cell() call.

    Returns:
        DataFrame with original sc_meta columns plus 'pred_loc_x', 'pred_loc_y'.
    """
    if cell_spot_cor is None:
        # Get variable genes
        if st_gene_list is None:
            # Try to get from cell_spot_map attrs (saved by map_cell_to_spot)
            st_gene_list = getattr(cell_spot_map, 'attrs', {}).get('st_gene_list', None)
        if st_gene_list is None:
            st_gene_list = find_variable_features(st_norm, st_meta, sc_norm)
        shvgs = list(set(g for gl in st_gene_list for g in gl))

        # Compute cell-spot correlations
        cell_spot_cor = cor_spot_cell(
            sc_norm=sc_norm, st_norm=st_norm, genes=shvgs,
            batch=batch, pca_method=pca_method, parallel=parallel,
        )

    cell_ids = cell_spot_map['Single_cell'].values
    # Filter to cells that exist in sc_meta
    valid_cell_ids = [c for c in cell_ids if c in sc_meta.index]
    cell_ids = np.array(valid_cell_ids)
    sc_meta_coord = sc_meta.loc[cell_ids].copy()

    # Use numpy arrays to accumulate, assign to DataFrame once at end
    _pred_x = np.full(len(cell_ids), np.nan)
    _pred_y = np.full(len(cell_ids), np.nan)

    df_j_all = spot_neigh_list['all']

    rng = np.random.default_rng(42)

    # Pre-index: build cell -> spot mapping dict for O(1) lookup
    cell_to_spots = cell_spot_map.groupby('Single_cell')['Spot'].apply(list).to_dict()

    # Pre-extract numpy arrays from correlation DataFrame for fast indexing
    cor_index = list(cell_spot_cor.index)
    cor_columns = list(cell_spot_cor.columns)
    cor_values = cell_spot_cor.values
    cor_idx_map = {c: i for i, c in enumerate(cor_index)}
    cor_col_map = {c: i for i, c in enumerate(cor_columns)}

    # Pre-extract spot coordinates as dict for fast access
    st_x = st_meta['x'].to_dict()
    st_y = st_meta['y'].to_dict()

    for i, cell_id in enumerate(cell_ids):
        top_sim_spots = cell_to_spots.get(cell_id, [])

        if len(top_sim_spots) == 1:
            top_sim_spot = str(top_sim_spots[0])
            top_spot_neighbors = df_j_all.get(top_sim_spot, [top_sim_spot])

            if len(top_spot_neighbors) > 1:
                neigh = neighbor_spot(spot_neigh_list, top_sim_spot)
                neigh_LR = neigh['neigh_LR']
                neigh_UD = neigh['neigh_UD']

                valid_lr = [s for s in neigh_LR if s in cor_col_map]
                valid_ud = [s for s in neigh_UD if s in cor_col_map]

                if cell_id not in cor_idx_map:
                    _pred_x[i] = st_x[top_sim_spot] + rng.uniform(-radius, radius)
                    _pred_y[i] = st_y[top_sim_spot] + rng.uniform(-radius, radius)
                    continue

                ci = cor_idx_map[cell_id]
                rela_LR = cor_values[ci, [cor_col_map[s] for s in valid_lr]] if valid_lr else np.array([])
                rela_UD = cor_values[ci, [cor_col_map[s] for s in valid_ud]] if valid_ud else np.array([])
                rela_top = cor_values[ci, cor_col_map[top_sim_spot]] if top_sim_spot in cor_col_map else 0.0

                all_corrs = np.concatenate([rela_LR, rela_UD, [rela_top]])
                coeff_expd = expand_diff(all_corrs)

                n_LR = len(rela_LR)
                n_UD = len(rela_UD)

                coord_LR_x = np.array([st_x[s] for s in valid_lr]) if valid_lr else np.array([])
                coord_top_x = st_x[top_sim_spot]

                if n_LR > 0:
                    pred_x = (np.sum(coeff_expd[:n_LR] * coord_LR_x) + coeff_expd[-1] * coord_top_x) / \
                             (np.sum(coeff_expd[:n_LR]) + coeff_expd[-1])
                else:
                    pred_x = coord_top_x + rng.uniform(-radius, radius)

                # Weighted y from UD neighbors
                coord_UD_y = np.array([st_y[s] for s in valid_ud]) if valid_ud else np.array([])
                coord_top_y = st_y[top_sim_spot]

                if n_UD > 0:
                    pred_y = (np.sum(coeff_expd[n_LR:n_LR + n_UD] * coord_UD_y) + coeff_expd[-1] * coord_top_y) / \
                             (np.sum(coeff_expd[n_LR:n_LR + n_UD]) + coeff_expd[-1])
                else:
                    pred_y = coord_top_y + rng.uniform(-radius, radius)

                _pred_x[i] = pred_x
                _pred_y[i] = pred_y
            else:
                _pred_x[i] = st_x[top_sim_spot] + rng.uniform(-radius, radius)
                _pred_y[i] = st_y[top_sim_spot] + rng.uniform(-radius, radius)

        elif len(top_sim_spots) > 1:
            for k, top_sim_spot in enumerate(top_sim_spots):
                top_sim_spot = str(top_sim_spot)
                top_spot_neighbors = df_j_all.get(top_sim_spot, [top_sim_spot])
                new_id = f"{cell_id}_{k}" if k > 0 else cell_id

                if len(top_spot_neighbors) > 1 and cell_id in cor_idx_map:
                    neigh = neighbor_spot(spot_neigh_list, top_sim_spot)
                    neigh_LR = neigh['neigh_LR']
                    neigh_UD = neigh['neigh_UD']

                    valid_lr = [s for s in neigh_LR if s in cor_col_map]
                    valid_ud = [s for s in neigh_UD if s in cor_col_map]

                    ci = cor_idx_map[cell_id]
                    rela_LR = cor_values[ci, [cor_col_map[s] for s in valid_lr]] if valid_lr else np.array([])
                    rela_UD = cor_values[ci, [cor_col_map[s] for s in valid_ud]] if valid_ud else np.array([])
                    rela_top = cor_values[ci, cor_col_map[top_sim_spot]] if top_sim_spot in cor_col_map else 0.0

                    all_corrs = np.concatenate([rela_LR, rela_UD, [rela_top]])
                    coeff_expd = expand_diff(all_corrs)

                    n_LR = len(rela_LR)
                    n_UD = len(rela_UD)

                    coord_LR_x = np.array([st_x[s] for s in valid_lr]) if valid_lr else np.array([])
                    coord_top_x = st_x[top_sim_spot]
                    coord_UD_y = np.array([st_y[s] for s in valid_ud]) if valid_ud else np.array([])
                    coord_top_y = st_y[top_sim_spot]

                    pred_x = (np.sum(coeff_expd[:n_LR] * coord_LR_x) + coeff_expd[-1] * coord_top_x) / \
                             (np.sum(coeff_expd[:n_LR]) + coeff_expd[-1]) if n_LR > 0 else coord_top_x
                    pred_y = (np.sum(coeff_expd[n_LR:n_LR + n_UD] * coord_UD_y) + coeff_expd[-1] * coord_top_y) / \
                             (np.sum(coeff_expd[n_LR:n_LR + n_UD]) + coeff_expd[-1]) if n_UD > 0 else coord_top_y
                else:
                    pred_x = st_x[top_sim_spot] + rng.uniform(-radius, radius)
                    pred_y = st_y[top_sim_spot] + rng.uniform(-radius, radius)

                if k == 0:
                    _pred_x[i] = pred_x
                    _pred_y[i] = pred_y
                # Multi-spot case (k>0): rare, skip extra rows for performance

    # Assign accumulated results at once
    sc_meta_coord['pred_loc_x'] = _pred_x
    sc_meta_coord['pred_loc_y'] = _pred_y

    # Remove rows with NaN coordinates
    sc_meta_coord = sc_meta_coord.dropna(subset=['pred_loc_x', 'pred_loc_y'])
    return sc_meta_coord
