"""
Cell-type co-localization analysis with permutation testing.

Replicates R functions from Celltype_colocaliztion.R:
- cell_type_colocalization(): Create empty count DataFrame
- celltype_colocalization_count(): Count co-localized cell type pairs
- permute_cell_coordinates(): Randomized null model via coordinate permutation
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import norm
from joblib import Parallel, delayed


def cell_type_colocalization(df, cell_type='celltype_0916'):
    """
    Create a DataFrame for storing cell-type pair co-localization counts.

    Equivalent to R: CMAP::cell_type_colocalization()

    Args:
        df: Cell metadata DataFrame containing cell type annotations.
        cell_type: Column name with cell type labels.

    Returns:
        DataFrame indexed by sorted cell-type pairs (e.g., 'TypeA_TypeB')
        with a 'number' column initialized to 0.
    """
    types = sorted(df[cell_type].unique())
    ct_vec = []
    for i, a in enumerate(types):
        for b in types[i:]:
            ct_vec.append(f"{a}_{b}")
    return pd.DataFrame({'number': 0}, index=ct_vec)


def celltype_colocalization_count(df, cut_distance=2.5, x='pred_loc_x', y='pred_loc_y',
                                  cell_type='celltype_0916'):
    """
    Count co-localized cell-type pairs within a distance threshold.

    Equivalent to R: CMAP::celltype_colocalization_count()

    Args:
        df: Cell metadata DataFrame with coordinates and cell type.
        cut_distance: Maximum Euclidean distance for co-localization.
        x: Column name for x-coordinate.
        y: Column name for y-coordinate.
        cell_type: Column name for cell type annotations.

    Returns:
        DataFrame with co-localization counts per cell-type pair.
    """
    coords = df[[x, y]].values
    dist_mat = cdist(coords, coords, metric='euclidean')
    dist_bool = dist_mat < cut_distance

    # Get pairs within distance
    rows, cols = np.where(dist_bool)
    ct1 = df[cell_type].values[rows]
    ct2 = df[cell_type].values[cols]

    # Sort each pair alphabetically
    pairs = []
    for a, b in zip(ct1, ct2):
        if a <= b:
            pairs.append(f"{a}_{b}")
        else:
            pairs.append(f"{b}_{a}")

    # Count occurrences
    pair_counts = pd.Series(pairs).value_counts()

    # Initialize standard DataFrame
    col_ct_df = cell_type_colocalization(df, cell_type)
    for pair_name, count in pair_counts.items():
        if pair_name in col_ct_df.index:
            col_ct_df.loc[pair_name, 'number'] = count

    return col_ct_df


def permute_cell_coordinates(df, radius_permute=25, x='pred_loc_x', y='pred_loc_y',
                             cut_distance=2.5, cell_type='celltype_0916'):
    """
    Randomly shift cell positions and recount co-localizations.

    Equivalent to R: CMAP::permute_cell_coordinates()

    Each cell is shifted to a random position within radius_permute
    from its original location using polar coordinate sampling.

    Args:
        df: Cell metadata DataFrame.
        radius_permute: Maximum shift distance.
        x: Column name for x-coordinate.
        y: Column name for y-coordinate.
        cut_distance: Distance threshold for co-localization.
        cell_type: Column name for cell type.

    Returns:
        DataFrame with co-localization counts for permuted positions.
    """
    df_rand = df.copy()
    n = len(df_rand)

    # Random polar shift
    r = radius_permute * np.sqrt(np.random.uniform(size=n))
    theta = np.random.uniform(size=n) * 2 * np.pi

    df_rand[x] = df_rand[x].values + r * np.cos(theta)
    df_rand[y] = df_rand[y].values + r * np.sin(theta)

    return celltype_colocalization_count(
        df_rand, cut_distance=cut_distance, x=x, y=y, cell_type=cell_type
    )


def run_colocalization_analysis(df, cell_type='celltype_0916', n_permutations=1000,
                                cut_distance=2.5, radius_permute=25,
                                x='pred_loc_x', y='pred_loc_y',
                                n_jobs=-1, min_count=50, alpha=0.05):
    """
    Full co-localization analysis with permutation testing.

    Replicates the analysis workflow from README.md Section 7.

    Args:
        df: Cell metadata DataFrame with coordinates and cell types.
        cell_type: Column name for cell type annotations.
        n_permutations: Number of permutation iterations.
        cut_distance: Distance threshold for co-localization.
        radius_permute: Permutation shift radius.
        x, y: Coordinate column names.
        n_jobs: Number of parallel jobs.
        min_count: Minimum observed count threshold.
        alpha: FDR significance threshold.

    Returns:
        Dict with:
        - 'observed': Observed co-localization counts.
        - 'results': DataFrame with fold_changes, p-values, adjusted p-values.
        - 'significant': Filtered significant results.
    """
    # Observed counts
    col_ct_df = celltype_colocalization_count(
        df, cut_distance=cut_distance, x=x, y=y, cell_type=cell_type
    )

    # Permutation null model
    perm_results = Parallel(n_jobs=n_jobs)(
        delayed(permute_cell_coordinates)(
            df, radius_permute=radius_permute, x=x, y=y,
            cut_distance=cut_distance, cell_type=cell_type
        )
        for _ in range(n_permutations)
    )

    # Combine permutation results
    perm_matrix = pd.concat([r['number'] for r in perm_results], axis=1).values

    # Remove self-pairs (TypeA_TypeA)
    types = sorted(df[cell_type].unique())
    self_pairs = [f"{t}_{t}" for t in types]
    non_self_mask = ~col_ct_df.index.isin(self_pairs)

    col_ct_sub = col_ct_df[non_self_mask]
    mask_arr = non_self_mask.values if hasattr(non_self_mask, 'values') else non_self_mask
    perm_sub = perm_matrix[mask_arr]

    # Z-scores and p-values
    null_means = perm_sub.mean(axis=1)
    null_sd = perm_sub.std(axis=1)
    null_sd = np.maximum(null_sd, np.sqrt(1.0 / n_permutations))

    z_scores = (col_ct_sub['number'].values - null_means) / null_sd
    p_values = norm.sf(z_scores)  # upper-tail p-value

    # FDR correction
    from statsmodels.stats.multitest import multipletests
    _, p_adj, _, _ = multipletests(p_values, method='fdr_bh')

    # Fold changes
    fold_changes = col_ct_sub['number'].values / (null_means + 1e-4)

    # Build results DataFrame
    results = pd.DataFrame({
        'number': col_ct_sub['number'].values,
        'fold_changes': fold_changes,
        'pval': p_values,
        'p.adj': p_adj,
    }, index=col_ct_sub.index)

    # Filter significant
    significant = results[(results['p.adj'] < alpha) & (results['number'] > min_count)]

    return {
        'observed': col_ct_df,
        'results': results,
        'significant': significant,
    }
