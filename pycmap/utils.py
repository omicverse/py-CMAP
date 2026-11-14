"""
Shared utility functions for the CMAP pipeline.

Includes data normalization, scaling, and transformation functions
that replicate the R equivalents exactly.
"""

import numpy as np
import pandas as pd


def save_expression_h5ad(data, path):
    """Save expression data as h5ad. Accepts AnnData or genes x cells DataFrame."""
    import anndata as ad
    if isinstance(data, ad.AnnData):
        data.write_h5ad(path)
    else:
        # DataFrame: genes x cells -> cells x genes AnnData
        adata = ad.AnnData(X=data.values.T, obs=pd.DataFrame(index=data.columns),
                           var=pd.DataFrame(index=data.index))
        adata.write_h5ad(path)


def load_expression_h5ad(path):
    """Load h5ad file. Returns AnnData directly (keeps sparse)."""
    import anndata as ad
    return ad.read_h5ad(path)


def log_normalize(counts, scale_factor=1e4):
    """
    Log-normalize a count matrix (genes x cells or cells x genes).

    Equivalent to R: log1p(sweep(counts, 2, colSums(counts), '/') * 1e4)
    Assumes genes are rows, cells/spots are columns.

    Args:
        counts: numpy array or pandas DataFrame (genes x cells).
        scale_factor: Scaling factor (default 1e4).

    Returns:
        Log-normalized matrix, same type as input.
    """
    if isinstance(counts, pd.DataFrame):
        col_sums = counts.sum(axis=0)
        normed = counts.div(col_sums, axis=1) * scale_factor
        return np.log1p(normed)
    else:
        col_sums = counts.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1  # avoid division by zero
        normed = counts / col_sums * scale_factor
        return np.log1p(normed)


def quantile_normalize(matrix):
    """
    Quantile normalization of a matrix (genes x samples).

    Exactly replicates R: preprocessCore::normalize.quantiles()
    including proper tie handling via rank averaging.

    Args:
        matrix: numpy array (genes x samples).

    Returns:
        Quantile-normalized numpy array.
    """
    from scipy.stats import rankdata

    n_rows, n_cols = matrix.shape

    # Sort each column to compute rank means (reference distribution)
    sorted_data = np.sort(matrix, axis=0)
    mean_values = sorted_data.mean(axis=1)

    # Vectorized: compute averaged ranks for all columns at once
    # rankdata with method='average' matches R's rank(ties.method="average")
    ranks = np.apply_along_axis(lambda col: rankdata(col, method='average'), 0, matrix)

    # Map ranks to reference distribution via interpolation (vectorized)
    # ranks are 1-based in [1, n_rows]
    lo = np.floor(ranks).astype(int) - 1  # 0-based index
    hi = np.ceil(ranks).astype(int) - 1
    hi = np.clip(hi, 0, n_rows - 1)
    frac = ranks - np.floor(ranks)

    normalized = mean_values[lo] * (1.0 - frac) + mean_values[hi] * frac
    return normalized


def scale_data(matrix, axis=0, max_value=10.0):
    """
    Z-score scale a matrix per gene (row).

    Equivalent to R: singlecellmethods::scaleData()
    Centers to mean=0, scales to sd=1, clips at max_value.

    Args:
        matrix: numpy array (genes x samples).
        axis: Axis along which to compute mean/std (0=per-column, 1=per-row).
              Default 0 means each column is scaled; use 1 for per-gene scaling.
        max_value: Clip absolute values at this threshold.

    Returns:
        Scaled numpy array.
    """
    mean = np.mean(matrix, axis=axis, keepdims=True)
    std = np.std(matrix, axis=axis, ddof=1, keepdims=True)
    std[std == 0] = 1  # avoid division by zero
    scaled = (matrix - mean) / std
    if max_value is not None:
        scaled = np.clip(scaled, -max_value, max_value)
    return scaled


def scale_data_by_column(train_set, test_set):
    """
    Min-max scale test data to [0,1] using training data statistics.

    Equivalent to R: CMAP::scale_data_by_column()
    Both train and test are scaled using the min/max from train.

    Args:
        train_set: pandas DataFrame (samples x features). Last column 'label' is preserved.
        test_set: pandas DataFrame (samples x features).

    Returns:
        Tuple of (scaled_train, scaled_test) DataFrames.
    """
    # Separate label column if present
    has_label = 'label' in train_set.columns
    if has_label:
        train_labels = train_set['label'].copy()
        train_features = train_set.drop(columns=['label'])
    else:
        train_features = train_set

    feature_cols = train_features.columns

    train_min = train_features.min(axis=0)
    train_max = train_features.max(axis=0)
    denom = train_max - train_min
    denom[denom == 0] = 1  # avoid division by zero

    scaled_train = (train_features - train_min) / denom
    scaled_test = (test_set[feature_cols] - train_min) / denom

    if has_label:
        scaled_train = scaled_train.copy()
        scaled_train['label'] = train_labels

    return scaled_train, scaled_test


def expand_diff(x):
    """
    Non-linear coefficient expansion using min-max normalization + tanh.

    Equivalent to R: CMAP::expand_diff()

    Args:
        x: numpy array of values.

    Returns:
        Transformed numpy array.
    """
    x = np.asarray(x, dtype=np.float64)
    x_min = x.min()
    x_max = x.max()
    if x_max == x_min:
        # All values equal: return uniform weights
        return np.ones_like(x)

    # Min-max normalize to [0, 1]
    scale_x = (x - x_min) / (x_max - x_min)
    # Tanh transformation
    tanh_x = 2.0 / (1.0 + np.exp(-2.0 * scale_x)) - 1.0
    # Add variance for emphasis (R uses var() which is n-1 denominator)
    fx = tanh_x + np.var(x, ddof=1)
    return fx


def cosine_normalize(X, margin=1, do_safe=True):
    """
    L2 (cosine) normalization of a matrix.

    Equivalent to R: CMAP::cosine_normalize()

    Args:
        X: numpy array.
        margin: 1 for row-wise, 2 for column-wise normalization.
        do_safe: If True, first divide by max to avoid Inf.

    Returns:
        Normalized numpy array.
    """
    axis = margin - 1  # R margin 1 = numpy axis 0
    if do_safe:
        max_vals = np.max(np.abs(X), axis=axis, keepdims=True)
        max_vals[max_vals == 0] = 1
        X = X / max_vals
    norms = np.sqrt(np.sum(X ** 2, axis=axis, keepdims=True))
    norms[norms == 0] = 1
    return X / norms


def filter_genes(counts, min_cells=50, min_genes_per_cell=None):
    """
    Filter genes and cells from a count matrix.

    Args:
        counts: numpy array or DataFrame (genes x cells).
        min_cells: Keep genes detected in at least this many cells.
        min_genes_per_cell: Keep cells with at least this many genes detected.

    Returns:
        Filtered matrix (same type as input).
    """
    if isinstance(counts, pd.DataFrame):
        # Filter genes
        gene_mask = (counts > 0).sum(axis=1) >= min_cells
        filtered = counts.loc[gene_mask]

        if min_genes_per_cell is not None:
            cell_mask = (filtered > 0).sum(axis=0) >= min_genes_per_cell
            filtered = filtered.loc[:, cell_mask]

        return filtered
    else:
        gene_mask = (counts > 0).sum(axis=1) >= min_cells
        filtered = counts[gene_mask]

        if min_genes_per_cell is not None:
            cell_mask = (filtered > 0).sum(axis=0) >= min_genes_per_cell
            filtered = filtered[:, cell_mask]

        return filtered
