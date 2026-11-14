"""
Evaluation metrics for spatial mapping (unpaired cross-modal).

No ground truth coordinates available. Use proxy metrics:

1. Expression Reconstruction Correlation (ERC):
   Map ATAC gene activity to space → compare with actual ST expression.

2. Moran's I Preservation (MIP):
   Spatially variable genes in ST should remain spatially variable
   in the ATAC-reconstructed spatial map.

3. Cross-Validation Error (CVE):
   Hold out ST spots, predict their coordinates from embeddings,
   measure error against known positions.
"""

import numpy as np
import logging
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr, pearsonr
from scipy.sparse import issparse

logger = logging.getLogger(__name__)


def expression_reconstruction_correlation(
    mapped_coords, gene_activity, st_coords, st_expr,
    shared_genes_atac, shared_genes_st,
    k_neighbors=10, n_genes_eval=200):
    """
    Evaluate mapping by reconstructing spatial expression from mapped ATAC cells.

    For each ST spot:
      1. Find k nearest mapped ATAC cells
      2. Average their gene activity → predicted expression
      3. Correlate with actual ST expression

    Args:
        mapped_coords: (n_cells, D) mapped ATAC cell coordinates
        gene_activity: (n_cells, n_genes_ga) gene activity matrix
        st_coords: (n_spots, D) ST spot coordinates
        st_expr: (n_spots, n_genes_st) ST expression matrix
        shared_genes_atac: indices of shared genes in gene_activity
        shared_genes_st: indices of shared genes in st_expr
        k_neighbors: K nearest ATAC cells per spot
        n_genes_eval: number of top variable genes to evaluate

    Returns:
        dict with:
          - 'per_spot_corr': (n_spots,) Spearman correlation per spot
          - 'per_gene_corr': (n_genes,) Pearson correlation per gene
          - 'mean_spot_corr': scalar
          - 'mean_gene_corr': scalar
          - 'gene_names': genes evaluated
    """
    # Filter to valid mapped cells
    valid = np.isfinite(mapped_coords[:, 0])
    coords_valid = mapped_coords[valid]
    ga_valid = gene_activity[valid]
    if issparse(ga_valid):
        ga_valid = ga_valid.toarray()

    # Extract shared genes
    ga_shared = ga_valid[:, shared_genes_atac]  # (n_valid, n_shared)
    st_shared = st_expr[:, shared_genes_st]     # (n_spots, n_shared)
    if issparse(st_shared):
        st_shared = st_shared.toarray()

    n_shared = ga_shared.shape[1]
    n_spots = len(st_coords)

    # Select top variable genes for evaluation
    gene_var = np.var(st_shared, axis=0)
    top_idx = np.argsort(gene_var)[::-1][:min(n_genes_eval, n_shared)]
    ga_eval = ga_shared[:, top_idx]
    st_eval = st_shared[:, top_idx]
    n_eval = len(top_idx)

    # Find K nearest mapped cells for each spot
    nn = NearestNeighbors(n_neighbors=min(k_neighbors, len(coords_valid)))
    nn.fit(coords_valid)
    dists, indices = nn.kneighbors(st_coords)

    # Reconstruct: average gene activity of nearest cells
    reconstructed = np.zeros((n_spots, n_eval))
    for i in range(n_spots):
        # Distance-weighted average
        w = np.exp(-dists[i] / (np.median(dists[i]) + 1e-6))
        w = w / (w.sum() + 1e-10)
        reconstructed[i] = w @ ga_eval[indices[i]]

    # Per-spot correlation (across genes)
    spot_corrs = np.zeros(n_spots)
    for i in range(n_spots):
        if np.std(reconstructed[i]) < 1e-10 or np.std(st_eval[i]) < 1e-10:
            spot_corrs[i] = 0
        else:
            spot_corrs[i] = spearmanr(reconstructed[i], st_eval[i])[0]

    # Per-gene correlation (across spots)
    gene_corrs = np.zeros(n_eval)
    for j in range(n_eval):
        if np.std(reconstructed[:, j]) < 1e-10 or np.std(st_eval[:, j]) < 1e-10:
            gene_corrs[j] = 0
        else:
            gene_corrs[j] = pearsonr(reconstructed[:, j], st_eval[:, j])[0]

    return {
        'per_spot_corr': spot_corrs,
        'per_gene_corr': gene_corrs,
        'mean_spot_corr': np.nanmean(spot_corrs),
        'mean_gene_corr': np.nanmean(gene_corrs),
        'median_spot_corr': np.nanmedian(spot_corrs),
        'median_gene_corr': np.nanmedian(gene_corrs),
        'n_valid': int(valid.sum()),
        'n_genes_eval': n_eval,
        'top_gene_idx': top_idx,
    }


def morans_i(values, coords, k=6):
    """
    Compute Moran's I spatial autocorrelation for a single variable.

    Args:
        values: (N,) variable values
        coords: (N, D) spatial coordinates
        k: number of spatial neighbors

    Returns:
        float: Moran's I statistic
    """
    N = len(values)
    if N < k + 1:
        return 0.0

    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    dists, indices = nn.kneighbors(coords)

    z = values - values.mean()
    z_var = np.var(values)
    if z_var < 1e-10:
        return 0.0

    # Build sparse weight matrix (binary: 1 if neighbor, 0 otherwise)
    W_sum = 0
    numerator = 0
    for i in range(N):
        for j_idx in range(1, k + 1):
            j = indices[i, j_idx]
            numerator += z[i] * z[j]
            W_sum += 1

    return (N / W_sum) * (numerator / (z_var * N))


def morans_i_preservation(mapped_coords, gene_activity, st_coords, st_expr,
                          shared_genes_atac, shared_genes_st,
                          n_genes=50, k_spatial=6):
    """
    Compare Moran's I between real ST and ATAC-reconstructed spatial patterns.

    For top spatially variable genes (high Moran's I in ST):
      - Compute Moran's I in real ST
      - Reconstruct spatial expression from mapped ATAC
      - Compute Moran's I in reconstructed data
      - Correlation between the two = preservation score

    Args:
        mapped_coords, gene_activity, st_coords, st_expr: as above
        shared_genes_atac, shared_genes_st: shared gene indices
        n_genes: number of top genes to evaluate
        k_spatial: neighbors for Moran's I

    Returns:
        dict with preservation score and per-gene Moran's I values
    """
    valid = np.isfinite(mapped_coords[:, 0])
    coords_valid = mapped_coords[valid]
    ga_valid = gene_activity[valid]
    if issparse(ga_valid): ga_valid = ga_valid.toarray()

    ga_shared = ga_valid[:, shared_genes_atac]
    st_shared = st_expr[:, shared_genes_st]
    if issparse(st_shared): st_shared = st_shared.toarray()

    n_shared = ga_shared.shape[1]

    # Compute Moran's I for ST genes (subsample spots for speed)
    n_spots = len(st_coords)
    logger.info(f"Computing Moran's I for {n_shared} shared genes...")

    st_morans = np.zeros(n_shared)
    for j in range(n_shared):
        st_morans[j] = morans_i(st_shared[:, j], st_coords, k=k_spatial)

    # Top spatially variable genes
    top_idx = np.argsort(st_morans)[::-1][:min(n_genes, n_shared)]

    # Reconstruct spatial ATAC expression
    nn = NearestNeighbors(n_neighbors=min(10, len(coords_valid))).fit(coords_valid)
    _, indices = nn.kneighbors(st_coords)
    reconstructed = np.array([ga_shared[indices[i]].mean(axis=0) for i in range(n_spots)])

    # Compute Moran's I for reconstructed
    recon_morans = np.zeros(len(top_idx))
    for k_i, j in enumerate(top_idx):
        recon_morans[k_i] = morans_i(reconstructed[:, j], st_coords, k=k_spatial)

    # Preservation = correlation between ST and reconstructed Moran's I
    corr = spearmanr(st_morans[top_idx], recon_morans)[0]

    return {
        'preservation_corr': corr,
        'st_morans': st_morans[top_idx],
        'recon_morans': recon_morans,
        'top_gene_idx': top_idx,
    }


def cross_validation_error(z_st, coords_st, mapper_class, mapper_kwargs,
                           n_folds=5, seed=42):
    """
    K-fold cross-validation on ST spots.

    Hold out spots, train mapper on remaining, predict held-out coordinates.
    Measures how well embedding→coords mapping generalizes.

    Args:
        z_st: (n_spots, z_dim) ST embeddings
        coords_st: (n_spots, D) ST coordinates
        mapper_class: callable that returns a mapper with .fit() and .get_coords()
        mapper_kwargs: kwargs for mapper
        n_folds: number of CV folds
        seed: random seed

    Returns:
        dict with per-fold and mean errors
    """
    from sklearn.model_selection import KFold

    n_spots = len(z_st)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    errors = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(z_st)):
        z_train = z_st[train_idx]
        c_train = coords_st[train_idx]
        z_test = z_st[test_idx]
        c_test = coords_st[test_idx]

        mapper = mapper_class(**mapper_kwargs)
        # Use train spots as both "ST" and "ATAC" (self-mapping)
        mapper.fit(z_train, c_train, z_test, verbose=False, **mapper_kwargs)
        pred = mapper.get_coords('smoothed')

        # Euclidean error
        err = np.sqrt(((pred - c_test) ** 2).sum(axis=1))
        errors.append(err.mean())
        logger.info(f"  Fold {fold}: mean error = {err.mean():.1f}")

    return {
        'fold_errors': errors,
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
    }
