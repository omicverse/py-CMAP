"""
Spatial gene selection and HMRF domain clustering.

Replaces R's Giotto-based pipeline (HMRF_gene.R) with Python equivalents.
Uses squidpy for spatial autocorrelation and scipy for clustering.

This is the most complex module — the HMRF pipeline has two modes:
  Phase A: Load pre-computed R results (for validation)
  Phase B: Full Python implementation using squidpy + smfishhmrf
"""

import logging
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


# =============================================================================
# Phase A: Load pre-computed R intermediate results
# =============================================================================

def load_hmrf_results(spatial_genes_path, cluster_labels_path):
    """
    Load pre-computed HMRF results from R.

    Args:
        spatial_genes_path: Path to CSV with spatial genes (one gene per row).
        cluster_labels_path: Path to CSV with spot IDs and HMRF cluster labels.

    Returns:
        Tuple of (spatial_genes_list, cluster_labels_series).
    """
    genes_df = pd.read_csv(spatial_genes_path, header=0)
    if genes_df.shape[1] == 1:
        spatial_genes = genes_df.iloc[:, 0].tolist()
    else:
        spatial_genes = genes_df.iloc[:, 0].tolist()

    labels_df = pd.read_csv(cluster_labels_path, index_col=0)
    if 'HMRF_cluster' in labels_df.columns:
        cluster_labels = labels_df['HMRF_cluster']
    else:
        cluster_labels = labels_df.iloc[:, 0]

    return spatial_genes, cluster_labels


# =============================================================================
# Phase B: Full Python HMRF pipeline
# =============================================================================

def create_spatial_network(spatial_locs, k=6, max_distance=370, min_k=1):
    """
    Create a k-NN spatial network.

    Equivalent to R: Giotto::createSpatialNetwork(method='kNN')

    Args:
        spatial_locs: DataFrame with 'x' and 'y' columns.
        k: Number of nearest neighbors.
        max_distance: Maximum distance for neighbor inclusion.
        min_k: Minimum number of neighbors.

    Returns:
        Dict with:
        - 'indices': Array of neighbor indices (n_spots x k).
        - 'distances': Array of neighbor distances (n_spots x k).
        - 'adjacency': Sparse adjacency matrix.
    """
    coords = spatial_locs[['x', 'y']].values
    n = len(coords)
    k_use = min(k + 1, n)

    nn = NearestNeighbors(n_neighbors=k_use, metric='euclidean')
    nn.fit(coords)
    distances, indices = nn.kneighbors(coords)

    # Remove self (first neighbor) and apply distance cutoff
    dist_no_self = distances[:, 1:]
    idx_no_self = indices[:, 1:]

    # Build adjacency
    from scipy.sparse import lil_matrix
    adj = lil_matrix((n, n))
    for i in range(n):
        for j_pos in range(dist_no_self.shape[1]):
            if dist_no_self[i, j_pos] <= max_distance:
                j = idx_no_self[i, j_pos]
                adj[i, j] = 1
                adj[j, i] = 1

    return {
        'indices': idx_no_self,
        'distances': dist_no_self,
        'adjacency': adj.tocsr(),
    }


def spatial_autocorrelation(expr_matrix, spatial_network, gene_names=None):
    """
    Compute spatial autocorrelation (Moran's I) per gene.

    Equivalent to R: Giotto::binSpect() (simplified).

    Args:
        expr_matrix: Expression matrix (genes x spots), numpy array or DataFrame.
        spatial_network: Dict from create_spatial_network() with 'adjacency'.
        gene_names: List of gene names.

    Returns:
        DataFrame with columns: 'gene', 'morans_i', 'pval', 'adj_pval'.
    """
    try:
        import squidpy as sq
        import anndata as ad

        if isinstance(expr_matrix, pd.DataFrame):
            gene_names = expr_matrix.index.tolist()
            X = expr_matrix.values.T  # spots x genes
        else:
            X = expr_matrix.T
            if gene_names is None:
                gene_names = [f"gene_{i}" for i in range(X.shape[1])]

        adata = ad.AnnData(X=X.astype(np.float32))
        adata.var_names = gene_names

        # Use the adjacency matrix
        from scipy.sparse import csr_matrix
        adata.obsp['spatial_connectivities'] = csr_matrix(spatial_network['adjacency'])
        adata.obsp['spatial_distances'] = csr_matrix(spatial_network['adjacency'])

        sq.gr.spatial_autocorr(adata, mode='moran', connectivity_key='spatial')

        result = adata.uns['moranI'].copy()
        result = result.reset_index()
        result.columns = ['gene'] + list(result.columns[1:])
        return result

    except ImportError:
        logger.warning("squidpy not available, using manual Moran's I computation")
        return _manual_morans_i(expr_matrix, spatial_network, gene_names)


def _manual_morans_i(expr_matrix, spatial_network, gene_names=None):
    """Compute Moran's I manually without squidpy."""
    from scipy.stats import norm
    from scipy.sparse import issparse

    W = spatial_network['adjacency']
    if issparse(W):
        W = W.toarray()  # needed for z @ W @ z; sparse would also work but slower per-gene
    W_sum = W.sum()

    if isinstance(expr_matrix, pd.DataFrame):
        gene_names = expr_matrix.index.tolist()
        X = expr_matrix.values
    else:
        X = expr_matrix
        if gene_names is None:
            gene_names = [f"gene_{i}" for i in range(X.shape[0])]

    n = X.shape[1]
    results = []

    for g in range(X.shape[0]):
        x = X[g]
        x_bar = x.mean()
        z = x - x_bar
        numerator = n * float(z @ W @ z)
        denominator = W_sum * float(np.sum(z ** 2))
        I = numerator / denominator if denominator != 0 else 0

        E_I = -1 / (n - 1)
        S1 = float(((W + W.T) ** 2).sum()) / 2
        S2 = float(((W.sum(axis=1) + W.sum(axis=0).T) ** 2).sum())
        S0 = W_sum
        n2 = n * n
        k2 = (n * np.sum(z ** 4)) / (np.sum(z ** 2) ** 2) if np.sum(z ** 2) > 0 else 0

        var_I = (n2 * S1 - n * S2 + 3 * S0 ** 2) / (S0 ** 2 * (n2 - 1)) - E_I ** 2
        var_I = max(var_I, 1e-10)
        z_score = (I - E_I) / np.sqrt(var_I)
        pval = 2 * norm.sf(abs(z_score))

        results.append({'gene': gene_names[g], 'I': I, 'pval_norm': pval})

    df = pd.DataFrame(results)
    from statsmodels.stats.multitest import multipletests
    _, adj_pvals, _, _ = multipletests(df['pval_norm'].values, method='fdr_bh')
    df['pval_norm_fdr'] = adj_pvals
    return df


def filter_spatial_genes(gene_scores, max_genes=2500, method='elbow'):
    """
    Filter top spatially significant genes.

    Equivalent to R: Giotto::filterSpatialGenes()

    Args:
        gene_scores: DataFrame from spatial_autocorrelation() with p-values.
        max_genes: Maximum number of genes to keep.
        method: 'elbow' or 'top' (top-N by p-value).

    Returns:
        List of filtered gene names.
    """
    # Determine p-value column
    pval_col = None
    for col in ['pval_norm_fdr', 'adj_pval', 'pval_norm', 'morans_pval']:
        if col in gene_scores.columns:
            pval_col = col
            break
    if pval_col is None:
        pval_col = gene_scores.columns[-1]

    # Sort by p-value
    sorted_df = gene_scores.sort_values(pval_col, ascending=True)

    if method == 'top' or len(sorted_df) <= max_genes:
        return sorted_df['gene'].head(max_genes).tolist()

    # Elbow method: find inflection point in -log10(pval)
    pvals = sorted_df[pval_col].values[:max_genes]
    pvals = np.clip(pvals, 1e-300, 1)
    neg_log = -np.log10(pvals)

    # Simple elbow: find maximum curvature
    if len(neg_log) > 10:
        # Normalize to [0, 1]
        x = np.arange(len(neg_log)) / len(neg_log)
        y = (neg_log - neg_log.min()) / (neg_log.max() - neg_log.min() + 1e-10)

        # Line from first to last point
        dx = x[-1] - x[0]
        dy = y[-1] - y[0]
        line_len = np.sqrt(dx ** 2 + dy ** 2)

        # Distance from each point to the line
        distances = np.abs(dy * x - dx * y + x[-1] * y[0] - y[-1] * x[0]) / (line_len + 1e-10)
        elbow_idx = np.argmax(distances)
        n_genes = max(elbow_idx + 1, 100)  # at least 100 genes
    else:
        n_genes = len(neg_log)

    n_genes = min(n_genes, max_genes)
    return sorted_df['gene'].head(n_genes).tolist()


def detect_spatial_cor_genes(expr_matrix, spatial_network, genes, network_smoothing=0):
    """
    Compute spatial gene-gene correlation matrix.

    Equivalent to R: Giotto::detectSpatialCorGenes()

    Args:
        expr_matrix: Expression matrix (genes x spots), DataFrame.
        spatial_network: Dict from create_spatial_network().
        genes: List of genes to analyze.
        network_smoothing: Smoothing parameter (0 = no smoothing).

    Returns:
        Gene-gene correlation matrix, DataFrame.
    """
    sub = expr_matrix.loc[genes].values  # genes x spots
    cor_mat = np.corrcoef(sub)
    return pd.DataFrame(cor_mat, index=genes, columns=genes)


def cluster_spatial_cor_genes(cor_matrix, k=3):
    """
    Cluster genes into modules based on correlation.

    Equivalent to R: Giotto::clusterSpatialCorGenes()

    Args:
        cor_matrix: Gene-gene correlation matrix, DataFrame.
        k: Number of clusters.

    Returns:
        Series mapping gene names to cluster IDs.
    """
    # Convert correlation to distance
    dist_matrix = 1 - cor_matrix.values
    np.fill_diagonal(dist_matrix, 0)
    dist_matrix = np.maximum(dist_matrix, 0)

    # Condensed distance matrix
    from scipy.spatial.distance import squareform
    condensed = squareform(dist_matrix, checks=False)

    # Hierarchical clustering
    Z = linkage(condensed, method='ward')
    clusters = fcluster(Z, t=k, criterion='maxclust')

    return pd.Series(clusters, index=cor_matrix.index, name='cluster')


def sampling_sp_genes(gene_clusters, sample_rate=2, target=500, seed=10):
    """
    Stratified sampling of spatial genes from clusters.

    Equivalent to R: CMAP::sampling_sp_genes()

    Args:
        gene_clusters: Series mapping gene names to cluster IDs.
        sample_rate: Sampling rate parameter (controls proportionality).
        target: Target number of genes.
        seed: Random seed.

    Returns:
        List of sampled gene names.
    """
    clusters = gene_clusters.unique()
    num_cluster = len(clusters)

    gene_list = {}
    for c in clusters:
        gene_list[c] = gene_clusters[gene_clusters == c].index.tolist()

    # Compute sampling factor
    tot = 0
    for c in clusters:
        num_g = len(gene_list[c])
        tot += num_g / (num_g ** (1.0 / sample_rate))

    factor = target / tot

    rng = np.random.default_rng(seed)
    union_genes = []

    for c in clusters:
        num_g = len(gene_list[c])
        num_sample = round(num_g / (num_g ** (1.0 / sample_rate)) * factor)

        if len(gene_list[c]) <= num_sample:
            samples = gene_list[c]
        else:
            samples = rng.choice(gene_list[c], size=num_sample, replace=False).tolist()

        union_genes.extend(samples)

    return list(set(union_genes))


def hmrf_spatial_gene(expr_matrix, spatial_locs, k=3, filter_hbb=True,
                      gene_sampling_from_top=2500, filter_method='elbow',
                      gene_samples=500, spatial_network_name='KNN_network',
                      knn_k=6, max_distance=370):
    """
    Full spatial gene selection pipeline using HMRF.

    Equivalent to R: CMAP::hmrf_spatial_gene()
    Chains: spatial autocorrelation → filter → correlation → cluster → sample.

    Args:
        expr_matrix: Normalized expression matrix (genes x spots), DataFrame.
        spatial_locs: Spot coordinates DataFrame with 'x', 'y' columns.
        k: Number of gene clusters (spatial domains).
        filter_hbb: Filter hemoglobin/ribosomal/mitochondrial genes.
        gene_sampling_from_top: Max genes to consider.
        filter_method: 'elbow' or 'top'.
        gene_samples: Target number of sampled genes.
        spatial_network_name: Ignored (for API compatibility).
        knn_k: Number of kNN neighbors.
        max_distance: Max distance for spatial network.

    Returns:
        List of selected spatial gene names.
    """
    gene_names = list(expr_matrix.index)

    # Filter unwanted genes
    if filter_hbb:
        import re
        pattern = re.compile(r'^mt-|^Hb[ba]-|^Rp[sl]|^Mrp[sl]', re.IGNORECASE)
        gene_names = [g for g in gene_names if not pattern.match(g)]

    # Create spatial network
    logger.info("Creating spatial network...")
    spatial_net = create_spatial_network(spatial_locs, k=knn_k, max_distance=max_distance)

    # Spatial autocorrelation
    logger.info("Computing spatial autocorrelation...")
    sub_expr = expr_matrix.loc[gene_names]
    gene_scores = spatial_autocorrelation(sub_expr, spatial_net)

    # Filter spatial genes
    logger.info("Filtering spatial genes...")
    spatial_genes = filter_spatial_genes(gene_scores, max_genes=gene_sampling_from_top,
                                        method=filter_method)

    if len(spatial_genes) == 0:
        raise ValueError("No spatial genes remaining. Please provide a larger gene list.")

    n = min(gene_samples, 500, len(spatial_genes))

    if n < len(spatial_genes):
        # Compute spatial co-expression modules
        logger.info("Computing spatial co-expression modules...")
        cor_mat = detect_spatial_cor_genes(expr_matrix, spatial_net, spatial_genes)

        logger.info("Clustering spatial genes...")
        gene_clusters = cluster_spatial_cor_genes(cor_mat, k=k)

        # Sample genes proportionally
        logger.info("Sampling spatial genes from modules...")
        spatial_genes_selected = sampling_sp_genes(gene_clusters, sample_rate=2,
                                                   target=n, seed=10)
        logger.info(f"Selected {len(spatial_genes_selected)} spatial genes.")
    else:
        spatial_genes_selected = spatial_genes

    return spatial_genes_selected


def run_hmrf(expr_matrix, spatial_locs, spatial_genes, k=3, beta=45,
             spatial_network=None, knn_k=6, max_distance=370):
    """
    Run HMRF spatial domain segmentation.

    Equivalent to R: Giotto::doHMRF() + addHMRF()

    Uses smfishhmrf Python package if available, otherwise falls back
    to a simple k-means + spatial smoothing approach.

    Args:
        expr_matrix: Scaled expression matrix (genes x spots), DataFrame.
        spatial_locs: Spot coordinates DataFrame with 'x', 'y' columns.
        spatial_genes: List of genes for HMRF.
        k: Number of spatial domains.
        beta: Spatial smoothness parameter.
        spatial_network: Pre-computed network dict.
        knn_k: kNN parameter if network not provided.
        max_distance: Max distance if network not provided.

    Returns:
        Series mapping spot IDs to domain labels (1-indexed).
    """
    if spatial_network is None:
        spatial_network = create_spatial_network(spatial_locs, k=knn_k,
                                                max_distance=max_distance)

    genes_use = [g for g in spatial_genes if g in expr_matrix.index]
    X = expr_matrix.loc[genes_use].values.T  # spots x genes

    try:
        from smfishhmrf import HMRFInstance
        logger.info("Running HMRF via smfishhmrf...")

        # Build neighbor list
        adj = spatial_network['adjacency']
        n_spots = X.shape[0]

        # smfishhmrf expects specific input format
        # Create HMRF instance and run
        hmrf = HMRFInstance(
            X, adj, k,
            betas=np.arange(0, beta + 1, 2) if beta > 0 else [0],
        )
        hmrf.init()
        labels = hmrf.run()

        return pd.Series(labels + 1, index=spatial_locs.index, name='HMRF_cluster')

    except (ImportError, Exception) as e:
        logger.warning(f"smfishhmrf not available or failed ({e}), using k-means fallback")
        return _kmeans_spatial(X, spatial_network, k, beta, spatial_locs.index)


def _kmeans_spatial(X, spatial_network, k, beta, spot_ids, max_iter=50):
    """
    Simple k-means with spatial MRF smoothing as HMRF fallback.

    Args:
        X: Expression matrix (spots x genes).
        spatial_network: Network dict with 'adjacency'.
        k: Number of clusters.
        beta: Spatial smoothness weight.
        spot_ids: Index for output Series.
        max_iter: Maximum EM iterations.

    Returns:
        Series mapping spot IDs to cluster labels.
    """
    from sklearn.cluster import KMeans

    # Initialize with k-means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_

    adj = spatial_network['adjacency'].toarray()
    n_spots = X.shape[0]

    for iteration in range(max_iter):
        old_labels = labels.copy()

        # Update labels with MRF prior
        for i in range(n_spots):
            # Expression likelihood (Gaussian)
            expr_dist = np.sum((X[i] - centers) ** 2, axis=1)

            # Spatial prior: count neighbors in each cluster
            neighbors = np.where(adj[i] > 0)[0]
            spatial_prior = np.zeros(k)
            for c in range(k):
                spatial_prior[c] = np.sum(labels[neighbors] == c)

            # Combined energy
            energy = expr_dist - beta * spatial_prior
            labels[i] = np.argmin(energy)

        # Update centers
        for c in range(k):
            mask = labels == c
            if mask.sum() > 0:
                centers[c] = X[mask].mean(axis=0)

        if np.all(labels == old_labels):
            break

    # Convert to 1-indexed labels with cluster_ prefix
    cluster_labels = [f"cluster_{l + 1}" for l in labels]
    return pd.Series(cluster_labels, index=spot_ids, name='HMRF_cluster')
