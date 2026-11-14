"""
Batch correction module using Harmony.

Replicates the R function harmony_embedding() from Harmony_embedding.R.
Uses harmonypy (direct Python port of the R harmony package).
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD

from .utils import scale_data, cosine_normalize


def harmony_embedding(st_norm, sc_norm, genes, batch=True, npc=100, nclust=100,
                      pca_method='pca', max_iter_harmony=50, max_iter_cluster=6,
                      cosine_norm=False, random_state=123):
    """
    Remove batch effects between ST and scRNA-seq data using Harmony.

    Equivalent to R: CMAP::harmony_embedding()

    Args:
        st_norm: Spatial expression matrix (genes x spots), DataFrame or array.
        sc_norm: Single-cell expression matrix (genes x cells), DataFrame or array.
        genes: List of genes to use.
        batch: If True, scale ST and SC separately before combining.
        npc: Number of principal components.
        nclust: Number of Harmony clusters.
        pca_method: 'pca' (full PCA, like R prcomp) or 'truncated_svd' (like R prcomp_irlba).
        max_iter_harmony: Maximum Harmony iterations.
        max_iter_cluster: Maximum k-means iterations per Harmony round.
        cosine_norm: Whether to apply cosine normalization.
        random_state: Random seed for reproducibility.

    Returns:
        Corrected embedding matrix (PCs x samples), with columns ordered as
        [sc_samples..., st_samples...].
    """
    import harmonypy

    # Convert to DataFrame if needed
    if isinstance(st_norm, np.ndarray):
        st_norm = pd.DataFrame(st_norm)
    if isinstance(sc_norm, np.ndarray):
        sc_norm = pd.DataFrame(sc_norm)

    # Filter zero-sum genes
    st_norm = st_norm.loc[st_norm.sum(axis=1) > 0]
    sc_norm = sc_norm.loc[sc_norm.sum(axis=1) > 0]

    # Intersect genes
    genes = list(set(genes) & set(st_norm.index) & set(sc_norm.index))
    if len(genes) == 0:
        raise ValueError("No overlapping genes found between st_norm, sc_norm, and provided gene list.")

    # Subset to selected genes
    tmp_st = st_norm.loc[genes].values.astype(np.float64)
    tmp_sc = sc_norm.loc[genes].values.astype(np.float64)

    # Z-score scaling per gene (row)
    if batch:
        # Scale ST and SC separately
        tmp_st = scale_data(tmp_st, axis=1)  # per-gene (row) scaling
        tmp_sc = scale_data(tmp_sc, axis=1)
        V = np.hstack([tmp_st, tmp_sc])
    else:
        V = np.hstack([tmp_st, tmp_sc])
        V = scale_data(V, axis=1)

    # 子集内零方差基因 z-score 后为 NaN -> 置零 (否则 PCA 拒收)
    V = np.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)

    if cosine_norm:
        V = cosine_normalize(V, margin=2)

    # Build metadata
    n_st = tmp_st.shape[1]
    n_sc = tmp_sc.shape[1]
    st_ids = list(st_norm.columns) if hasattr(st_norm, 'columns') else [f'st_{i}' for i in range(n_st)]
    sc_ids = list(sc_norm.columns) if hasattr(sc_norm, 'columns') else [f'sc_{i}' for i in range(n_sc)]

    meta_data = pd.DataFrame({
        'id': st_ids + sc_ids,
        'dataset': ['Spatial'] * n_st + ['SingleCell'] * n_sc
    })

    # PCA on transposed data (samples x genes)
    V_T = V.T  # (n_samples x n_genes)
    npc = min(npc, V_T.shape[0], V_T.shape[1])

    np.random.seed(random_state)
    if pca_method == 'truncated_svd':
        # Equivalent to R prcomp_irlba (randomized SVD, no centering)
        pca = PCA(n_components=npc, svd_solver='randomized', random_state=random_state)
        pca_result = pca.fit_transform(V_T)
    else:
        # Equivalent to R prcomp (full SVD, no centering)
        pca = PCA(n_components=npc, svd_solver='full')
        pca_result = pca.fit_transform(V_T)

    # Harmony integration
    nclust = min(nclust, pca_result.shape[0] // 2)
    harmony_out = harmonypy.run_harmony(
        pca_result,
        meta_data,
        'dataset',
        theta=1.0,
        nclust=nclust,
        max_iter_harmony=max_iter_harmony,
        max_iter_kmeans=max_iter_cluster,
        random_state=random_state,
    )

    # Z_corr shape varies by harmonypy version:
    #   0.0.x: (n_PCs x n_samples)
    #   0.2.x: (n_samples x n_PCs)
    corrected = harmony_out.Z_corr
    if corrected.shape[0] == pca_result.shape[0]:
        pass  # already (n_samples x n_PCs)
    else:
        corrected = corrected.T  # transpose to (n_samples x n_PCs)

    # Reorder: SingleCell first, then Spatial (matching R behavior)
    sc_mask = meta_data['dataset'] == 'SingleCell'
    st_mask = meta_data['dataset'] == 'Spatial'

    result = np.vstack([corrected[sc_mask.values], corrected[st_mask.values]])

    # Transpose to (PCs x samples) and set column names
    matrix = result.T

    # Build column names
    col_names = [meta_data.loc[i, 'id'] for i in meta_data.index[sc_mask]] + \
                [meta_data.loc[i, 'id'] for i in meta_data.index[st_mask]]

    result_df = pd.DataFrame(matrix, columns=col_names)
    return result_df
