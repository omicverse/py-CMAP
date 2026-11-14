"""
GLUE-CMAP bridge: Use GLUE embeddings to map scATAC-seq cells to spatial locations.

Architecture:
  GLUE (cross-modal alignment) → shared embedding → CMAP Level 1/2/3 (spatial mapping)

GLUE replaces Harmony:
  - Harmony: same modality batch correction (RNA vs RNA)
  - GLUE: cross-modal alignment (ATAC vs RNA) via guidance graph

The shared GLUE embedding space allows CMAP to operate without any
gene activity score conversion — ATAC information is fully preserved.
"""

import logging
import time
import numpy as np
import pandas as pd
import anndata as ad
import torch

logger = logging.getLogger(__name__)


def build_guidance_graph(adata_rna, adata_atac, gtf_path=None):
    """
    Build the genomic guidance graph connecting RNA genes and ATAC peaks.

    Args:
        adata_rna: AnnData with RNA gene expression (needs var annotations: chrom, chromStart, chromEnd)
        adata_atac: AnnData with ATAC peaks (needs var: chrom, chromStart, chromEnd)
        gtf_path: Path to GTF file for gene annotations. If None, uses Ensembl default.

    Returns:
        networkx.Graph: Guidance graph with genes and peaks as nodes.
    """
    import scglue
    import networkx as nx

    # Annotate RNA genes if not already done
    if 'chrom' not in adata_rna.var.columns:
        if gtf_path is None:
            raise ValueError("Provide gtf_path for gene annotations")
        scglue.data.get_gene_annotation(adata_rna, gtf=gtf_path, gtf_by='gene_name')

    # Parse ATAC peaks if needed
    if 'chrom' not in adata_atac.var.columns:
        chrom, start, end, valid = [], [], [], []
        for p in adata_atac.var_names:
            try:
                c, se = p.split(':')
                s, e = se.split('-')
                chrom.append(c); start.append(int(s)); end.append(int(e)); valid.append(p)
            except:
                pass
        adata_atac = adata_atac[:, valid].copy()
        adata_atac.var['chrom'] = chrom
        adata_atac.var['chromStart'] = start
        adata_atac.var['chromEnd'] = end

    # Remove genes without annotations
    has = adata_rna.var[['chrom', 'chromStart', 'chromEnd']].notna().all(axis=1)
    if not has.all():
        adata_rna = adata_rna[:, has].copy()

    graph = scglue.genomics.rna_anchored_guidance_graph(adata_rna, adata_atac)
    logger.info(f"Guidance graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    return graph


def train_glue(adata_rna, adata_atac, graph, save_dir, n_comps=100):
    """
    Train GLUE model to get shared embeddings for RNA and ATAC cells.

    Args:
        adata_rna: AnnData with raw counts in .X, PCA in .obsm['X_pca']
        adata_atac: AnnData with binary peaks in .X, LSI in .obsm['X_lsi']
        graph: Guidance graph from build_guidance_graph()
        save_dir: Directory to save model checkpoints
        n_comps: Number of embedding dimensions

    Returns:
        Trained GLUE model
    """
    import scglue

    scglue.models.configure_dataset(adata_rna, "NB", use_highly_variable=True, use_rep="X_pca")
    scglue.models.configure_dataset(adata_atac, "Bernoulli", use_highly_variable=False, use_rep="X_lsi")

    glue = scglue.models.fit_SCGLUE(
        {"rna": adata_rna, "atac": adata_atac},
        graph,
        fit_kws={"directory": save_dir}
    )

    adata_rna.obsm["X_glue"] = glue.encode_data("rna", adata_rna)
    adata_atac.obsm["X_glue"] = glue.encode_data("atac", adata_atac)

    return glue


def run_cmap_atac(adata_atac, adata_st, glue_emb_atac, glue_emb_st,
                  cluster_key='HMRF_cluster', spatial_data_type='honeycomb',
                  svm_threshold=0.5, num_epochs=2000,
                  para_distance=1.0, para_density=1.0,
                  n_near_spot=None, dis_cut=None, radius=0.5,
                  device=None, verbose=True):
    """
    Map scATAC-seq cells to spatial locations using GLUE embeddings.

    This replaces Harmony with GLUE: instead of aligning in gene expression space,
    we work in the GLUE shared embedding space where ATAC and RNA are already aligned.

    Args:
        adata_atac: AnnData for scATAC-seq cells
        adata_st: AnnData for spatial spots, with .obs[cluster_key] and .obsm['spatial']
        glue_emb_atac: GLUE embedding for ATAC cells (n_cells x n_dims) numpy array
        glue_emb_st: GLUE embedding for ST spots (n_spots x n_dims) numpy array
        cluster_key: Column in adata_st.obs for domain labels
        spatial_data_type: 'honeycomb', 'square', or 'singlecell_level'
        svm_threshold: Confidence threshold for Level 1
        num_epochs: Level 2 optimizer epochs
        n_near_spot, dis_cut: Level 3 neighborhood parameters
        radius: Level 3 jitter radius
        device: 'cuda' or 'cpu'
        verbose: Print progress

    Returns:
        adata_atac with CMAP results in .obs and .obsm['spatial_CMAP']
    """
    import sys
    from ..fast import (svm_tune_np, svm_predict_np, minmax_scale_cols_np,
                               optimizer_np, spearman_cor_np, expand_diff_np)
    from sklearn.neighbors import NearestNeighbors

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t_start = time.time()

    n_cells = glue_emb_atac.shape[0]
    n_spots = glue_emb_st.shape[0]
    n_dims = glue_emb_atac.shape[1]

    domain_labels = np.array(adata_st.obs[cluster_key])
    spot_coords = adata_st.obsm['spatial'].astype(float)
    st_names = np.array(adata_st.obs_names)

    if verbose:
        print(f"[Data] {n_cells} ATAC cells, {n_spots} spots, {n_dims}-dim GLUE embedding")

    # ===== Level 1: SVM on GLUE embeddings (replaces Harmony + quantile norm) =====
    t = time.time()
    train_X, test_X = minmax_scale_cols_np(glue_emb_st, glue_emb_atac)

    # Downsample for tuning if needed
    max_svm_train = 1000
    if train_X.shape[0] > max_svm_train:
        np.random.seed(321)
        ds_idx = []
        for d in np.unique(domain_labels):
            d_idx = np.where(domain_labels == d)[0]
            n_d = max(int(len(d_idx) * max_svm_train / len(domain_labels)), 10)
            ds_idx.extend(np.random.choice(d_idx, min(n_d, len(d_idx)), replace=False))
        ds_idx = np.array(sorted(ds_idx))
        cost, gamma = svm_tune_np(train_X[ds_idx], domain_labels[ds_idx], cross=4)
    else:
        cost, gamma = svm_tune_np(train_X, domain_labels, cross=4)

    pred_st, _, classes = svm_predict_np(train_X, domain_labels, train_X, cost, gamma)
    pred_sc, prob_sc, _ = svm_predict_np(train_X, domain_labels, test_X, cost, gamma)
    max_prob = prob_sc.max(axis=1)

    adata_atac.obs['CMAP_domain'] = pred_sc
    adata_atac.obs['CMAP_svm_prob'] = max_prob
    keep = max_prob > svm_threshold

    if verbose:
        print(f"[Level 1] {time.time()-t:.1f}s | C={cost}, gamma={gamma:.6f} | "
              f"kept {keep.sum()}/{n_cells} (>{svm_threshold})")

    # ===== Level 2: SSIM optimizer on GLUE embeddings =====
    t = time.time()
    domains = np.unique(pred_sc[keep])
    spot_arr = np.full(n_cells, '', dtype=object)
    prob_arr = np.full(n_cells, np.nan)
    ent_arr = np.full(n_cells, np.nan)
    nent_arr = np.full(n_cells, np.nan)
    all_cell_idx = []

    for domain in sorted(domains):
        cell_mask = keep & (pred_sc == domain)
        spot_mask = pred_st == domain
        cell_idx = np.where(cell_mask)[0]
        spot_idx = np.where(spot_mask)[0]

        if len(cell_idx) == 0 or len(spot_idx) == 0:
            continue

        # Use GLUE embeddings directly (no Harmony/data_to_transform needed!)
        cell_feat = glue_emb_atac[cell_idx]
        spot_feat = glue_emb_st[spot_idx]

        mapping = optimizer_np(cell_feat, spot_feat, device=device,
                               num_epochs=num_epochs, lr=0.1,
                               para_distance=para_distance, para_density=para_density)

        best_spot = np.argmax(mapping, axis=1)
        best_prob = np.max(mapping, axis=1)
        clipped = np.clip(mapping, 1e-10, 1.0)
        ent = -np.sum(clipped * np.log2(clipped), axis=1)
        max_ent = np.log2(len(spot_idx))

        for i, ci in enumerate(cell_idx):
            spot_arr[ci] = st_names[spot_idx[best_spot[i]]]
            prob_arr[ci] = best_prob[i]
            ent_arr[ci] = ent[i]
            nent_arr[ci] = ent[i] / max_ent
        all_cell_idx.extend(cell_idx.tolist())

        if verbose:
            print(f"  Domain {domain}: {len(cell_idx)} cells -> {len(spot_idx)} spots")

    adata_atac.obs['CMAP_spot'] = spot_arr
    adata_atac.obs['CMAP_spot_prob'] = prob_arr
    adata_atac.obs['CMAP_entropy'] = ent_arr
    adata_atac.obs['CMAP_norm_entropy'] = nent_arr

    if verbose:
        print(f"[Level 2] {time.time()-t:.1f}s | {len(all_cell_idx)} cells mapped")

    # ===== Level 3: Precise location using GLUE embedding correlations =====
    t = time.time()
    spot_x, spot_y = spot_coords[:, 0], spot_coords[:, 1]

    defaults = {'honeycomb': (6, 30), 'square': (4, 30), 'singlecell_level': (4, 10)}
    top_n, cutoff = defaults.get(spatial_data_type, (6, 30))
    if n_near_spot is not None: top_n = n_near_spot
    if dis_cut is not None: cutoff = dis_cut

    k_use = min(top_n + 1, n_spots)
    nn = NearestNeighbors(n_neighbors=k_use, metric='euclidean')
    nn.fit(spot_coords)
    dists, indices = nn.kneighbors(spot_coords)

    # Spearman correlation in GLUE embedding space
    mapped_idx = np.array(all_cell_idx)
    cor_mat = spearman_cor_np(glue_emb_atac[mapped_idx].T, glue_emb_st.T)

    spot_name_to_idx = {n: i for i, n in enumerate(st_names)}
    pred_x = np.full(n_cells, np.nan)
    pred_y = np.full(n_cells, np.nan)
    rng = np.random.default_rng(42)

    # Auto-detect spot spacing and scale parameters for pixel coordinates
    nn_spacing = NearestNeighbors(n_neighbors=2).fit(spot_coords)
    d_spacing, _ = nn_spacing.kneighbors(spot_coords)
    spot_spacing = np.median(d_spacing[:, 1])

    if dis_cut is None and spot_spacing > 50:
        # dis_cut=30 is for integer-grid coords; scale to pixel space
        # Use 1.5x spot spacing so direct neighbors pass the filter
        cutoff = spot_spacing * 1.5
        if verbose:
            print(f"  Auto dis_cut: {cutoff:.1f} (spot spacing={spot_spacing:.1f})")

    if radius < 1 and spot_spacing > 10:
        radius = spot_spacing * radius

    for mi, ci in enumerate(mapped_idx):
        spot_name = spot_arr[ci]
        if spot_name == '' or spot_name not in spot_name_to_idx:
            continue
        si = spot_name_to_idx[spot_name]

        neigh_si = [indices[si, j] for j in range(1, k_use) if dists[si, j] < cutoff]
        if not neigh_si:
            pred_x[ci] = spot_x[si] + rng.uniform(-radius, radius)
            pred_y[ci] = spot_y[si] + rng.uniform(-radius, radius)
            continue

        # Classify neighbors as left-right vs up-down by ANGLE from center
        # (works for any coordinate scale, not just integer grids)
        cx, cy = spot_x[si], spot_y[si]
        lr_idx, ud_idx = [], []
        for ni in neigh_si:
            dx = spot_x[ni] - cx
            dy = spot_y[ni] - cy
            angle = np.abs(np.arctan2(dy, dx))  # 0=right, pi/2=up, pi=left
            if angle < np.pi / 4 or angle > 3 * np.pi / 4:
                lr_idx.append(ni)  # mostly horizontal
            else:
                ud_idx.append(ni)  # mostly vertical
        if not lr_idx and not ud_idx:
            lr_idx = ud_idx = neigh_si

        cor_lr = cor_mat[mi, lr_idx] if lr_idx else np.array([])
        cor_ud = cor_mat[mi, ud_idx] if ud_idx else np.array([])
        cor_top = cor_mat[mi, si]
        all_cors = np.concatenate([cor_lr, cor_ud, [cor_top]])
        coeff = expand_diff_np(all_cors)

        n_lr, n_ud = len(cor_lr), len(cor_ud)
        if n_lr > 0:
            pred_x[ci] = (np.sum(coeff[:n_lr] * spot_x[lr_idx]) + coeff[-1] * spot_x[si]) / \
                         (np.sum(coeff[:n_lr]) + coeff[-1])
        else:
            pred_x[ci] = spot_x[si] + rng.uniform(-radius, radius)
        if n_ud > 0:
            pred_y[ci] = (np.sum(coeff[n_lr:n_lr+n_ud] * spot_y[ud_idx]) + coeff[-1] * spot_y[si]) / \
                         (np.sum(coeff[n_lr:n_lr+n_ud]) + coeff[-1])
        else:
            pred_y[ci] = spot_y[si] + rng.uniform(-radius, radius)

    adata_atac.obsm['spatial_CMAP'] = np.column_stack([pred_x, pred_y])
    adata_atac.obs['CMAP_quality'] = max_prob * (1.0 - nent_arr)

    if verbose:
        mapped = np.isfinite(pred_x).sum()
        print(f"[Level 3] {time.time()-t:.1f}s | {mapped} cells positioned")
        print(f"[Total]   {time.time()-t_start:.1f}s")

    return adata_atac


def run_cmap_atac_3d(adata_atac, adata_st, glue_emb_atac, glue_emb_st,
                     cluster_key='HMRF_cluster', svm_threshold=0.5,
                     num_epochs=2000, para_distance=1.0, para_density=1.0,
                     dis_cut=None, radius=0.5, device=None, verbose=True):
    """
    Map scATAC-seq cells to 3D spatial locations using GLUE embeddings.

    Same as run_cmap_atac but handles 3D coordinates (x, y, z).
    Outputs obsm['spatial_CMAP_3d'] with shape (n_cells, 3).
    """
    import sys
    from ..fast import (svm_tune_np, svm_predict_np, minmax_scale_cols_np,
                               optimizer_np, spearman_cor_np, expand_diff_np)
    from sklearn.neighbors import NearestNeighbors

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t_start = time.time()

    n_cells = glue_emb_atac.shape[0]
    n_spots = glue_emb_st.shape[0]
    n_dims = glue_emb_atac.shape[1]

    domain_labels = np.array(adata_st.obs[cluster_key])
    spot_coords = adata_st.obsm['spatial'].astype(float)  # (n_spots, 3)
    ndim_spatial = spot_coords.shape[1]
    st_names = np.array(adata_st.obs_names)

    if verbose:
        print(f"[Data] {n_cells} ATAC cells, {n_spots} spots, {n_dims}-dim embedding, "
              f"{ndim_spatial}D spatial")

    # ===== Level 1: SVM on GLUE embeddings =====
    t = time.time()
    train_X, test_X = minmax_scale_cols_np(glue_emb_st, glue_emb_atac)

    max_svm_train = 1000
    if train_X.shape[0] > max_svm_train:
        np.random.seed(321)
        ds_idx = []
        for d in np.unique(domain_labels):
            d_idx = np.where(domain_labels == d)[0]
            n_d = max(int(len(d_idx) * max_svm_train / len(domain_labels)), 10)
            ds_idx.extend(np.random.choice(d_idx, min(n_d, len(d_idx)), replace=False))
        ds_idx = np.array(sorted(ds_idx))
        cost, gamma = svm_tune_np(train_X[ds_idx], domain_labels[ds_idx], cross=4)
    else:
        cost, gamma = svm_tune_np(train_X, domain_labels, cross=4)

    pred_st, _, classes = svm_predict_np(train_X, domain_labels, train_X, cost, gamma)
    pred_sc, prob_sc, _ = svm_predict_np(train_X, domain_labels, test_X, cost, gamma)
    max_prob = prob_sc.max(axis=1)

    adata_atac.obs['CMAP_domain'] = pred_sc
    adata_atac.obs['CMAP_svm_prob'] = max_prob
    keep = max_prob > svm_threshold

    if verbose:
        print(f"[Level 1] {time.time()-t:.1f}s | C={cost}, gamma={gamma:.6f} | "
              f"kept {keep.sum()}/{n_cells} (>{svm_threshold})")

    # ===== Level 2: SSIM optimizer =====
    t = time.time()
    domains = np.unique(pred_sc[keep])
    spot_arr = np.full(n_cells, '', dtype=object)
    prob_arr = np.full(n_cells, np.nan)
    ent_arr = np.full(n_cells, np.nan)
    nent_arr = np.full(n_cells, np.nan)
    all_cell_idx = []

    for domain in sorted(domains):
        cell_mask = keep & (pred_sc == domain)
        spot_mask = pred_st == domain
        cell_idx = np.where(cell_mask)[0]
        spot_idx = np.where(spot_mask)[0]

        if len(cell_idx) == 0 or len(spot_idx) == 0:
            continue

        cell_feat = glue_emb_atac[cell_idx]
        spot_feat = glue_emb_st[spot_idx]

        mapping = optimizer_np(cell_feat, spot_feat, device=device,
                               num_epochs=num_epochs, lr=0.1,
                               para_distance=para_distance, para_density=para_density)

        best_spot = np.argmax(mapping, axis=1)
        best_prob = np.max(mapping, axis=1)
        clipped = np.clip(mapping, 1e-10, 1.0)
        ent = -np.sum(clipped * np.log2(clipped), axis=1)
        max_ent = np.log2(len(spot_idx))

        for i, ci in enumerate(cell_idx):
            spot_arr[ci] = st_names[spot_idx[best_spot[i]]]
            prob_arr[ci] = best_prob[i]
            ent_arr[ci] = ent[i]
            nent_arr[ci] = ent[i] / max_ent
        all_cell_idx.extend(cell_idx.tolist())

        if verbose:
            print(f"  Domain {domain}: {len(cell_idx)} cells -> {len(spot_idx)} spots")

    adata_atac.obs['CMAP_spot'] = spot_arr
    adata_atac.obs['CMAP_spot_prob'] = prob_arr
    adata_atac.obs['CMAP_entropy'] = ent_arr
    adata_atac.obs['CMAP_norm_entropy'] = nent_arr

    if verbose:
        print(f"[Level 2] {time.time()-t:.1f}s | {len(all_cell_idx)} cells mapped")

    # ===== Level 3: 3D precise location =====
    t = time.time()

    # Auto-detect spot spacing in 3D
    nn_spacing = NearestNeighbors(n_neighbors=2).fit(spot_coords)
    d_spacing, _ = nn_spacing.kneighbors(spot_coords)
    spot_spacing = np.median(d_spacing[:, 1])

    n_near = 10  # more neighbors for 3D
    cutoff = dis_cut if dis_cut is not None else spot_spacing * 1.5
    if radius < 1 and spot_spacing > 10:
        radius = spot_spacing * radius

    k_use = min(n_near + 1, n_spots)
    nn = NearestNeighbors(n_neighbors=k_use, metric='euclidean')
    nn.fit(spot_coords)
    dists, indices = nn.kneighbors(spot_coords)

    mapped_idx = np.array(all_cell_idx)
    cor_mat = spearman_cor_np(glue_emb_atac[mapped_idx].T, glue_emb_st.T)

    spot_name_to_idx = {n: i for i, n in enumerate(st_names)}
    pred_coords = np.full((n_cells, ndim_spatial), np.nan)
    rng = np.random.default_rng(42)

    if verbose:
        print(f"  Spot spacing: {spot_spacing:.1f}, dis_cut: {cutoff:.1f}, "
              f"n_near: {n_near}, radius: {radius:.1f}")

    for mi, ci in enumerate(mapped_idx):
        spot_name = spot_arr[ci]
        if spot_name == '' or spot_name not in spot_name_to_idx:
            continue
        si = spot_name_to_idx[spot_name]

        neigh_si = [indices[si, j] for j in range(1, k_use) if dists[si, j] < cutoff]
        if not neigh_si:
            for d in range(ndim_spatial):
                pred_coords[ci, d] = spot_coords[si, d] + rng.uniform(-radius, radius)
            continue

        # In 3D: use correlation-weighted interpolation for ALL dimensions
        cors = np.array([cor_mat[mi, ni] for ni in neigh_si])
        cors = np.maximum(cors, 0)  # only positive correlations contribute
        cor_center = max(cor_mat[mi, si], 0)

        # expand_diff for weighting
        all_cors = np.concatenate([cors, [cor_center]])
        weights = expand_diff_np(all_cors)

        w_neigh = weights[:-1]
        w_center = weights[-1]
        w_total = w_neigh.sum() + w_center

        if w_total < 1e-10:
            for d in range(ndim_spatial):
                pred_coords[ci, d] = spot_coords[si, d] + rng.uniform(-radius, radius)
            continue

        for d in range(ndim_spatial):
            neigh_vals = np.array([spot_coords[ni, d] for ni in neigh_si])
            pred_coords[ci, d] = (np.sum(w_neigh * neigh_vals) + w_center * spot_coords[si, d]) / w_total

    adata_atac.obsm['spatial_CMAP_3d'] = pred_coords
    # Also store 2D projection
    adata_atac.obsm['spatial_CMAP'] = pred_coords[:, :2]
    adata_atac.obs['CMAP_quality'] = max_prob * (1.0 - nent_arr)

    if verbose:
        mapped = np.isfinite(pred_coords[:, 0]).sum()
        print(f"[Level 3] {time.time()-t:.1f}s | {mapped} cells positioned in {ndim_spatial}D")
        print(f"[Total]   {time.time()-t_start:.1f}s")

    return adata_atac
