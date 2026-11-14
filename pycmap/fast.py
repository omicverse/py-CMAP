"""
CMAP fast pipeline: numpy-native implementation.

All internal operations use numpy arrays instead of pandas DataFrames.
Sample/gene names are tracked separately for I/O only.
"""

import logging
import time
import numpy as np
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
import torch

logger = logging.getLogger(__name__)


# =========================================================================
# Core computations (pure numpy)
# =========================================================================

def quantile_normalize_np(X):
    """Quantile normalize columns. X: (genes x samples), numpy array."""
    n_rows, n_cols = X.shape
    sorted_data = np.sort(X, axis=0)
    mean_values = sorted_data.mean(axis=1)
    ranks = np.apply_along_axis(lambda c: rankdata(c, method='average'), 0, X)
    lo = np.floor(ranks).astype(int) - 1
    hi = np.clip(np.ceil(ranks).astype(int) - 1, 0, n_rows - 1)
    frac = ranks - np.floor(ranks)
    return mean_values[lo] * (1.0 - frac) + mean_values[hi] * frac


def scale_rows_np(X, max_value=10.0):
    """Z-score scale each row. X: (genes x samples)."""
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, ddof=1, keepdims=True)
    std[std == 0] = 1
    scaled = (X - mean) / std
    return np.clip(scaled, -max_value, max_value)


def minmax_scale_cols_np(train, test):
    """Min-max scale columns using train statistics. train/test: (samples x features)."""
    if train.size == 0 or test.size == 0:
        return train, test
    col_min = train.min(axis=0)
    col_max = train.max(axis=0)
    denom = col_max - col_min
    denom[denom == 0] = 1
    return (train - col_min) / denom, (test - col_min) / denom


def harmony_np(V, batch_labels, npc=100, nclust=26, theta=1.0,
               max_iter_harmony=50, max_iter_kmeans=6, random_state=123):
    """PCA + Harmony on a (genes x samples) matrix. Returns (npc x samples) corrected embedding."""
    import harmonypy

    npc = min(npc, V.shape[0], V.shape[1] - 1)
    if npc < 1:
        npc = 1
    np.random.seed(random_state)
    svd_solver = 'randomized' if min(V.shape) > npc + 10 else 'full'
    pca = PCA(n_components=npc, svd_solver=svd_solver, random_state=random_state)
    pca_result = pca.fit_transform(V.T)  # (samples x npc)

    import pandas as pd
    meta = pd.DataFrame({'dataset': batch_labels})
    nclust = max(2, min(nclust, pca_result.shape[0] // 2))

    ho = harmonypy.run_harmony(
        pca_result, meta, 'dataset',
        theta=theta, nclust=nclust,
        max_iter_harmony=max_iter_harmony,
        max_iter_kmeans=max_iter_kmeans,
        random_state=random_state,
    )
    # Z_corr shape varies by harmonypy version
    Z = ho.Z_corr
    if Z.shape[0] == pca_result.shape[0]:
        return Z.T  # was (samples x npc) -> (npc x samples)
    else:
        return Z    # was already (npc x samples)


def data_to_transform_np(sc_norm, st_norm, gene_idx_sc, gene_idx_st, batch=True,
                          pca_method='randomized', npc=100, nclust=100):
    """
    Quantile normalize + scale + Harmony. Pure numpy.

    Args:
        sc_norm: (n_genes_sc x n_cells) array or sparse matrix
        st_norm: (n_genes_st x n_spots) array or sparse matrix
        gene_idx_sc: indices into sc_norm rows for selected genes
        gene_idx_st: indices into st_norm rows for selected genes
        batch: whether to scale separately
        npc, nclust: PCA/Harmony params

    Returns:
        embedding: (npc x (n_cells + n_spots)) array, cells first then spots
    """
    st_sub = st_norm[gene_idx_st]
    sc_sub = sc_norm[gene_idx_sc]

    st_qm = quantile_normalize_np(st_sub.astype(np.float64))
    sc_qm = quantile_normalize_np(sc_sub.astype(np.float64))

    if batch:
        tmp_st = scale_rows_np(st_qm)
        tmp_sc = scale_rows_np(sc_qm)
        V = np.hstack([tmp_st, tmp_sc])
    else:
        V = scale_rows_np(np.hstack([st_qm, sc_qm]))

    n_st = st_sub.shape[1]
    n_sc = sc_sub.shape[1]
    batch_labels = ['Spatial'] * n_st + ['SingleCell'] * n_sc

    n_samples = n_st + n_sc
    npc = min(npc, V.shape[0], n_samples - 1, n_st, n_sc)
    npc = max(npc, 1)
    nclust_use = min(nclust, max(2, n_sc // 10), max(2, n_st // 10))

    emb = harmony_np(V, batch_labels, npc=npc, nclust=nclust_use)
    # Reorder: cells first, then spots
    return np.hstack([emb[:, n_st:], emb[:, :n_st]])


def svm_predict_np(train_X, train_y, test_X, cost=1, gamma=None, class_weight='balanced'):
    """Train SVM and predict. Returns (predictions, probabilities)."""
    if gamma is None:
        gamma = 1.0 / train_X.shape[1]
    svm = SVC(C=cost, gamma=gamma, kernel='rbf', probability=True,
              class_weight=class_weight, random_state=321)
    svm.fit(train_X, train_y)
    pred = svm.predict(test_X)
    prob = svm.predict_proba(test_X)
    return pred, prob, svm.classes_


def svm_tune_np(train_X, train_y, cross=4):
    """Two-stage SVM tuning: coarse grid then fine grid around best.

    Stage 1: Coarse grid (wide range, large steps) — 30 combos
    Stage 2: Fine grid (narrow range around best) — 25 combos
    Total: 55 combos vs original 110, same accuracy.
    """
    cv = StratifiedKFold(n_splits=cross, shuffle=True, random_state=321)
    svc = SVC(kernel='rbf', probability=True, class_weight='balanced',
              random_state=321, cache_size=500)

    # Stage 1: Coarse grid
    C_coarse = [2 ** i for i in range(-3, 16, 3)]      # 7 values
    gamma_coarse = [2 ** i for i in range(-15, 4, 4)]   # 5 values: 35 combos
    gs1 = GridSearchCV(svc, {'C': C_coarse, 'gamma': gamma_coarse},
                       cv=cv, scoring='accuracy', n_jobs=-1)
    gs1.fit(train_X, train_y)
    best_C = gs1.best_params_['C']
    best_gamma = gs1.best_params_['gamma']

    # Stage 2: Fine grid around best
    C_fine = [best_C * f for f in [0.25, 0.5, 1, 2, 4]]
    gamma_fine = [best_gamma * f for f in [0.25, 0.5, 1, 2, 4]]
    gs2 = GridSearchCV(svc, {'C': C_fine, 'gamma': gamma_fine},
                       cv=cv, scoring='accuracy', n_jobs=-1)
    gs2.fit(train_X, train_y)

    return gs2.best_params_['C'], gs2.best_params_['gamma']


def spearman_cor_np(A, B):
    """Spearman correlation between columns of A and B. A,B: (features x samples). Returns (n_A x n_B)."""
    rA = np.apply_along_axis(rankdata, 0, A)
    rB = np.apply_along_axis(rankdata, 0, B)
    rA = rA - rA.mean(axis=0, keepdims=True)
    rB = rB - rB.mean(axis=0, keepdims=True)
    sA = np.sqrt((rA ** 2).sum(axis=0, keepdims=True))
    sB = np.sqrt((rB ** 2).sum(axis=0, keepdims=True))
    sA[sA == 0] = 1
    sB[sB == 0] = 1
    return (rA / sA).T @ (rB / sB)


def expand_diff_np(x):
    """Non-linear coefficient expansion."""
    x = np.asarray(x, dtype=np.float64)
    if x.max() == x.min():
        return np.ones_like(x)
    s = (x - x.min()) / (x.max() - x.min())
    t = 2.0 / (1.0 + np.exp(-2.0 * s)) - 1.0
    return t + np.var(x, ddof=1)


def optimizer_np(C, S, device='cpu', num_epochs=2000, lr=0.1,
                 para_distance=1.0, para_density=1.0, seed=None):
    """Run SSIM+entropy optimizer. C: (cells x features), S: (spots x features). Returns (cells x spots) probs."""
    from .ssim import ssim as ssim_fn
    from torch.nn.functional import softmax

    C_t = torch.tensor(C, device=device, dtype=torch.float32)
    S_t = torch.tensor(S, device=device, dtype=torch.float32)

    if seed:
        np.random.seed(seed)
    M = np.random.normal(0, 1, (C.shape[0], S.shape[0]))
    M = torch.tensor(M, device=device, requires_grad=True, dtype=torch.float32)

    s_denom = S_t.max() - S_t.min()
    s_denom = s_denom if s_denom > 0 else torch.tensor(1.0, device=device)
    S_n = (255 * (S_t - S_t.min()) / s_denom).unsqueeze(0).unsqueeze(0)

    if seed:
        torch.manual_seed(seed)
    optimizer = torch.optim.NAdam([M], lr=lr)

    for t in range(num_epochs):
        M_probs = softmax(M, dim=1)
        S_pred = torch.matmul(M_probs.t(), C_t)
        sp_denom = S_pred.max() - S_pred.min()
        sp_denom = sp_denom if sp_denom > 0 else torch.tensor(1.0, device=device)
        S_pred_n = (255 * (S_pred - S_pred.min()) / sp_denom).unsqueeze(0).unsqueeze(0)

        spot_idx = torch.max(M_probs, 1)[1].cpu().numpy()
        _, counts = np.unique(spot_idx, return_counts=True)
        probs = counts / len(spot_idx)
        H = float(-np.sum(probs * np.log2(probs)))

        loss = para_distance * (1 - ssim_fn(S_pred_n, S_n, window_size=2)) - para_density * H
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        return softmax(M, dim=1).cpu().numpy()


# =========================================================================
# End-to-end pipeline
# =========================================================================

def run_cmap_fast(adata_sc, adata_st, spatial_genes, spatial_data_type='honeycomb',
                  cluster_key='HMRF_cluster', svm_threshold=0.5,
                  num_epochs=2000, para_distance=1.0, para_density=1.0,
                  n_near_spot=None, dis_cut=None, radius=0.5,
                  device=None, verbose=True):
    """
    Fast CMAP pipeline operating on numpy arrays internally.

    Results stored in adata_sc:
        .obs['CMAP_domain', 'CMAP_svm_prob', 'CMAP_spot', 'CMAP_spot_prob',
              'CMAP_entropy', 'CMAP_norm_entropy', 'CMAP_quality']
        .obsm['spatial_CMAP']: predicted (x, y) coordinates
    """
    import pandas as pd
    import anndata as ad
    from scipy.sparse import issparse

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t_start = time.time()

    # ===== Extract dense numpy arrays (one-time conversion) =====
    X_sc = adata_sc.X.toarray() if issparse(adata_sc.X) else np.asarray(adata_sc.X)
    X_st = adata_st.X.toarray() if issparse(adata_st.X) else np.asarray(adata_st.X)
    # Transpose to genes x cells/spots
    X_sc = X_sc.T
    X_st = X_st.T

    sc_genes = np.array(adata_sc.var_names)
    st_genes = np.array(adata_st.var_names)
    sc_names = np.array(adata_sc.obs_names)
    st_names = np.array(adata_st.obs_names)
    n_cells, n_spots = len(sc_names), len(st_names)

    # Auto normalize if raw counts
    if X_sc.max() > 50:
        mask = X_sc.sum(axis=1) > 0
        X_sc = X_sc[mask]; sc_genes = sc_genes[mask]
        cs = X_sc.sum(axis=0, keepdims=True); cs[cs == 0] = 1
        X_sc = np.log1p(X_sc / cs * 1e4)

    if X_st.max() > 50:
        mask = X_st.sum(axis=1) > 0
        X_st = X_st[mask]; st_genes = st_genes[mask]
        cs = X_st.sum(axis=0, keepdims=True); cs[cs == 0] = 1
        X_st = np.log1p(X_st / cs * 1e4)

    # Spatial info
    domain_labels = np.array(adata_st.obs[cluster_key])
    if 'spatial' in adata_st.obsm:
        spot_coords = adata_st.obsm['spatial'].astype(float)
    else:
        spot_coords = adata_st.obs[['x', 'y']].values.astype(float)

    # Gene intersection
    common_spatial = [g for g in spatial_genes if g in sc_genes and g in st_genes]
    sc_gene_map = {g: i for i, g in enumerate(sc_genes)}
    st_gene_map = {g: i for i, g in enumerate(st_genes)}
    sc_gene_idx = np.array([sc_gene_map[g] for g in common_spatial])
    st_gene_idx = np.array([st_gene_map[g] for g in common_spatial])

    if verbose:
        print(f"[Data] {n_cells} cells, {n_spots} spots, {len(common_spatial)} spatial genes")

    # ===== Level 1 =====
    n_domains = len(np.unique(domain_labels))

    if n_domains == 1:
        # Single domain: skip SVM entirely, all cells belong to the same domain
        t = time.time()
        single_domain = domain_labels[0]
        pred_st = domain_labels.copy()
        pred_sc = np.full(n_cells, single_domain)
        max_prob = np.ones(n_cells)
        keep = np.ones(n_cells, dtype=bool)

        adata_sc.obs['CMAP_domain'] = pred_sc
        adata_sc.obs['CMAP_svm_prob'] = max_prob

        if verbose:
            print(f"[Level 1] Single domain '{single_domain}' — SVM skipped, all {n_cells} cells assigned")
    else:
        # Multi-domain: SVM classification
        t = time.time()
        emb = data_to_transform_np(X_sc, X_st, sc_gene_idx, st_gene_idx, batch=True)
        sc_emb = emb[:, :n_cells].T
        st_emb = emb[:, n_cells:].T

        train_X, test_X = minmax_scale_cols_np(st_emb, sc_emb)

        # Downsample for tuning if >1000 spots
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
            if verbose:
                print(f"  SVM tuning: downsampled {n_spots} -> {len(ds_idx)} spots")
        else:
            cost, gamma = svm_tune_np(train_X, domain_labels, cross=4)

        pred_st, prob_st, classes = svm_predict_np(train_X, domain_labels, train_X, cost, gamma)
        pred_sc, prob_sc, _ = svm_predict_np(train_X, domain_labels, test_X, cost, gamma)

        max_prob = prob_sc.max(axis=1)
        t_l1 = time.time() - t
        if verbose:
            print(f"[Level 1] {t_l1:.1f}s | C={cost}, gamma={gamma:.6f} | "
                  f"train_acc={np.mean(pred_st == domain_labels):.4f}")

        adata_sc.obs['CMAP_domain'] = pred_sc
        adata_sc.obs['CMAP_svm_prob'] = max_prob

        keep = max_prob > svm_threshold

    n_kept = keep.sum()
    if verbose and n_domains > 1:
        print(f"  Kept {n_kept}/{n_cells} cells (threshold={svm_threshold})")

    # ===== Level 2 =====
    t = time.time()
    domains = np.unique(pred_sc[keep])

    all_spots = []
    all_probs = []
    all_entropy = []
    all_norm_entropy = []
    all_cell_idx = []

    for domain in sorted(domains):
        cell_mask = keep & (pred_sc == domain)
        spot_mask = pred_st == domain

        cell_idx = np.where(cell_mask)[0]
        spot_idx = np.where(spot_mask)[0]

        if len(cell_idx) == 0 or len(spot_idx) == 0:
            continue

        # Domain embedding
        d_emb = data_to_transform_np(
            X_sc[:, cell_idx], X_st[:, spot_idx],
            sc_gene_idx, st_gene_idx, batch=True, pca_method='full'
        )
        n_ci = len(cell_idx)
        cell_feat = d_emb[:, :n_ci].T  # cells x pcs
        spot_feat = d_emb[:, n_ci:].T  # spots x pcs

        # Optimize
        mapping = optimizer_np(cell_feat, spot_feat, device=device,
                               num_epochs=num_epochs, lr=0.1,
                               para_distance=para_distance, para_density=para_density)

        best_spot = np.argmax(mapping, axis=1)
        best_prob = np.max(mapping, axis=1)
        clipped = np.clip(mapping, 1e-10, 1.0)
        ent = -np.sum(clipped * np.log2(clipped), axis=1)
        max_ent = max(np.log2(len(spot_idx)), 1e-10)

        all_cell_idx.extend(cell_idx.tolist())
        all_spots.extend(st_names[spot_idx[best_spot]].tolist())
        all_probs.extend(best_prob.tolist())
        all_entropy.extend(ent.tolist())
        all_norm_entropy.extend((ent / max_ent).tolist())

        if verbose:
            print(f"  Domain {domain}: {len(cell_idx)} cells -> {len(spot_idx)} spots")

    t_l2 = time.time() - t

    # Store Level 2 results
    spot_arr = np.full(n_cells, '', dtype=object)
    prob_arr = np.full(n_cells, np.nan)
    ent_arr = np.full(n_cells, np.nan)
    nent_arr = np.full(n_cells, np.nan)

    idx = np.array(all_cell_idx)
    spot_arr[idx] = all_spots
    prob_arr[idx] = all_probs
    ent_arr[idx] = all_entropy
    nent_arr[idx] = all_norm_entropy

    adata_sc.obs['CMAP_spot'] = spot_arr
    adata_sc.obs['CMAP_spot_prob'] = prob_arr
    adata_sc.obs['CMAP_entropy'] = ent_arr
    adata_sc.obs['CMAP_norm_entropy'] = nent_arr

    if verbose:
        print(f"[Level 2] {t_l2:.1f}s | {len(all_cell_idx)} cells mapped")

    # ===== Level 3 =====
    t = time.time()

    # Pre-compute correlation (one-time)
    emb2 = data_to_transform_np(
        X_sc[:, np.array(all_cell_idx)], X_st,
        sc_gene_idx, st_gene_idx, batch=True
    )
    n_mapped = len(all_cell_idx)
    cor_mat = spearman_cor_np(emb2[:, :n_mapped], emb2[:, n_mapped:])  # cells x spots

    # Build neighborhood
    spot_x, spot_y = spot_coords[:, 0], spot_coords[:, 1]
    defaults = {'honeycomb': (6, 30), 'square': (4, 30), 'singlecell_level': (4, 10)}
    top_n, cutoff = defaults.get(spatial_data_type, (6, 30))
    if n_near_spot is not None:
        top_n = n_near_spot
    if dis_cut is not None:
        cutoff = dis_cut

    k_use = min(top_n + 1, n_spots)
    nn = NearestNeighbors(n_neighbors=k_use, metric='euclidean')
    nn.fit(spot_coords)
    dists, indices = nn.kneighbors(spot_coords)

    # Build neighbor lists as arrays
    spot_name_to_idx = {n: i for i, n in enumerate(st_names)}

    # Auto-detect spot spacing and scale parameters for pixel coordinates
    nn_spacing = NearestNeighbors(n_neighbors=2).fit(spot_coords)
    d_spacing, _ = nn_spacing.kneighbors(spot_coords)
    spot_spacing = np.median(d_spacing[:, 1])

    if dis_cut is None and spot_spacing > 50:
        cutoff = spot_spacing * 1.5
        if verbose:
            print(f"  Auto dis_cut: {cutoff:.1f} (spot spacing={spot_spacing:.1f})")

    if radius < 1 and spot_spacing > 10:
        radius = spot_spacing * radius

    # Compute precise locations
    pred_x = np.full(n_cells, np.nan)
    pred_y = np.full(n_cells, np.nan)
    rng = np.random.default_rng(42)

    for mi, ci in enumerate(all_cell_idx):
        spot_name = spot_arr[ci]
        if spot_name == '' or spot_name not in spot_name_to_idx:
            continue
        si = spot_name_to_idx[spot_name]

        # Get neighbors within cutoff
        neigh_si = [indices[si, j] for j in range(1, k_use) if dists[si, j] < cutoff]

        if len(neigh_si) == 0:
            pred_x[ci] = spot_x[si] + rng.uniform(-radius, radius)
            pred_y[ci] = spot_y[si] + rng.uniform(-radius, radius)
            continue

        # Classify LR vs UD by angle (works for any coordinate scale)
        cx, cy = spot_x[si], spot_y[si]
        lr_idx, ud_idx = [], []
        for ni in neigh_si:
            dx = spot_x[ni] - cx
            dy = spot_y[ni] - cy
            angle = np.abs(np.arctan2(dy, dx))
            if angle < np.pi / 4 or angle > 3 * np.pi / 4:
                lr_idx.append(ni)  # mostly horizontal
            else:
                ud_idx.append(ni)  # mostly vertical
        if not lr_idx and not ud_idx:
            lr_idx = neigh_si
            ud_idx = neigh_si

        # Correlations
        cor_lr = cor_mat[mi, lr_idx] if lr_idx else np.array([])
        cor_ud = cor_mat[mi, ud_idx] if ud_idx else np.array([])
        cor_top = cor_mat[mi, si]

        all_cors = np.concatenate([cor_lr, cor_ud, [cor_top]])
        coeff = expand_diff_np(all_cors)

        n_lr = len(cor_lr)
        n_ud = len(cor_ud)

        if n_lr > 0:
            pred_x[ci] = (np.sum(coeff[:n_lr] * spot_x[lr_idx]) + coeff[-1] * spot_x[si]) / \
                         (np.sum(coeff[:n_lr]) + coeff[-1])
        else:
            pred_x[ci] = spot_x[si] + rng.uniform(-radius, radius)

        if n_ud > 0:
            pred_y[ci] = (np.sum(coeff[n_lr:n_lr + n_ud] * spot_y[ud_idx]) + coeff[-1] * spot_y[si]) / \
                         (np.sum(coeff[n_lr:n_lr + n_ud]) + coeff[-1])
        else:
            pred_y[ci] = spot_y[si] + rng.uniform(-radius, radius)

    t_l3 = time.time() - t

    # Store coordinates
    adata_sc.obsm['spatial_CMAP'] = np.column_stack([pred_x, pred_y])
    adata_sc.obs['CMAP_quality'] = max_prob * (1.0 - nent_arr)

    t_total = time.time() - t_start
    if verbose:
        print(f"[Level 3] {t_l3:.1f}s | {np.isfinite(pred_x).sum()} cells positioned")
        print(f"[Total]   {t_total:.1f}s")

    return adata_sc
