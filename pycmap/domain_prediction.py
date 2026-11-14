"""
Level 1: Domain Division — SVM-based spatial domain prediction.

Replicates R functions from PredictDomain.R:
- data_to_transform(): Quantile normalize + Harmony embedding
- scale_data_by_column(): Min-max scaling
- tune_parameter(): SVM hyperparameter tuning
- PredictDomain(): SVM classification
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from .utils import quantile_normalize, scale_data_by_column
from .harmony import harmony_embedding


def data_to_transform(sc_norm, st_norm, spatial_genes, batch=True, npc=100,
                      nclust=100, pca_method='truncated_svd',
                      max_iter_harmony=50, max_iter_cluster=6,
                      cosine_norm=False):
    """
    Transform expression data: quantile normalization + Harmony batch correction.

    Equivalent to R: CMAP::data_to_transform()

    Args:
        sc_norm: Single-cell expression (genes x cells), DataFrame.
        st_norm: Spatial expression (genes x spots), DataFrame.
        spatial_genes: List of spatial genes to use.
        batch: Whether to apply batch correction.
        npc: Number of PCs.
        nclust: Number of Harmony clusters.
        pca_method: 'pca' or 'truncated_svd'.
        max_iter_harmony: Max Harmony iterations.
        max_iter_cluster: Max k-means iterations.
        cosine_norm: Whether to cosine normalize.

    Returns:
        Embedding matrix (PCs x samples), DataFrame with columns = sample IDs.
    """
    # Intersect genes
    genes = [g for g in spatial_genes if g in sc_norm.index and g in st_norm.index]

    # Quantile normalize ST and SC separately
    st_qm_vals = quantile_normalize(st_norm.loc[genes].values.astype(np.float64))
    sc_qm_vals = quantile_normalize(sc_norm.loc[genes].values.astype(np.float64))
    # 极小子集(1-2列)的零方差会产生 NaN, 下游 PCA 拒收 -> 置零保护
    st_qm_vals = np.nan_to_num(st_qm_vals, nan=0.0)
    sc_qm_vals = np.nan_to_num(sc_qm_vals, nan=0.0)

    st_qm = pd.DataFrame(st_qm_vals, index=genes, columns=st_norm.columns)
    sc_qm = pd.DataFrame(sc_qm_vals, index=genes, columns=sc_norm.columns)

    if batch:
        npc_use = min(npc, st_qm.shape[1], sc_qm.shape[1])
        nclust_use = min(nclust, sc_qm.shape[1] // 10 + 1, st_qm.shape[1] // 10 + 1)
        # R uses ceiling(dim/10)
        nclust_use = min(nclust, int(np.ceil(sc_qm.shape[1] / 10)),
                         int(np.ceil(st_qm.shape[1] / 10)))

        matrix = harmony_embedding(
            st_norm=st_qm,
            sc_norm=sc_qm,
            genes=genes,
            batch=batch,
            npc=npc_use,
            nclust=nclust_use,
            pca_method=pca_method,
            max_iter_harmony=max_iter_harmony,
            max_iter_cluster=max_iter_cluster,
            cosine_norm=cosine_norm,
        )
    else:
        matrix = pd.concat([sc_qm, st_qm], axis=1)

    return matrix


def tune_parameter(train_set, test_set, kernel='rbf', scale=True,
                   class_weight=True, verbose=False, cross_para=None):
    """
    Tune SVM hyperparameters (C, gamma) via grid search cross-validation.

    Equivalent to R: CMAP::tune_parameter()

    Args:
        train_set: DataFrame (samples x features + 'label' column).
        test_set: DataFrame (samples x features).
        kernel: SVM kernel type ('rbf' for R 'radial').
        scale: Whether to min-max scale data.
        class_weight: Whether to use balanced class weights.
        verbose: Print best parameters.
        cross_para: List of cross-validation fold counts (default [4]).

    Returns:
        Dict of {f'cross_{k}': {'cost': C, 'gamma': gamma}} for each fold count.
    """
    if cross_para is None:
        cross_para = [4]

    parameters = {}

    # Scale data
    if scale:
        scaled_train, scaled_test = scale_data_by_column(train_set, test_set)
    else:
        scaled_train = train_set.copy()
        scaled_test = test_set.copy()

    # Separate features and labels
    train_labels = scaled_train['label']
    train_features = scaled_train.drop(columns=['label'])

    # Class weights
    weight = 'balanced' if class_weight else None

    from sklearn.model_selection import StratifiedKFold

    for cross_i in cross_para:
        svc = SVC(kernel=kernel, probability=True, class_weight=weight,
                  random_state=321, cache_size=500)
        cv = StratifiedKFold(n_splits=cross_i, shuffle=True, random_state=321)

        # Two-stage tuning: coarse then fine (60 combos vs original 110)
        # Stage 1: Coarse grid
        C_coarse = [2 ** i for i in range(-3, 16, 3)]     # 7 values
        gamma_coarse = [2 ** i for i in range(-15, 4, 4)]  # 5 values
        gs1 = GridSearchCV(svc, {'C': C_coarse, 'gamma': gamma_coarse},
                           cv=cv, scoring='accuracy', n_jobs=-1)
        gs1.fit(train_features.values, train_labels.values)

        # Stage 2: Fine grid around best
        C_fine = [gs1.best_params_['C'] * f for f in [0.25, 0.5, 1, 2, 4]]
        gamma_fine = [gs1.best_params_['gamma'] * f for f in [0.25, 0.5, 1, 2, 4]]
        grid_search = GridSearchCV(svc, {'C': C_fine, 'gamma': gamma_fine},
                                   cv=cv, scoring='accuracy', n_jobs=-1, refit=True)
        grid_search.fit(train_features.values, train_labels.values)

        best_C = grid_search.best_params_['C']
        best_gamma = grid_search.best_params_['gamma']

        parameters[f'cross_{cross_i}'] = {'cost': best_C, 'gamma': best_gamma}

        if verbose:
            print(f"SVM: cross={cross_i}")
            print(f"  best cost: {best_C}")
            print(f"  best gamma: {best_gamma}")
            print(f"  best score: {grid_search.best_score_:.4f}")

    return parameters


def predict_domain(train_set, test_set, scale=True, class_weight=True,
                   cost=1, gamma=None, kernel='rbf', st_svm=False, verbose=False):
    """
    Predict spatial domain using SVM classifier.

    Equivalent to R: CMAP::PredictDomain()

    Args:
        train_set: DataFrame (samples x features + 'label' column).
        test_set: DataFrame (samples x features).
        scale: Whether to min-max scale.
        class_weight: Whether to use balanced class weights.
        cost: SVM C parameter.
        gamma: SVM gamma parameter (default: 1/n_features).
        kernel: SVM kernel ('rbf' for R 'radial').
        st_svm: If True, return predictions on training data.
        verbose: Print prediction stats.

    Returns:
        Tuple of (predictions, probabilities).
        - predictions: pandas Series with predicted labels.
        - probabilities: DataFrame (samples x classes) with prediction probabilities.
    """
    # Scale data
    if scale:
        scaled_train, scaled_test = scale_data_by_column(train_set, test_set)
    else:
        scaled_train = train_set.copy()
        scaled_test = test_set.copy()

    # Separate features and labels
    train_labels = scaled_train['label']
    train_features = scaled_train.drop(columns=['label'])

    if gamma is None:
        gamma = 1.0 / train_features.shape[1]

    # Class weights
    weight = 'balanced' if class_weight else None

    # Train SVM
    svm = SVC(C=cost, gamma=gamma, kernel=kernel, probability=True,
              class_weight=weight, random_state=321)
    svm.fit(train_features.values, train_labels.values)

    # Predict on training data
    pred_st = svm.predict(train_features.values)
    prob_st = svm.predict_proba(train_features.values)
    pred_st = pd.Series(pred_st, index=train_features.index, name='prediction')
    prob_st = pd.DataFrame(prob_st, index=train_features.index, columns=svm.classes_)

    # Predict on test data
    pred_sc = svm.predict(scaled_test.values)
    prob_sc = svm.predict_proba(scaled_test.values)
    pred_sc = pd.Series(pred_sc, index=scaled_test.index, name='prediction')
    prob_sc = pd.DataFrame(prob_sc, index=scaled_test.index, columns=svm.classes_)

    if verbose:
        print("Prediction table of training data:")
        print(pred_st.value_counts())
        accuracy = (pred_st.values == train_labels.values).mean()
        print(f"Accuracy of spatial data: {accuracy:.4f}")
        print("\nPrediction table of test data:")
        print(pred_sc.value_counts())

    if st_svm:
        return pred_st, prob_st

    return pred_sc, prob_sc
