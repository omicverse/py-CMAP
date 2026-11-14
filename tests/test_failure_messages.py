"""The two ways a CMAP run dies must say what to do about it.

Both messages exist because the first real end-to-end execution of
`ov.space.CMAP` produced neither. With a domain classifier at chance, Level 1
filtered out every cell, Level 2 then skipped every domain, and the run ended
forty lines later in

    ValueError: No objects to concatenate

from `pd.concat([])` — a message that names neither the stage, the cause, nor
anything the caller can change.
"""

from __future__ import annotations

import numpy as np
import pytest

anndata = pytest.importorskip("anndata")
pytest.importorskip("torch")
import scanpy as sc                                      # noqa: E402

from pycmap.pipeline import run_cmap                      # noqa: E402


def _uninformative_case(n_sp=45, n_sc=60, n_g=40, n_dom=3, seed=0):
    """Spots and cells whose domains are NOT separable from expression.

    Every cell and spot is drawn from the same distribution; only the domain
    *label* varies. A classifier therefore cannot beat 1/n_dom, which is the
    situation the message has to explain.
    """
    rng = np.random.default_rng(seed)
    genes = [f"G{i}" for i in range(n_g)]

    X_sp = rng.poisson(3, size=(n_sp, n_g)).astype(np.float32)
    st = anndata.AnnData(X_sp)
    st.var_names = genes
    st.obs_names = [f"spot{i}" for i in range(n_sp)]
    st.obs["HMRF_cluster"] = [f"D{i % n_dom}" for i in range(n_sp)]
    xs, ys = np.meshgrid(np.arange(n_sp), np.arange(1))
    st.obsm["spatial"] = np.c_[xs.ravel()[:n_sp], ys.ravel()[:n_sp] * 0.0]

    X_sc = rng.poisson(3, size=(n_sc, n_g)).astype(np.float32)
    scd = anndata.AnnData(X_sc)
    scd.var_names = genes
    scd.obs_names = [f"cell{i}" for i in range(n_sc)]

    for a in (st, scd):
        a.layers["counts"] = a.X.copy()
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
    return scd, st, genes


def test_level1_filtering_everything_says_why():
    scd, st, genes = _uninformative_case()
    with pytest.raises(ValueError) as e:
        # cross_para=[3]: the SVM grid search is the slow part and three
        # folds are enough to reach the failure this test is about.
        run_cmap(scd, st, spatial_genes=genes, cluster_key="HMRF_cluster",
                 svm_threshold=0.5, verbose=False, cross_para=[3])
    msg = str(e.value)

    # Names the stage and the quantity the caller controls.
    assert "Level 1 kept 0" in msg
    assert "svm_threshold=0.5" in msg
    # Gives the evidence needed to decide whether to lower it: the best
    # probability actually seen, and what chance would be.
    assert "highest observed" in msg and "chance is" in msg
    # Points at the first thing to check rather than at the knob.
    assert "spatial_genes" in msg
    # And does NOT read like the old failure.
    assert "No objects to concatenate" not in msg


def test_the_old_message_is_gone_from_the_level2_path():
    """`map_cell_to_spot` with nothing to map names the per-domain counts."""
    import pandas as pd

    from pycmap.spot_mapping import map_cell_to_spot

    scd, st, genes = _uninformative_case(n_sp=30, n_sc=30, n_g=30)
    st_norm = pd.DataFrame(st.X.T, index=st.var_names, columns=st.obs_names)
    sc_norm = pd.DataFrame(scd.X.T, index=scd.var_names, columns=scd.obs_names)
    spatial_location = st.obs[["HMRF_cluster"]].copy()
    spatial_location["x"] = st.obsm["spatial"][:, 0]
    spatial_location["y"] = st.obsm["spatial"][:, 1]

    # Cells claim a domain the spots never do: every domain is one-sided.
    pred_sc = pd.Series(["Dx"] * scd.n_obs, index=scd.obs_names)
    pred_st = pd.Series(st.obs["HMRF_cluster"].astype(str).values,
                        index=st.obs_names)

    with pytest.raises(ValueError) as e:
        map_cell_to_spot(sc_norm=sc_norm, sc_meta=scd.obs,
                         st_norm=st_norm, spatial_location=spatial_location,
                         pred_sc_svm=pred_sc, pred_st_svm=pred_st,
                         num_epochs=5)
    msg = str(e.value)
    assert "Level 2 produced no mapping" in msg
    assert "cells /" in msg and "spots" in msg      # the per-domain counts
    assert "No objects to concatenate" not in msg
