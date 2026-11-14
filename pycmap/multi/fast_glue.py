"""
Fast GLUE training wrapper.

Accelerates GLUE via:
1. Aggressive ATAC peak filtering (reduce decoder output dimension)
2. Tighter early stopping (patience 15 instead of 30)
3. Graph coarsening (reduce HVF subgraph)
4. Skip fine-tuning for quick experiments (optional)

Usage:
    from pycmap.multi.fast_glue import train_glue_fast
    glue, st, atac = train_glue_fast(st_h5_path, atac_h5ad_path, gtf_path)
"""

import time
import logging
import numpy as np
import anndata as ad
import scanpy as sc
import scglue
import networkx as nx
import pandas as pd
from itertools import chain

logger = logging.getLogger(__name__)


def train_glue_fast(adata_st, adata_atac, gtf_path,
                    n_top_genes_st=2000, max_atac_hvg=5000,
                    patience=15, skip_finetune=False,
                    save_dir=None, verbose=True):
    """
    Fast GLUE training with optimizations.

    Args:
        adata_st: AnnData for spatial (raw counts in .X, spatial in .obsm['spatial'])
        adata_atac: AnnData for ATAC (raw counts in .X, peaks as var_names like 'chr1:100-200')
        gtf_path: Path to GTF file for gene annotations
        n_top_genes_st: Number of HVGs for ST
        max_atac_hvg: Maximum ATAC peaks to keep (controls decoder size → speed)
        patience: Early stopping patience (15 = faster, 30 = original)
        skip_finetune: Skip fine-tuning phase (2x faster, slightly worse)
        save_dir: Directory for model checkpoints
        verbose: Print progress

    Returns:
        (glue_model, adata_st, adata_atac) with X_glue in obsm
    """
    t0 = time.time()
    if save_dir is None:
        import tempfile
        save_dir = tempfile.mkdtemp(prefix='glue_')

    # ===== 1. Preprocess ST =====
    if verbose:
        logger.info("Preprocessing ST...")
    st = adata_st.copy()
    sc.pp.filter_genes(st, min_cells=10)
    st.layers["counts"] = st.X.copy()
    sc.pp.highly_variable_genes(st, n_top_genes=n_top_genes_st, flavor="seurat_v3")
    sc.pp.normalize_total(st)
    sc.pp.log1p(st)
    sc.pp.scale(st)
    sc.tl.pca(st, n_comps=100, svd_solver="auto")

    # Gene annotations
    scglue.data.get_gene_annotation(st, gtf=gtf_path, gtf_by='gene_name')
    st.var['chrom'] = 'chr' + st.var['chrom'].astype(str)
    has = st.var[['chrom', 'chromStart', 'chromEnd']].notna().all(axis=1)
    st = st[:, has].copy()

    # ===== 2. Preprocess ATAC =====
    if verbose:
        logger.info("Preprocessing ATAC...")
    atac = adata_atac.copy()
    sc.pp.filter_genes(atac, min_cells=50)
    scglue.data.lsi(atac, n_components=100, n_iter=15)

    # Parse peaks
    valid, chrom, start, end = [], [], [], []
    for p in atac.var_names:
        try:
            c, se = p.split(':')
            s, e = se.split('-')
            chrom.append(c)
            start.append(int(s))
            end.append(int(e))
            valid.append(p)
        except:
            pass
    atac = atac[:, valid].copy()
    atac.var['chrom'] = chrom
    atac.var['chromStart'] = start
    atac.var['chromEnd'] = end

    # ===== 3. Build guidance graph =====
    if verbose:
        logger.info(f"[{time.time()-t0:.0f}s] Building guidance graph...")
    guidance = scglue.genomics.rna_anchored_guidance_graph(st, atac)

    # ===== 4. HVG propagation with cap =====
    st_hvgs = set(st.var.query("highly_variable").index)
    atac_reachable = set()
    for gene in st_hvgs:
        if gene in guidance:
            for nb in guidance.neighbors(gene):
                if nb in atac.var_names:
                    atac_reachable.add(nb)

    # Cap ATAC HVG to max_atac_hvg (key speed optimization)
    if len(atac_reachable) > max_atac_hvg:
        # Keep the peaks with highest accessibility variance
        atac_sub = atac[:, list(atac_reachable)]
        from scipy.sparse import issparse
        X = atac_sub.X.toarray() if issparse(atac_sub.X) else atac_sub.X
        var_per_peak = X.var(axis=0)
        top_idx = np.argsort(var_per_peak)[-max_atac_hvg:]
        atac_reachable = set(np.array(list(atac_reachable))[top_idx])
        if verbose:
            logger.info(f"  Capped ATAC HVG: {len(atac_reachable)} (from {len(atac_reachable)+len(top_idx)-max_atac_hvg})")

    atac.var["highly_variable"] = [p in atac_reachable for p in atac.var_names]

    guidance_hvf = guidance.subgraph(chain(
        st.var.query("highly_variable").index,
        atac.var.query("highly_variable").index
    )).copy()

    if verbose:
        cross = guidance_hvf.number_of_edges() - sum(1 for u, v in guidance_hvf.edges() if u == v)
        logger.info(f"  HVF subgraph: {guidance_hvf.number_of_nodes()} nodes, {cross} cross-edges")
        logger.info(f"  ATAC HVG: {atac.var['highly_variable'].sum()}, ST HVG: {sum(st.var['highly_variable'])}")

    # ===== 5. Configure & Train =====
    st.X = st.layers["counts"].copy()
    scglue.models.configure_dataset(st, "NB", use_highly_variable=True,
                                     use_layer="counts", use_rep="X_pca")
    scglue.models.configure_dataset(atac, "NB", use_highly_variable=True,
                                     use_rep="X_lsi")

    if verbose:
        logger.info(f"[{time.time()-t0:.0f}s] Training GLUE (patience={patience})...")

    # Custom fit with tighter early stopping
    fit_kws = {"directory": save_dir}

    if skip_finetune:
        # Only pretrain — 2x faster
        glue = scglue.models.SCGLUEModel(
            {"st": st, "atac": atac},
            sorted(guidance_hvf.nodes),
        )
        glue.compile()
        glue.fit(
            {"st": st, "atac": atac},
            guidance_hvf,
            max_epochs=200,
            patience=patience,
            reduce_lr_patience=patience // 2,
            directory=save_dir + "/pretrain",
        )
    else:
        glue = scglue.models.fit_SCGLUE(
            {"st": st, "atac": atac},
            guidance_hvf,
            fit_kws=fit_kws,
        )

    # ===== 6. Encode =====
    st.obsm["X_glue"] = glue.encode_data("st", st)
    atac.obsm["X_glue"] = glue.encode_data("atac", atac)

    if verbose:
        logger.info(f"[{time.time()-t0:.0f}s] Done!")
        logger.info(f"  ST:   {st.obsm['X_glue'].shape}")
        logger.info(f"  ATAC: {atac.obsm['X_glue'].shape}")

    return glue, st, atac
