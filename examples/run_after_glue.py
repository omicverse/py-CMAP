#!/usr/bin/env python3
"""
Run after GLUE training completes.
Encodes ST spots into GLUE embedding, then runs CMAP mapping for ATAC cells.
"""
import sys, time

import numpy as np
import anndata as ad
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logging.getLogger('harmonypy').setLevel(logging.WARNING)

data_dir = '/scratch/users/steorra/analysis/26_CMAP/CMAP_multi/data/'
fig_dir = '/scratch/users/steorra/analysis/26_CMAP/CMAP_multi/figures/'

import os
os.makedirs(fig_dir, exist_ok=True)

t0 = time.time()

# ===== 1. Load GLUE embeddings =====
print("=" * 60)
print("  CMAP_multi: scATAC-seq -> Spatial Mapping")
print("=" * 60)

adata_atac = ad.read_h5ad(data_dir + 'atac_glue.h5ad')
adata_rna = ad.read_h5ad(data_dir + 'rna_glue.h5ad')
print(f"ATAC: {adata_atac.shape}, GLUE embedding: {adata_atac.obsm['X_glue'].shape}")
print(f"RNA:  {adata_rna.shape}, GLUE embedding: {adata_rna.obsm['X_glue'].shape}")

# ===== 2. Load and cluster Visium data =====
adata_st = ad.read_h5ad(data_dir + 'visium_preprocessed.h5ad')
print(f"Visium: {adata_st.shape}, spatial: {adata_st.obsm['spatial'].shape}")

# Cluster Visium spots (替代HMRF)
sc.pp.scale(adata_st)
sc.tl.pca(adata_st, n_comps=50, mask_var='highly_variable')
sc.pp.neighbors(adata_st, n_pcs=30)
sc.tl.leiden(adata_st, resolution=0.3, key_added='spatial_cluster')
print(f"Spatial clusters: {adata_st.obs['spatial_cluster'].value_counts().to_dict()}")

# ===== 3. Encode ST spots with GLUE =====
# ST spots use the RNA modality of GLUE
# We need to reload the GLUE model and encode ST
print("\nEncoding Visium spots with GLUE...")
import scglue
import networkx as nx

# Load graph and model
graph = nx.read_graphml(data_dir + 'guidance_graph.graphml.gz')

# Re-prepare ST data for GLUE encoding
# ST needs the same gene set as the RNA used in GLUE training
common_genes = list(set(adata_rna.var_names) & set(adata_st.var_names))
print(f"  Common genes (GLUE RNA ∩ ST): {len(common_genes)}")

# Create ST AnnData with matching genes
adata_st_for_glue = adata_st[:, common_genes].copy()
# Need raw counts for NB model
if 'counts' in adata_st.layers:
    adata_st_for_glue.X = adata_st[:, common_genes].layers['counts'].copy()
else:
    # The preprocessed data is already log-normalized, we need raw
    # Re-read raw
    adata_st_raw = sc.read_10x_h5(data_dir + 'visium_mouse_brain.h5')
    adata_st_raw.var_names_make_unique()
    import pandas as pd
    positions = pd.read_csv(data_dir + 'spatial/tissue_positions_list.csv', header=None, index_col=0)
    positions.columns = ['in_tissue', 'row', 'col', 'pxl_row', 'pxl_col']
    common_obs = adata_st_raw.obs_names.intersection(positions.index)
    adata_st_raw = adata_st_raw[common_obs]
    positions = positions.loc[common_obs]
    adata_st_raw = adata_st_raw[positions['in_tissue'] == 1]

    adata_st_for_glue = adata_st_raw[:, [g for g in common_genes if g in adata_st_raw.var_names]].copy()

# Add PCA (required for GLUE rep)
sc.pp.normalize_total(adata_st_for_glue, target_sum=1e4)
sc.pp.log1p(adata_st_for_glue)
adata_st_for_glue.layers['lognorm'] = adata_st_for_glue.X.copy()

# Re-read raw for .X (GLUE NB needs counts)
adata_st_raw2 = sc.read_10x_h5(data_dir + 'visium_mouse_brain.h5')
adata_st_raw2.var_names_make_unique()
common_obs2 = adata_st_for_glue.obs_names.intersection(adata_st_raw2.obs_names)
adata_st_for_glue = adata_st_for_glue[common_obs2]
gene_overlap = [g for g in adata_st_for_glue.var_names if g in adata_st_raw2.var_names]
adata_st_for_glue = adata_st_for_glue[:, gene_overlap].copy()
adata_st_for_glue.X = adata_st_raw2[common_obs2][:, gene_overlap].X.copy()

# Mark HVGs to match GLUE training
rna_hvgs = set(adata_rna.var_names[adata_rna.var['highly_variable']]) if 'highly_variable' in adata_rna.var.columns else set()
adata_st_for_glue.var['highly_variable'] = [g in rna_hvgs for g in adata_st_for_glue.var_names]

# PCA on log-normalized
from scipy.sparse import issparse
X_log = adata_st_for_glue.layers['lognorm']
if issparse(X_log): X_log = X_log.toarray()
from sklearn.decomposition import PCA
pca = PCA(n_components=min(100, X_log.shape[1]-1), random_state=0)
adata_st_for_glue.obsm['X_pca'] = pca.fit_transform(X_log)

# Configure and encode
scglue.models.configure_dataset(adata_st_for_glue, "NB", use_highly_variable=True, use_rep="X_pca")

# Load trained GLUE model
glue = scglue.models.load_model(data_dir + "glue_model/final")
adata_st_for_glue.obsm['X_glue'] = glue.encode_data("rna", adata_st_for_glue)
print(f"  ST GLUE embedding: {adata_st_for_glue.obsm['X_glue'].shape}")

# ===== 4. Run CMAP_multi =====
print(f"\nRunning CMAP mapping ({time.time()-t0:.0f}s elapsed)...")

# Add spatial info to adata_st_for_glue
adata_st_for_glue.obsm['spatial'] = adata_st.obsm['spatial'][
    [list(adata_st.obs_names).index(c) for c in adata_st_for_glue.obs_names]
]
adata_st_for_glue.obs['HMRF_cluster'] = adata_st.obs.loc[adata_st_for_glue.obs_names, 'spatial_cluster'].values
# Rename clusters
adata_st_for_glue.obs['HMRF_cluster'] = 'cluster_' + adata_st_for_glue.obs['HMRF_cluster'].astype(str)

from pycmap.multi.glue_bridge import run_cmap_atac

run_cmap_atac(
    adata_atac, adata_st_for_glue,
    glue_emb_atac=adata_atac.obsm['X_glue'],
    glue_emb_st=adata_st_for_glue.obsm['X_glue'],
    cluster_key='HMRF_cluster',
    spatial_data_type='honeycomb',
    svm_threshold=0.3,
    num_epochs=2000,
    device='cuda',
)

# ===== 5. Visualize =====
print(f"\nGenerating figures...")
coords = adata_atac.obsm['spatial_CMAP']
mapped = np.isfinite(coords[:, 0])
spot_xy = adata_st_for_glue.obsm['spatial']

fig, axes = plt.subplots(1, 3, figsize=(21, 6))

# A: Visium spatial clusters
ax = axes[0]
clusters = sorted(adata_st_for_glue.obs['HMRF_cluster'].unique())
for cl in clusters:
    m = adata_st_for_glue.obs['HMRF_cluster'] == cl
    ax.scatter(spot_xy[m, 0], spot_xy[m, 1], s=20, alpha=0.8, label=cl)
ax.set_aspect('equal'); ax.set_title('Visium Spatial Clusters', fontsize=14)
ax.legend(fontsize=8, markerscale=2)

# B: ATAC cells mapped to space
ax = axes[1]
ax.scatter(coords[mapped, 0], coords[mapped, 1], s=0.3, alpha=0.5, c='steelblue')
ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f'ATAC Cells Mapped ({mapped.sum()} cells)', fontsize=14)

# C: Overlay
ax = axes[2]
ax.scatter(spot_xy[:, 0], spot_xy[:, 1], s=20, alpha=0.2, c='lightgray', label='Spots')
ax.scatter(coords[mapped, 0], coords[mapped, 1], s=0.3, alpha=0.5, c='steelblue', label='ATAC cells')
ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
ax.set_title('ATAC Cells on Spots', fontsize=14)
ax.legend(markerscale=10)

plt.suptitle('CMAP_multi: scATAC-seq → Spatial (via GLUE)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(fig_dir + 'atac_spatial_mapping.png', dpi=200, bbox_inches='tight')
plt.close()

# Quality metrics
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
import pandas as pd
prob = pd.to_numeric(adata_atac.obs['CMAP_spot_prob'], errors='coerce')
ent = pd.to_numeric(adata_atac.obs['CMAP_norm_entropy'], errors='coerce')

axes[0].hist(prob.dropna(), bins=50, color='steelblue', edgecolor='black')
axes[0].set_xlabel('Mapping Probability'); axes[0].set_title('Level 2 Confidence')

axes[1].hist(ent.dropna(), bins=50, color='coral', edgecolor='black')
axes[1].set_xlabel('Normalized Entropy'); axes[1].set_title('Level 2 Uncertainty')

plt.tight_layout()
plt.savefig(fig_dir + 'atac_quality_metrics.png', dpi=150, bbox_inches='tight')
plt.close()

# Save
adata_atac.write_h5ad(data_dir + 'atac_cmap_result.h5ad')

print(f"\n{'=' * 60}")
print(f"  Results")
print(f"{'=' * 60}")
print(f"  Cells mapped: {mapped.sum()} / {len(adata_atac)}")
print(f"  Coordinates: X=[{coords[mapped,0].min():.0f}, {coords[mapped,0].max():.0f}], "
      f"Y=[{coords[mapped,1].min():.0f}, {coords[mapped,1].max():.0f}]")
print(f"  Saved: atac_cmap_result.h5ad")
print(f"  Figures: {fig_dir}")
print(f"  Total time: {time.time()-t0:.0f}s")
