"""
Chen-2019: ATAC + ST only (no scRNA) — the real use case.
Compare with three-modality results.
"""
import sys; sys.path.insert(0, '/scratch/users/steorra/analysis/26_CMAP')
import scanpy as sc, anndata as ad, numpy as np, scglue, networkx as nx, time, logging, pandas as pd
from itertools import chain
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

data_dir = '/scratch/users/steorra/analysis/26_CMAP/CMAP_multi/data/'
chen_dir = data_dir + 'chen2019/'
fig_dir = '/scratch/users/steorra/analysis/26_CMAP/CMAP_multi/figures/'
t0 = time.time()

print("=" * 60)
print("  Chen-2019: ATAC + ST ONLY (no scRNA)")
print("=" * 60)

# Load
atac = ad.read_h5ad(chen_dir + 'atac.h5ad')
st = sc.read_10x_h5(data_dir + 'visium_mouse_brain.h5')
st.var_names_make_unique()
positions = pd.read_csv(data_dir + 'spatial/tissue_positions_list.csv', header=None, index_col=0)
positions.columns = ['in_tissue', 'row', 'col', 'pxl_row', 'pxl_col']
common_obs = st.obs_names.intersection(positions.index)
st = st[common_obs]; st = st[positions.loc[common_obs, 'in_tissue'] == 1]
st.obsm['spatial'] = positions.loc[st.obs_names, ['pxl_col', 'pxl_row']].values.astype(float)

# Preprocess ST
sc.pp.filter_genes(st, min_cells=10)
st.layers["counts"] = st.X.copy()
sc.pp.highly_variable_genes(st, n_top_genes=2000, flavor="seurat_v3")
sc.pp.normalize_total(st); sc.pp.log1p(st); sc.pp.scale(st)
sc.tl.pca(st, n_comps=100, svd_solver="auto")
scglue.data.get_gene_annotation(st, gtf=data_dir+'Mus_musculus.GRCm38.98.gtf.gz', gtf_by='gene_name')
st.var['chrom'] = 'chr' + st.var['chrom'].astype(str)
has = st.var[['chrom','chromStart','chromEnd']].notna().all(axis=1)
st = st[:, has].copy()
sc.pp.neighbors(st, n_pcs=30)
sc.tl.leiden(st, resolution=0.3, key_added='spatial_cluster')
st.obs['HMRF_cluster'] = 'cluster_' + st.obs['spatial_cluster'].astype(str)

# Preprocess ATAC
sc.pp.filter_genes(atac, min_cells=50)
scglue.data.lsi(atac, n_components=100, n_iter=15)
valid, chrom, start, end = [], [], [], []
for p in atac.var_names:
    try:
        c, se = p.split(':'); s, e = se.split('-')
        chrom.append(c); start.append(int(s)); end.append(int(e)); valid.append(p)
    except: pass
atac = atac[:, valid].copy()
atac.var['chrom'] = chrom; atac.var['chromStart'] = start; atac.var['chromEnd'] = end

# Guidance graph: ST ↔ ATAC (no RNA!)
print(f"[{time.time()-t0:.0f}s] Building guidance graph (ST ↔ ATAC)...")
guidance = scglue.genomics.rna_anchored_guidance_graph(st, atac)
cross = guidance.number_of_edges() - sum(1 for u,v in guidance.edges() if u==v)
print(f"  Graph: {guidance.number_of_nodes()} nodes, {cross} cross-edges")

# HVG propagation
st_hvgs = set(st.var.query("highly_variable").index)
atac_reachable = set()
for gene in st_hvgs:
    if gene in guidance:
        for nb in guidance.neighbors(gene):
            if nb in atac.var_names: atac_reachable.add(nb)
atac.var["highly_variable"] = [p in atac_reachable for p in atac.var_names]
print(f"  ATAC HVG: {atac.var['highly_variable'].sum()}")

guidance_hvf = guidance.subgraph(chain(
    st.var.query("highly_variable").index,
    atac.var.query("highly_variable").index
)).copy()
print(f"  HVF subgraph: {guidance_hvf.number_of_nodes()} nodes, {guidance_hvf.number_of_edges()} edges")

# Train two-modality GLUE (ST + ATAC)
st.X = st.layers["counts"].copy()
scglue.models.configure_dataset(st, "NB", use_highly_variable=True, use_layer="counts", use_rep="X_pca")
scglue.models.configure_dataset(atac, "NB", use_highly_variable=True, use_rep="X_lsi")

print(f"\n[{time.time()-t0:.0f}s] Training GLUE (ATAC + ST only)...")
glue = scglue.models.fit_SCGLUE(
    {"st": st, "atac": atac}, guidance_hvf,
    fit_kws={"directory": chen_dir + "glue_model_atac_st"}
)

st.obsm["X_glue"] = glue.encode_data("st", st)
atac.obsm["X_glue"] = glue.encode_data("atac", atac)
print(f"[{time.time()-t0:.0f}s] GLUE Done!")
print(f"  ST: {st.obsm['X_glue'].shape}, range=[{st.obsm['X_glue'].min():.2f}, {st.obsm['X_glue'].max():.2f}]")
print(f"  ATAC: {atac.obsm['X_glue'].shape}, range=[{atac.obsm['X_glue'].min():.2f}, {atac.obsm['X_glue'].max():.2f}]")

# Map ATAC to spatial
print(f"\n[{time.time()-t0:.0f}s] Mapping ATAC to spatial...")
from pycmap.multi.glue_bridge import run_cmap_atac
run_cmap_atac(atac, st, glue_emb_atac=atac.obsm['X_glue'], glue_emb_st=st.obsm['X_glue'],
              cluster_key='HMRF_cluster', spatial_data_type='honeycomb',
              svm_threshold=0.3, num_epochs=2000, device='cuda')

coords = atac.obsm['spatial_CMAP']
mapped = np.isfinite(coords[:, 0])
spot_xy = st.obsm['spatial']

# Cell-type spatial specificity
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=1).fit(spot_xy)
_, spot_idx = nn.kneighbors(coords[mapped])
atac_sub = atac[mapped].copy()
atac_sub.obs['nearest_cluster'] = st.obs['HMRF_cluster'].values[spot_idx.ravel()]

print(f"\n{'='*60}")
print(f"  ATAC+ST only: {mapped.sum()}/{atac.n_obs} cells mapped")
print(f"\nCell type → Visium domain:")
ct = pd.crosstab(atac_sub.obs['cell_type'], atac_sub.obs['nearest_cluster'])
specificity = ct.max(axis=1) / ct.sum(axis=1)
for cell_type in specificity.sort_values(ascending=False).index:
    top = ct.loc[cell_type].idxmax()
    n = ct.loc[cell_type].sum()
    print(f"  {cell_type:20s}: {specificity[cell_type]:.0%} → {top} (n={n})")
print(f"\n  Mean specificity: {specificity.mean():.1%}")

# Save
atac.write_h5ad(chen_dir + 'atac_cmap_2mod.h5ad')
st.write_h5ad(chen_dir + 'st_cmap_2mod.h5ad')

# Plot
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(21, 6))
ct_colors = plt.cm.tab20(np.linspace(0, 1, 22))
cell_types = sorted(atac_sub.obs['cell_type'].unique())

ax = axes[0]
for cl in sorted(st.obs['HMRF_cluster'].unique()):
    m = st.obs['HMRF_cluster'] == cl
    ax.scatter(spot_xy[m, 0], spot_xy[m, 1], s=15, alpha=0.8, label=cl)
ax.set_aspect('equal'); ax.set_title('Visium Clusters'); ax.legend(fontsize=6, ncol=2)

ax = axes[1]
for i, ct_name in enumerate(cell_types):
    m = mapped & (atac.obs['cell_type'].values == ct_name)
    if m.sum() > 0:
        ax.scatter(coords[m, 0], coords[m, 1], c=[ct_colors[i%20]], s=0.5, alpha=0.5, label=ct_name)
ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f'ATAC+ST only ({mapped.sum()} cells)'); ax.legend(fontsize=4, ncol=3, markerscale=8)

ax = axes[2]
spec_sorted = specificity.sort_values(ascending=False)
colors = ['steelblue' if v > 0.5 else 'lightcoral' for v in spec_sorted]
ax.barh(range(len(spec_sorted)), spec_sorted.values, color=colors)
ax.set_yticks(range(len(spec_sorted)))
ax.set_yticklabels(spec_sorted.index, fontsize=7)
ax.axvline(0.5, color='red', ls='--', alpha=0.5)
ax.set_xlabel('Specificity'); ax.set_title('Cell-type Mapping Specificity')
ax.invert_yaxis()

plt.suptitle('CMAP_multi: ATAC+ST Only (no scRNA) — Chen-2019', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(fig_dir + 'chen2019_atac_st_only.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"\nSaved: {fig_dir}chen2019_atac_st_only.png")
print(f"Total: {time.time()-t0:.0f}s")
