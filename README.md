# py-CMAP

Pure-Python port of **CMAP** (Cell Mapping Across Positions) — maps individual
cells from scRNA-seq onto precise spatial coordinates using spatial
transcriptomics, through a three-level divide-and-conquer strategy.

Upstream R package: <https://github.com/SuoLab-GZLab/CMAP> (MIT, © 2025 KE Jincan).

| Level | What it does | Entry point |
|---|---|---|
| 1 — DomainDivision | an SVM assigns cells to spatial domains | `predict_domain` |
| 2 — OptimalSpot | optimisation assigns cells to specific spots | `map_cell_to_spot` |
| 3 — PreciseLocation | interpolation gives exact coordinates | `calculate_cell_location` |

## Install

```bash
pip install pycmap                 # core
pip install "pycmap[torch]"        # torch backend for the level-2 optimiser
pip install "pycmap[hmrf]"         # HMRF spatial-gene preprocessing
pip install "pycmap[all]"
```

## Use

```python
import anndata as ad
from pycmap import run_cmap, run_cmap_fast

adata_sc = ad.read_h5ad("sc.h5ad")     # single-cell reference
adata_st = ad.read_h5ad("st.h5ad")     # spatial transcriptomics

result = run_cmap(
    adata_sc, adata_st,
    spatial_genes=spatial_genes,
    spatial_data_type="honeycomb",
)
```

`run_cmap_fast` is the accelerated path over the same three levels.

## Namespace coverage

All **18** functions exported by the R package have a Python counterpart; see
[`NAMESPACE_PARITY.md`](NAMESPACE_PARITY.md) for the one-to-one map, and
[`port.yaml`](port.yaml) for the machine-readable manifest.

## Numerical parity: not yet verified

**This port has not been run against the R reference under a parity gate.**
The namespace is complete and the modules carry `Equivalent to R: …`
annotations, but no measured agreement is published yet, so `port.yaml`
records `parity.status: unverified`.

Until that changes, do not describe this package as parity-validated. The
intended gate, matched to the output classes CMAP produces:

| Output | Class | Proposed metric |
|---|---|---|
| spot assignment | discrete labelling | adjusted Rand index against the R assignment |
| cell coordinates | continuous positions | RMSE, after accounting for the interpolation's tolerance |
| colocalisation counts | counts | rank correlation of the cell-type pair matrix |

A natural fixture is the mouse-olfactory-bulb example shipped with the R
package (`MOB.CMAP.R`).

## Origin of this repository

The code was written as `CMAP_py` inside a private analysis tree and existed
only there. It is published here so it is versioned, installable and citable,
and so the spatial analyses that depend on it can be reproduced.
