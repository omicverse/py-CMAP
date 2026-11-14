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
pip install pycmap                 # core — torch comes with it
pip install "pycmap[hmrf]"         # + HMRF spatial-gene preprocessing
pip install "pycmap[multi]"        # + the cross-modal scATAC extension
pip install "pycmap[all]"
```

torch is a **required** dependency, not an extra: the level-2 optimiser, the
SSIM scoring and the fast paths import it at module scope, so `import pycmap`
needs it. `pycmap[torch]` still resolves, for anything written against 1.1.0,
but it adds nothing.

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

## Cross-modal extension — `pycmap.multi`

Maps **scATAC-seq onto spatial coordinates**, which the R package does not do.
It bridges the modality gap with GLUE guidance-graph embeddings, then runs
CMAP's three levels for the spatial assignment.

```bash
pip install "pycmap[multi]"     # scglue, torch, networkx, matplotlib
```

```python
from pycmap.multi import run_cmap_atac
adata_atac = run_cmap_atac(adata_rna, adata_atac, adata_st, ...)
```

`pycmap` itself does **not** import this subpackage, so the core install stays
free of the GPU stack.

The design notes carried over from development are in
[`docs/multi_DESIGN.md`](docs/multi_DESIGN.md), including the preprocessing
checklist that the implementation actually needed (chromosome-name matching
between a GTF-derived RNA and 10x ATAC, the negative-binomial rather than
Bernoulli ATAC model, HVG propagation into the guidance graph) and the measured
progression across three attempts:

| Version | Same-cell correlation | Top-10 match | Training |
|---|---|---|---|
| v1, Bernoulli, unbinarised | 0.054 | — | 100 min (embedding blew up) |
| v2, binarised | 0.346 | 3.8 % | 60 min (no guidance graph — chromosome names mismatched) |
| **v3, NB + chromosome fix + HVF subgraph** | **0.804** | **25.7 %** | **36 min** |

Those are development figures from the authors' own runs, not an independent
benchmark.

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
