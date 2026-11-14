# Namespace parity — R `CMAP` → `pycmap`

Source of truth: the `NAMESPACE` file of the upstream R package, which exports
18 functions. Every one has a Python counterpart.

**Namespace coverage is not numerical parity.** This table says the function
exists; it does not say the two implementations agree on any input. See
`port.yaml` → `parity`, currently `unverified`.

| R export | Python | R source file | Note |
|---|---|---|---|
| `PredictDomain` | `predict_domain` | `PredictDomain.R` | level 1 |
| `data_to_transform` | `data_to_transform` | `PredictDomain.R` | |
| `tune_parameter` | `tune_parameter` | `PredictDomain.R` | SVM hyper-parameters |
| `map_spot_genes` | `find_variable_features` | `CellMapToSpot.R` | renamed; source marks it *Equivalent to R: CMAP::map_spot_genes()* |
| `map_cell_to_spot` | `map_cell_to_spot` | `CellMapToSpot.R` | level 2 |
| `spatial_relation_all` | `spatial_relation_all` | `ExactLocation.R` | |
| `neighbor_spot` | `neighbor_spot` | `ExactLocation.R` | |
| `cor_spot_cell` | `cor_spot_cell` | `ExactLocation.R` | |
| `calculate_cell_location` | `calculate_cell_location` | `ExactLocation.R` | level 3 |
| `cell_type_colocalization` | `run_colocalization_analysis` | `Celltype_colocaliztion.R` | renamed |
| `celltype_colocalization_count` | `celltype_colocalization_count` | `Celltype_colocaliztion.R` | |
| `permute_cell_coordinates` | `permute_cell_coordinates` | `Celltype_colocaliztion.R` | null model |
| `harmony_embedding` | `harmony_embedding` | `Harmony_embedding.R` | via `harmonypy` |
| `hmrf_spatial_gene` | `hmrf_spatial_gene` | `HMRF_gene.R` | needs `[hmrf]` extra |
| `sampling_sp_genes` | `sampling_sp_genes` | `HMRF_gene.R` | |
| `cosine_normalize` | `cosine_normalize` | — | utility |
| `scale_data_by_column` | `scale_data_by_column` | — | utility |
| `expand_diff` | `expand_diff` | — | utility |

## Beyond the R namespace

| Python | Note |
|---|---|
| `run_cmap` | end-to-end AnnData pipeline; the R package has no single entry point |
| `run_cmap_fast` | accelerated path over the same three levels |
| `Mapping`, `map_cell_to_spot_optimize` | optimiser object backing level 2 |
| `log_normalize`, `quantile_normalize`, `scale_data`, `filter_genes` | preprocessing helpers |
| `save_expression_h5ad`, `load_expression_h5ad` | AnnData I/O |
| `load_hmrf_results`, `run_hmrf`, `create_spatial_network`, `spatial_autocorrelation` | HMRF preprocessing |
