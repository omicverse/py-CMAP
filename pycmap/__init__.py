"""
CMAP_py: Pure Python implementation of CMAP (Cell Mapping Across Positions).

Maps individual cells from scRNA-seq to precise spatial locations
using spatial transcriptomics data through a 3-level divide-and-conquer strategy:
  Level 1 (DomainDivision): SVM assigns cells to spatial domains
  Level 2 (OptimalSpot): Optimization assigns cells to specific spots
  Level 3 (PreciseLocation): Interpolation provides exact coordinates

Original R package: https://github.com/SuoLab-GZLab/CMAP
"""

__version__ = "1.1.2"

# Core utilities
from .utils import (
    log_normalize,
    quantile_normalize,
    scale_data,
    scale_data_by_column,
    expand_diff,
    cosine_normalize,
    filter_genes,
    save_expression_h5ad,
    load_expression_h5ad,
)

# Batch correction
from .harmony import harmony_embedding

# Level 1: Domain prediction
from .domain_prediction import (
    data_to_transform,
    tune_parameter,
    predict_domain,
)

# Level 2: Spot mapping
from .spot_mapping import (
    find_variable_features,
    map_cell_to_spot,
)

# Level 3: Exact location
from .exact_location import (
    spatial_relation_all,
    neighbor_spot,
    cor_spot_cell,
    calculate_cell_location,
)

# Colocalization analysis
from .colocalization import (
    celltype_colocalization_count,
    permute_cell_coordinates,
    run_colocalization_analysis,
)

# Preprocessing / HMRF
from .preprocessing import (
    load_hmrf_results,
    hmrf_spatial_gene,
    run_hmrf,
    create_spatial_network,
    spatial_autocorrelation,
    sampling_sp_genes,
)

# Optimizer
from .optimizer import (
    Mapping,
    map_cell_to_spot_optimize,
)

# End-to-end pipeline (AnnData API)
from .pipeline import run_cmap
from .fast import run_cmap_fast

# Cross-modal extension (scATAC-seq -> spatial) via GLUE guidance graphs.
# Deliberately NOT imported here: it needs scglue + torch, which are an
# optional extra. Import it explicitly:
#     from pycmap import multi            # requires pycmap[multi]
#     from pycmap.multi import run_cmap_atac
__all_subpackages__ = ["multi"]
