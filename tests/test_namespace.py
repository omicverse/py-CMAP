"""The namespace claim in NAMESPACE_PARITY.md, checked by machine.

The R package exports 18 functions. README and port.yaml both claim every one
has a Python counterpart. This test is what makes that claim checkable rather
than a sentence someone wrote once.
"""
import importlib

import pytest

# R export -> attribute expected on the pycmap namespace.
# Kept in lockstep with NAMESPACE_PARITY.md and port.yaml.
R_TO_PY = {
    "PredictDomain": "predict_domain",
    "data_to_transform": "data_to_transform",
    "tune_parameter": "tune_parameter",
    "map_spot_genes": "find_variable_features",
    "map_cell_to_spot": "map_cell_to_spot",
    "spatial_relation_all": "spatial_relation_all",
    "neighbor_spot": "neighbor_spot",
    "cor_spot_cell": "cor_spot_cell",
    "calculate_cell_location": "calculate_cell_location",
    "cell_type_colocalization": "run_colocalization_analysis",
    "celltype_colocalization_count": "celltype_colocalization_count",
    "permute_cell_coordinates": "permute_cell_coordinates",
    "harmony_embedding": "harmony_embedding",
    "hmrf_spatial_gene": "hmrf_spatial_gene",
    "sampling_sp_genes": "sampling_sp_genes",
    "cosine_normalize": "cosine_normalize",
    "scale_data_by_column": "scale_data_by_column",
    "expand_diff": "expand_diff",
}

EXTENSIONS = ["run_cmap", "run_cmap_fast", "Mapping", "map_cell_to_spot_optimize"]


@pytest.fixture(scope="module")
def pycmap():
    return importlib.import_module("pycmap")


def test_r_namespace_is_covered(pycmap):
    missing = [f"{r} -> {py}" for r, py in R_TO_PY.items() if not hasattr(pycmap, py)]
    assert not missing, "R exports without a Python counterpart: " + ", ".join(missing)


def test_r_export_count_is_eighteen():
    assert len(R_TO_PY) == 18, "the upstream NAMESPACE exports 18 functions"


def test_extensions_present(pycmap):
    missing = [n for n in EXTENSIONS if not hasattr(pycmap, n)]
    assert not missing, "declared extensions missing: " + ", ".join(missing)


def test_version_is_declared(pycmap):
    assert getattr(pycmap, "__version__", None), "pycmap.__version__ must be set"


def test_callables_are_callable(pycmap):
    not_callable = [
        py for py in R_TO_PY.values()
        if hasattr(pycmap, py) and not callable(getattr(pycmap, py))
    ]
    assert not not_callable, "not callable: " + ", ".join(not_callable)
