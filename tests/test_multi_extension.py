"""The cross-modal extension: importable, declared, and not eagerly loaded.

`pycmap.multi` needs scglue and torch, which are an optional extra. The core
package must stay importable without them, so this checks both halves: that
importing `pycmap` does not drag the extension in, and that the extension
works when its dependencies are present.
"""
import importlib
import sys

import pytest


def test_core_does_not_eagerly_import_multi():
    for mod in [m for m in list(sys.modules) if m.startswith("pycmap")]:
        del sys.modules[mod]
    importlib.import_module("pycmap")
    assert "pycmap.multi" not in sys.modules, (
        "importing pycmap pulled in pycmap.multi; the extension needs scglue "
        "and torch and must stay optional"
    )


def test_multi_is_declared_on_the_core_namespace():
    pycmap = importlib.import_module("pycmap")
    assert "multi" in getattr(pycmap, "__all_subpackages__", []), (
        "pycmap.__all_subpackages__ must list the extension"
    )


@pytest.mark.skipif(
    importlib.util.find_spec("scglue") is None,
    reason="pycmap[multi] not installed (scglue missing)",
)
def test_multi_entry_point_is_importable():
    multi = importlib.import_module("pycmap.multi")
    assert hasattr(multi, "run_cmap_atac")
    assert callable(multi.run_cmap_atac)


@pytest.mark.skipif(
    importlib.util.find_spec("scglue") is None,
    reason="pycmap[multi] not installed (scglue missing)",
)
def test_multi_modules_have_no_hardcoded_paths():
    import pathlib
    root = pathlib.Path(importlib.import_module("pycmap.multi").__file__).parent
    offenders = [
        p.name for p in root.glob("*.py")
        if "/scratch/users/" in p.read_text(encoding="utf-8")
    ]
    assert not offenders, (
        "modules still carry a developer's absolute path: " + ", ".join(offenders)
    )
