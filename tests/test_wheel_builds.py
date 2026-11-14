"""The wheel must build, and `multi` must land only inside `pycmap`.

This exists because it did not. `[tool.hatch.build.targets.wheel]` listed
`packages = ["pycmap", "pycmap/multi"]`; hatchling maps every listed package
to the wheel root, so the second entry tried to write `multi/__init__.py`
alongside the copy already inside `pycmap/`, and the build died with

    ValueError: A second file is being added to the wheel archive at the
    same path: `multi/__init__.py`.

Every import test in this suite passed throughout, because they run against
the source tree. Nothing exercised the one thing a user does first —
`pip install`. So this test builds the real artifact and reads its manifest.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import zipfile

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _build(tmp_path: pathlib.Path) -> pathlib.Path:
    # Build isolation on purpose: pip fetches the declared backend itself, so
    # the test exercises what a user's `pip install` does rather than whatever
    # happens to be importable here. (An earlier draft passed
    # --no-build-isolation and skipped whenever hatchling was absent — which is
    # most interpreters, so the test was inert and read as a pass.)
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps",
         "-w", str(tmp_path), str(ROOT)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        blob = proc.stdout + proc.stderr
        offline = any(s in blob for s in (
            "Temporary failure in name resolution", "Network is unreachable",
            "Could not find a version that satisfies the requirement hatchling",
            "Failed to establish a new connection",
        ))
        if offline:
            pytest.skip("no network: pip cannot fetch the build backend")
        raise AssertionError(
            "wheel build failed:\n" + proc.stdout[-4000:] + proc.stderr[-4000:]
        )
    wheels = sorted(tmp_path.glob("pycmap-*.whl"))
    assert wheels, f"no wheel produced; pip said:\n{proc.stdout[-2000:]}"
    return wheels[-1]


def test_wheel_builds_and_has_one_multi(tmp_path):
    wheel = _build(tmp_path)
    names = zipfile.ZipFile(wheel).namelist()

    # The extension ships, inside the package.
    assert "pycmap/multi/__init__.py" in names, \
        f"pycmap.multi missing from the wheel: {names[:20]}"

    # And nowhere else. A top-level `multi/` is the duplicate-path bug
    # reappearing: it would also squat the import name `multi` for every
    # other package in the environment.
    stray = [n for n in names if n.startswith("multi/")]
    assert not stray, f"`multi` also at the wheel root: {stray}"

    # The core modules are present too — a packages= typo that dropped them
    # would otherwise still satisfy the assertions above.
    for mod in ("pycmap/__init__.py", "pycmap/pipeline.py",
                "pycmap/spot_mapping.py"):
        assert mod in names, f"{mod} missing from the wheel"


def test_declared_packages_do_not_nest():
    """A listed package must not be inside another listed package.

    This is the invariant that was violated, checked directly so the
    failure names the cause rather than a zip path. Guarded on tomllib so
    the suite still runs on 3.10 without a toml backend.
    """
    try:
        import tomllib
    except ModuleNotFoundError:          # py<3.11
        tomllib = pytest.importorskip("tomli")

    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pkgs = (cfg.get("tool", {}).get("hatch", {}).get("build", {})
               .get("targets", {}).get("wheel", {}).get("packages", []))
    assert pkgs, "wheel target declares no packages"
    for a in pkgs:
        for b in pkgs:
            if a != b:
                assert not a.startswith(b.rstrip("/") + "/"), (
                    f"`{a}` is inside `{b}`; hatchling maps both to the wheel "
                    "root and the build fails on the duplicate path. List only "
                    "the top-level package."
                )
