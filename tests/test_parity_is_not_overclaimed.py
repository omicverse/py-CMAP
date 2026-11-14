"""Guards the one claim this package must not make by accident.

Namespace coverage is not numerical parity. Until a parity gate has actually
been run against the R reference, port.yaml must say so, and the README must
not describe the port as validated. This test fails if someone flips the
manifest without supplying a metric, a threshold and a measured value.
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _parity_block() -> dict:
    text = (ROOT / "port.yaml").read_text(encoding="utf-8")
    block = re.search(r"^parity:\n((?:[ \t]+.*\n|\n)*)", text, re.M)
    assert block, "port.yaml has no parity block"
    out = {}
    for line in block.group(1).splitlines():
        m = re.match(r"\s+(\w+):\s*(.*?)\s*(?:#.*)?$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def test_parity_status_is_honest():
    parity = _parity_block()
    status = parity.get("status")
    if status == "unverified":
        return  # the current, honest state
    evidence = [k for k in ("metric", "threshold", "value", "fixture", "test")
                if parity.get(k) in (None, "", "null")]
    assert not evidence, (
        f"parity.status is {status!r} but these carry no evidence: {evidence}. "
        "A port is parity-validated only when a metric, a threshold, a measured "
        "value, the fixture it was measured on and the test that measures it are "
        "all recorded."
    )


def test_readme_does_not_claim_validation_while_unverified():
    parity = _parity_block()
    if parity.get("status") != "unverified":
        return
    readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    assert "not yet verified" in readme, (
        "port.yaml says parity is unverified, so README.md must say so too"
    )
