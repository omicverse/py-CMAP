# Changelog

## 1.1.0 — first public release

Published from a private analysis tree where the code existed as `CMAP_py`
and had no remote. Version number carried over from that source rather than
reset, so the history of the code is not misrepresented.

- Three-level CMAP port: SVM domain assignment, spot optimisation, coordinate interpolation
- `run_cmap` end-to-end AnnData pipeline and the accelerated `run_cmap_fast`
- Colocalisation analysis with a permutation null
- HMRF spatial-gene preprocessing (optional extra)
- Namespace covers all 18 R exports — see `NAMESPACE_PARITY.md`

**Not included:** a verified numerical parity gate against the R reference.
`port.yaml` records `parity.status: unverified`.
