# Result bundles

This directory contains authoritative `simd-lab-result-v1` JSON bundles.

Rules:

- one file describes one concrete experiment bundle;
- do not commit synthetic benchmark numbers as real results;
- provenance, environment, target, implementation, numerical contract, and observation units must be explicit;
- raw evidence belongs under `results/artifacts/<result-id>/` and should be referenced from the bundle;
- `site/data/catalog.json` is generated from these files and is not authoritative;
- schema examples belong under `schema/`, not here.

Build the Pages catalog locally with:

```bash
python3 scripts/build_results_catalog.py
```
