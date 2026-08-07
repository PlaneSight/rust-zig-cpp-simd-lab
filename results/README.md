# Results layout

The `results/` tree separates reviewed evidence from generated or local output.
The files under it are not interchangeable.

## Directories

| Directory | Ownership | Commit policy |
|---|---|---|
| `codegen/` | Generated target manifests and assembly snapshots | Keep generated candidates ignored; keep only manually reviewed known-good manifests and explicitly reviewed cross-target evidence. |
| `artifacts/` | Raw assembly, semantic dumps, benchmark metadata, and other provenance | Keep artifacts only when they are referenced by a reviewed result or case study and include host/toolchain context. |
| `runs/` | Machine-readable `simd-lab-result-v1` experiment bundles | One file per concrete experiment; preserve identity, target, implementation, semantics, units, and observation provenance. |

`results/runs/README.md` defines the bundle-level rules. Schemas and examples
belong under `schema/`; do not duplicate schema files inside result directories.

## Generated presentation data

`site/data/catalog.json` is generated from reviewed `results/runs/` bundles by:

```bash
python3 scripts/build_results_catalog.py
```

Edit the source bundle, not the catalog. Do not treat the catalog or a hosted
page as an independent benchmark source.

## Evidence boundaries

- Codegen manifests and assembly are compile/static evidence, not runtime
  throughput.
- Runtime bundles are comparable only within their recorded host, compiler,
  target, optimization, dataset, and sampling metadata.
- Process-aggregate counter output cannot be assigned to an individual benchmark
  row without a dedicated workload boundary.
- Cross-target output establishes a lowering artifact, not execution support on
  the target ISA.

Use temporary paths such as `/tmp/` for local candidates and scratch output.
Before committing a result, check that the bundle references its raw artifacts,
contains complete metadata, and has a corresponding human-readable
interpretation in `docs/`.
