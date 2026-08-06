# Results schema

The lab uses a single versioned result envelope: `simd-lab-result-v1`.

The JSON Schema lives at `schema/result-bundle-v1.schema.json`. Every persisted result should describe four things before it reports any numbers:

1. **provenance** — repository, commit, branch/workflow and source files;
2. **environment** — OS, architecture, CPU/feature information and compilation target;
3. **experiment** — kernel family, implementation, data type, ISA and numerical contract;
4. **observations** — one or more typed measurements or findings.

## Observation kinds

### `runtime`

Measured performance. Metrics are name/value/unit records so the schema can hold `ns/element`, `GiB/s`, cycles, latency, throughput, allocations, energy or future metrics without changing the envelope.

### `codegen`

Generated-code evidence: instruction counts, vector instruction counts, tracked mnemonic families, top mnemonics and links to assembly/IR artifacts.

### `semantics`

Correctness/numerical-contract comparison. Records whether the implementation matches a named reference, diverges, is skipped, or remains unknown. Detailed per-case output can remain in a separate artifact.

### `counters`

Hardware/performance-counter data from tools such as `perf stat`.

### `analysis`

Human- or tool-produced interpretation that is explicitly separate from raw measurements. Analysis entries carry a severity and evidence references.

## Identity and comparison

A result `id` identifies one concrete experiment bundle. UI grouping should normally use the tuple:

```text
experiment.family
experiment.kernel
experiment.data_type
experiment.dataset
```

Implementations can then be compared across:

```text
toolchain.language
toolchain.compiler
toolchain.version
experiment.implementation
experiment.variant
experiment.isa
environment.target
```

Results from materially different numerical contracts must not be silently ranked together. `experiment.semantics` is therefore part of the visible comparison context.

## Units

Metric names are stable identifiers; units are explicit strings. Prefer SI-ish names that are easy to aggregate:

- `ns_per_element` / `ns`
- `gib_per_second` / `GiB/s`
- `cycles_per_element` / `cycles`
- `instructions` / `count`
- `uops` / `count`
- `branch_misses` / `count`
- `cache_misses` / `count`
- `energy_joules` / `J`

Do not encode units into values or infer them from display labels.

## Raw evidence vs derived analysis

Raw benchmark/codegen/semantic output should remain reproducible artifacts. Derived summaries and UI-friendly catalog rows can be regenerated. A Pages build must never be the only copy of benchmark evidence.

## Schema evolution

Breaking changes create a new schema name (`simd-lab-result-v2`). Readers should key on the top-level `schema` field. Additive optional fields may be introduced in v1 only when existing v1 readers can safely ignore them.

## Recommended result layout

```text
results/
  runs/
    <run-id>.json
  artifacts/
    <run-id>/...
  catalog.json          # generated, not authoritative
```

The GitHub Pages interface consumes the generated catalog, while the individual result bundles remain the authoritative structured records.
