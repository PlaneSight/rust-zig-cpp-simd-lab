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

Observed hardware/performance-counter data from tools such as Linux `perf
stat`. In v1 a counter observation has only `kind: "counters"`, `metrics`, and
an optional `tool`; scope, event names, repeat count, command, and parser
status belong in `experiment.parameters` and artifacts rather than invented
observation fields. `<not supported>` and `<not counted>` are collection
errors, never zero-valued metrics.

### `analysis`

Human- or tool-produced interpretation that is explicitly separate from raw
measurements. A v1 analysis observation has `kind: "analysis"` and `summary`,
with optional `severity` and `evidence` references. llvm-mca model data belongs
in its raw JSON artifact and is static analysis, not a runtime measurement or a
hardware-counter observation.

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

## Stage 9 adapter contract

The common writer in `scripts/result_bundle.py` exposes
`add_result_arguments(parser)`, `artifact_record(path, kind, description)`,
`build_result_bundle(args, observations, *, parameters=None, artifacts=None)`,
and `write_result(document, args)`. Both adapters use the required shared
arguments `--id --family --kernel --implementation --target --cpu`. Optional
shared arguments are `--variant --data-type --lanes --isa --semantics
--dataset`, repeatable `--parameter KEY=JSON`, `--source`, `--cpu-feature`,
`--compiler-flag`, and `--tag`, plus `--runner --virtualized {yes,no,unknown}
--language --compiler --compiler-version --optimization --notes --output
--pretty`.
The writer keeps the v1 envelope unchanged while filling current UTC
`created_at`, repository commit/branch and OS/architecture provenance, and
SHA-256 metadata for recorded artifacts. It does not add adapter-specific
fields to `counterObservation` or `analysisObservation`.

The Linux perf adapter is:

```text
python3 scripts/collect_perf_stat.py [shared arguments] \
  --perf perf --event cycles --event instructions --event branches \
  --event branch-misses --repeat 5 \
  --scope {process-aggregate,dedicated-workload} --raw-output RAW -- COMMAND [ARGS...]
```

It runs the documented Linux command
`LC_ALL=C perf stat --json-output --no-big-num --output RAW --repeat N
--event cycles,instructions,branches,branch-misses -- COMMAND`, emits one v1
`counters` observation, and records `RAW` using an artifact record with
`kind`, `path`, optional `sha256`, and `description`. It is Linux-only; no
local run or authoritative PMU baseline is implied. Current benchmark
processes are aggregates contaminated by allocation, validation, warmup, and
all rows, so only a dedicated workload can support row-level attribution.

The llvm-mca adapter is:

```text
python3 scripts/analyze_mca.py [shared arguments] \
  --llvm-mca llvm-mca --iterations 100 --mattr FEATURE ... \
  [--start-label START --end-label END] [--region REGION] \
  --raw-output RAW ASSEMBLY
```

It requires matched, non-nested markers or an explicit start/end-label region,
maps `--target`/`--cpu` to explicit `-mtriple`/`-mcpu`, and records the exact
tool version, command, triple, CPU, features, iterations, and raw JSON artifact.
The parser selects one `CodeRegions` entry and requires `SummaryView` and
`ResourcePressureView`. The bundle carries a v1 `analysis` summary/evidence
reference; numeric model data stays in raw JSON, not `counters` or `runtime`.
Static estimates are not runtime evidence and are not alone sufficient to
justify inline assembly. Raw tool JSON must never use `results/runs`. When a
bundle is persisted there, its raw JSON must be under `results/artifacts`;
transient bundle/artifact pairs may use explicit paths elsewhere.

See the [Linux `perf stat` manual](https://man7.org/linux/man-pages/man1/perf-stat.1.html)
and the [LLVM 22.1 `llvm-mca` command guide](https://releases.llvm.org/22.1.0/docs/CommandGuide/llvm-mca.html).

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
