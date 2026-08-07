# Benchmark methodology

This repository is intended to compare implementations, not marketing claims.

## Correctness first

Every optimized kernel must be checked against a scalar reference over:

- lengths smaller than one vector
- exact vector-width lengths
- non-multiple tails
- large buffers
- representative and adversarial numeric inputs

Floating-point implementations may differ because of FMA contraction, reassociation, reduction order, or fast-math settings. Tests should therefore use an explicitly documented error tolerance rather than silently demanding bit identity.

## Benchmark dimensions

For every result record:

- CPU model and microarchitecture
- operating system
- compiler and exact version
- optimization mode/flags
- target CPU / ISA features
- kernel and implementation name
- element type
- buffer size
- alignment if controlled
- number of warmup and measured iterations
- median and dispersion, not only the best sample

Useful derived metrics include wall-clock `ns/element`, elements/s, and effective bytes/s. `cycles/element` is valid only when reliable cycle counters measure a dedicated workload with an explicit attribution boundary; it must not be inferred from the current process-aggregate benchmark rows.

## Avoid common benchmark traps

Do not include allocation, random-data generation, logging, or validation in timed regions unless they are specifically what is being measured.

Prevent dead-code elimination by consuming outputs. Keep input generation deterministic. Use enough work that timer resolution is irrelevant, but also test several working-set sizes so L1/L2/L3/DRAM behavior is visible.

Run on an otherwise quiet machine. For serious measurements on Linux, consider CPU affinity, a performance governor, and `perf stat` counters. Record these choices rather than pretending the machine is noise-free; the shared benchmark binaries still do not turn a process aggregate into per-row counter evidence.

## Evidence boundaries

`simd-lab-benchmark-v2` is runtime wall-clock evidence. The current benchmark
binaries are process aggregates: allocation, correctness validation, warmup, and
all benchmark rows run in the measured process. That allocation/validation/
warmup/all-row contamination means a surrounding `perf stat` count cannot be
assigned to one row. Do not make a per-row cycles or cycles/element claim from
those binaries; a dedicated workload (one row or an equivalent explicit
harness boundary) is required for attribution.

Unsupported or not-counted counter output is an error, never zero. A collector
must preserve the failure rather than inventing a numeric observation.

## Stage 9 result adapters

The Linux perf and target-model-aware llvm-mca adapters share
`scripts/result_bundle.py` and emit the existing `simd-lab-result-v1` envelope.
The writer's public helpers are `add_result_arguments(parser)`,
`artifact_record(path, kind, description)`,
`build_result_bundle(args, observations, *, parameters=None, artifacts=None)`,
`paths_alias(first, second)`,
`validate_artifact_output_paths(raw_output, result_output, *, tool)`, and
`write_result(document, args)`. The shared required CLI metadata is
`--id --family --kernel --implementation --target --cpu`. Shared optional
metadata is `--variant --data-type --lanes --isa --semantics --dataset`,
repeatable `--parameter KEY=JSON`, `--source`, `--cpu-feature`,
`--compiler-flag`, and `--tag`, plus `--runner`,
`--virtualized {yes,no,unknown}`, `--language`, `--compiler`,
`--compiler-version`, `--optimization`, `--notes`, `--output`, and `--pretty`.

The adapter CLI shapes are:

```text
python3 scripts/collect_perf_stat.py [shared arguments] \
  --perf perf --event cycles --event instructions --event branches \
  --event branch-misses --repeat 5 \
  --scope {process-aggregate,dedicated-workload} --raw-output RAW -- COMMAND [ARGS...]

python3 scripts/analyze_mca.py [shared arguments] \
  --llvm-mca llvm-mca --iterations 100 --mattr FEATURE ... \
  [--start-label START --end-label END] [--region REGION] \
  --raw-output RAW ASSEMBLY
```

`collect_perf_stat.py` is Linux-only. It invokes
`LC_ALL=C perf stat --json-output --no-big-num --output RAW --repeat N
--event cycles,instructions,branches,branch-misses -- COMMAND`, preserves the
exact command, scope, events, and repeats in `experiment.parameters`, emits a
v1 `counters` observation, and records `RAW` as an artifact. Process scope is
still aggregate evidence; only a dedicated workload supports attribution.

`analyze_mca.py` requires matched, non-nested LLVM-MCA marker regions or an
explicitly extracted `--start-label`/`--end-label` region and never silently
analyzes an unbounded multi-function assembly file. `--target` and `--cpu`
become explicit `-mtriple` and `-mcpu` settings; `--mattr` and `--iterations`
are recorded too.
The parser selects one JSON `CodeRegions` entry and requires
`SummaryView` and `ResourcePressureView`. It records the exact llvm-mca
version, command, target/triple, CPU, features, iterations, and raw JSON
artifact, while emitting a v1 `analysis` observation rather than counters or
runtime metrics. Static estimates are not runtime evidence and are not alone
sufficient to justify inline assembly. No local perf run or authoritative mca
baseline is claimed.
Raw JSON is never written under `results/runs`. When a bundle is persisted
there, its raw output must be persisted under `results/artifacts`; transient
bundle/artifact pairs may use explicit paths elsewhere.

See the [Linux `perf stat` manual](https://man7.org/linux/man-pages/man1/perf-stat.1.html)
and the [LLVM 22.1 `llvm-mca` command guide](https://releases.llvm.org/22.1.0/docs/CommandGuide/llvm-mca.html).

## Compare generated code

When results differ, inspect assembly before inventing a language-level explanation. Useful questions:

1. Did the scalar loop autovectorize?
2. Which vector width was selected?
3. Was FMA emitted?
4. Was the loop unrolled?
5. How is the tail handled?
6. Did alias analysis inhibit vectorization?
7. Is the reduction dependency-bound?
8. Are there unexpected spills, calls, bounds checks, or conversions?
9. Are the implementations actually using the same floating-point semantics?

## Optimization tiers

Use consistent labels:

- `scalar`: intentionally straightforward reference code
- `autovec`: ordinary source written to allow compiler vectorization
- `portable`: language/library vector abstraction independent of a specific ISA
- `avx2`, `avx512`, `neon`, `wasm-simd128`: explicitly ISA-oriented implementation
- `asm`: inline/standalone assembly implementation

A language comparison should normally show more than one tier. Comparing Zig `@Vector` to deliberately scalar Rust/C++ would not answer a useful question.
