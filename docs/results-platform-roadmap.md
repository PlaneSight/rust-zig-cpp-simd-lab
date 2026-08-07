# Results platform roadmap

This document is the forward plan for the structured-results system and GitHub Pages explorer. The authoritative experiment roadmap remains separate; this file describes how evidence is represented, validated, stored, compared, analyzed, and presented.

## Principles

- Result bundles are the source of truth; UI catalogs and derived reports are disposable.
- Every comparison must retain enough provenance to reproduce or reject it.
- Runtime, codegen, numerical semantics, hardware counters, and analysis belong to one versioned evidence model.
- Derived analysis must link back to raw observations and artifacts.
- Comparisons must not silently mix incompatible CPUs, targets, compiler flags, numerical contracts, datasets, or benchmark protocols.
- Schema evolution is explicit and migratable.
- GitHub Pages remains static and inspectable; data generation happens before deployment.

## 1. Producer migration

- [ ] Change Rust benchmark harnesses to emit `simd-lab-result-v1` directly.
- [ ] Change Zig benchmark harnesses to emit `simd-lab-result-v1` directly.
- [ ] Change C++ benchmark harnesses to emit `simd-lab-result-v1` directly.
- [ ] Convert codegen snapshot manifests into result bundles rather than a parallel schema.
- [ ] Convert FP16 semantics reports into result bundles.
- [ ] Convert codegen regression checks into analysis observations.
- [x] Add adapters for Linux `perf stat` and target-model-aware `llvm-mca`.
- [ ] Add adapters for uiCA, compiler optimization remarks, and future profiler outputs.
- [x] Add a common result-writer helper where doing so does not contaminate benchmark timing.

Stage 9 adapter work does not migrate the benchmark producers: Rust, Zig, and
C++ remain `simd-lab-benchmark-v2` producers, and codegen remains
`simd-lab-codegen-v1`. The common writer in `scripts/result_bundle.py` exposes
`add_result_arguments(parser)`, `artifact_record(path, kind, description)`,
`build_result_bundle(args, observations, *, parameters=None, artifacts=None)`,
`paths_alias(first, second)`,
`validate_artifact_output_paths(raw_output, result_output, *, tool)`, and
`write_result(document, args)`. Both adapters require
`--id --family --kernel --implementation --target --cpu`; shared optional
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

The perf adapter is Linux-only and emits a v1 `counters` observation from
`LC_ALL=C perf stat --json-output --no-big-num --output RAW --repeat N
--event cycles,instructions,branches,branch-misses -- COMMAND`; it records the
exact command, scope, events, and repeats in `experiment.parameters`. The mca
adapter emits a v1 `analysis` observation and raw JSON artifact, with explicit
version, triple/target, CPU, features, iterations, and command. Raw JSON never
uses `results/runs`; a persisted bundle there requires its raw JSON under
`results/artifacts`. Transient pairs may use explicit paths elsewhere.
Unsupported or not-counted perf output is an error, never zero.

## 2. Schema hardening

- [x] Validate bundles with full JSON Schema validation in CI, not only envelope checks.
- [ ] Add stable controlled vocabularies for language, architecture, ISA, data type, observation kind, and common metric names.
- [ ] Define canonical units (`ns/element`, `cycles/element`, `GiB/s`, instructions, uops, bytes, etc.).
- [ ] Add benchmark-protocol identity and version.
- [ ] Record clock/turbo/governor/power-policy information where available.
- [ ] Record memory topology, cache sizes, core model, microarchitecture, and relevant CPU errata/stepping where useful.
- [ ] Record compiler backend/LLVM version independently of frontend version.
- [ ] Add explicit build identity: source commit, dirty state, build profile, LTO, codegen units, panic/exception mode, sanitizer state, and debug-info state.
- [ ] Add dataset identity with checksum and generation parameters.
- [ ] Add numerical-contract identifiers rather than relying on free-form prose.
- [ ] Add explicit warmup, sampling, repetition, outlier, and aggregation metadata.
- [ ] Represent unavailable/skipped/unsupported observations without fake zero values.
- [ ] Add artifact MIME/type metadata and integrity hashes.
- [ ] Add relationships between result bundles: baseline, candidate, supersedes, rerun-of, derived-from.
- [ ] Define schema compatibility and deprecation rules.
- [ ] Add migration tooling for future `simd-lab-result-v2+` versions.
Stage 9 preserves the existing v1 shape: counter observations contain only
`kind`, `metrics`, and optional `tool`, while analysis observations contain
`kind`, `summary`, optional `severity`, and optional `evidence`. Scope,
event/repeat metadata, commands, and raw tool output belong in
`experiment.parameters` and `artifacts`; they are not new schema fields.

## 3. Result identity and reproducibility

- [ ] Define deterministic experiment keys separate from unique run IDs.
- [ ] Generate stable fingerprints from kernel + implementation + toolchain + target + protocol + dataset.
- [ ] Detect accidental duplicate runs.
- [ ] Preserve repeated measurements as distinct runs rather than overwriting history.
- [ ] Add reproducibility status: reproducible, environment-dependent, incomplete, or superseded.
- [ ] Add command-line/build-command capture.
- [ ] Add environment capture scripts for Linux, Windows, and macOS.
- [ ] Link each result to exact source files and relevant codegen probe functions.

## 4. Storage and catalog design

- [ ] Partition `results/runs/` by date or experiment family once volume warrants it.
- [ ] Keep immutable historical result bundles where practical.
- [ ] Generate compact catalog shards so the browser does not need to download the entire history.
- [ ] Add indexes by kernel, language, compiler, target, CPU, ISA, data type, and date.
- [ ] Add latest-known-good and baseline indexes.
- [ ] Add artifact indexes for assembly, IR, profiler output, semantic diffs, and logs.
- [ ] Support compressed historical catalogs if repository size becomes material.
- [ ] Define retention rules for bulky raw artifacts versus compact summaries.

## 5. Result detail pages

- [ ] Give every run a stable URL.
- [ ] Show complete provenance and environment metadata.
- [ ] Show all observations without flattening away detail.
- [ ] Link raw JSON, assembly, IR, semantic diff, benchmark stdout, and profiler artifacts.
- [ ] Display source commit and source-file links.
- [ ] Show related runs with the same experiment fingerprint.
- [ ] Show baseline/candidate relationships.
- [ ] Clearly mark incompatible or non-comparable runs.

## 6. Interactive comparisons

- [ ] Select two or more implementations for side-by-side comparison.
- [ ] Compare Rust vs Zig vs C++ for the same kernel/contract.
- [ ] Compare scalar/autovec/native-vector/intrinsics/inline-asm tiers.
- [ ] Compare compiler versions for one implementation.
- [ ] Compare ISA targets: baseline, AVX2, AVX-512, NEON, SVE/SVE2, wasm-simd128.
- [ ] Compare data types and vector widths.
- [ ] Compare native-f16 versus promote-to-f32 versus F16C/native-FP16 paths.
- [ ] Compute speedup relative to a user-selected baseline.
- [ ] Show absolute values and normalized ratios together.
- [ ] Prevent invalid comparisons by default and explain why runs are incompatible.
- [ ] Allow an expert override for intentionally imperfect comparisons.

## 7. Charts and visual analysis

- [ ] Runtime bar/point charts with uncertainty/error intervals.
- [ ] Throughput charts.
- [ ] Speedup-over-baseline charts.
- [ ] Compiler-version trend charts.
- [ ] Instruction-count and vector-instruction trend charts.
- [ ] FP16 conversion-count charts.
- [ ] Tracked-mnemonic/idiom charts (`VPSADBW`, `VPADDUSB`, `UQADD`, etc.).
- [ ] Hardware-counter charts: cycles, instructions, IPC, branches, misses, cache/TLB events.
- [ ] Working-set/cache-size curves.
- [ ] Alignment and tail-size sensitivity plots.
- [ ] Vector-width scaling plots.
- [ ] Roofline-style or bandwidth/compute contextual views where meaningful.
- [ ] Avoid misleading axes and normalize only when the normalization is visible.

## 8. Codegen explorer

- [ ] Syntax-highlight assembly in the browser.
- [ ] Side-by-side assembly diff between implementations/toolchains.
- [ ] Highlight tracked instructions and suspicious conversion traffic.
- [ ] Group assembly by source function/basic block where metadata permits.
- [ ] Link source -> IR -> assembly artifacts.
- [ ] Show compiler optimization/vectorization remarks next to generated code.
- [ ] Surface instruction counts, code size, uop estimates, throughput and latency estimates.
- [ ] Integrate `llvm-mca` reports.
- [ ] Integrate uiCA/x86 analytical predictions where applicable.
- [ ] Add target-specific idiom recognition summaries.

The checked adapter is not a Pages/codegen-explorer integration. The
`llvm-mca` report UI, uiCA predictions, throughput/latency displays, and all
runtime/hardware-counter charts remain unchecked; static estimates remain
clearly distinct from runtime evidence.

## 9. Numerical semantics explorer

- [ ] Dedicated FP16 semantics page.
- [ ] Browse edge cases by input/output bit pattern.
- [ ] Decode binary16/f32/f64 values and classifications.
- [ ] Highlight signed-zero, subnormal, infinity, NaN payload, and rounding differences.
- [ ] Compare strict/native/promoted/fast-math numerical contracts.
- [ ] Show ULP and absolute/relative error where meaningful.
- [ ] Extend semantic testing beyond FP16 to saturation, overflow, shifts, reductions, FMA/reassociation, and conversion boundaries.
- [ ] Link semantic divergences directly to performance comparisons so faster-but-different implementations are obvious.

## 10. Automated analysis

- [ ] Detect runtime regressions relative to a reviewed baseline.
- [ ] Detect instruction-count explosions.
- [ ] Detect suspicious FP16 conversion growth.
- [ ] Detect loss of compact ISA idioms such as SAD/saturating instructions.
- [ ] Detect scalarization or vector-width regression.
- [ ] Detect code-size explosions.
- [ ] Detect semantic regressions separately from performance regressions.
- [ ] Attach machine-generated findings as structured `analysis` observations.
- [ ] Keep generated findings distinguishable from human-reviewed conclusions.
- [ ] Add confidence/evidence metadata to analysis findings.
- [ ] Never label one language/compiler globally faster from a single kernel or host.

## 11. Baselines and historical trends

- [ ] Manually promote reviewed runs/manifests to known-good baselines.
- [ ] Support baseline sets per architecture and CPU family.
- [ ] Track compiler/toolchain upgrades over time.
- [ ] Show when a compiler bug appears, improves, regresses, or disappears.
- [ ] Preserve links to upstream compiler issues and fixes.
- [ ] Mark historical results invalidated by benchmark-methodology changes.
- [ ] Add release-to-release reports for Rust, Zig, LLVM/Clang, and GCC.

## 12. CI and benchmark execution

- [ ] Separate correctness/codegen CI from trusted performance runners.
- [ ] Do not treat noisy shared GitHub runners as authoritative runtime evidence.
- [ ] Support self-hosted benchmark runners with stable hardware configuration.
- [ ] Add runner identity and calibration checks.
- [ ] Detect thermal throttling/frequency instability where possible.
- [ ] Run benchmark repetitions across separate processes, not only inner-loop iterations.
- [ ] Upload raw artifacts before catalog generation.
- [ ] Validate all bundles before publishing Pages.
- [ ] Refuse Pages deployment when authoritative result data is schema-invalid.
- [ ] Generate PR preview artifacts/site bundles where practical.

Stage 9 does not establish a trusted runner or an authoritative benchmark
baseline. Current benchmark processes remain aggregate and contaminated by
allocation, validation, warmup, and all rows; a dedicated workload is required
for attribution, and no per-row cycles claim is valid from the shared binaries.

## 13. Pages UX

- [ ] Shareable URLs encoding filters/comparison selections.
- [ ] Column sorting and configurable visible columns.
- [ ] Pagination/virtualization for large histories.
- [ ] Keyboard-accessible filtering and tables.
- [ ] Accessible chart descriptions and non-color-only encodings.
- [ ] Compact and expanded density modes.
- [ ] Copy/export selected results as JSON/CSV/Markdown.
- [ ] Deep links to result details, raw evidence, source and upstream issues.
- [ ] Saved comparison presets encoded in URLs rather than server state.
- [ ] Mobile-friendly result inspection without sacrificing desktop density.

## 14. Analysis/report pages

- [ ] Kernel overview pages summarizing the current evidence.
- [ ] Language implementation-tier comparisons.
- [ ] Compiler-specific case-study pages.
- [ ] FP16 pathology case study based on `ziglang/zig#19550`.
- [ ] zsmooth/F16 practical case-study page.
- [ ] SAD idiom-recognition case study.
- [ ] Saturating-arithmetic idiom-recognition case study.
- [ ] Architecture pages for x86-64, AArch64 and WebAssembly.
- [ ] Automatically generated 'interesting changes' feed from structured analysis observations.
- [ ] Human-authored conclusions that cite exact result IDs.

## 15. External interoperability

- [ ] Document the result schema as a public interchange format for this project.
- [ ] Provide example producers in Rust, Zig, C++, and Python.
- [ ] Provide JSON -> CSV/Parquet conversion tooling if datasets become large enough.
- [ ] Support importing historical Criterion/hyperfine/custom benchmark data through adapters where provenance is sufficient.
- [ ] Consider export compatible with common visualization/data-analysis tools.
- [ ] Keep the core JSON schema independent of the Pages implementation.

## 16. Integrity and trust

- [ ] Hash raw artifacts and datasets.
- [ ] Detect stale catalog entries.
- [ ] Make generated versus manually authored fields explicit.
- [ ] Preserve the exact raw measurement alongside summaries.
- [ ] Avoid silently editing historical evidence; supersede it with a new result when possible.
- [ ] Record why a result was excluded, invalidated, or superseded.
- [ ] Document benchmark disclosure rules for publication-quality comparisons.

## 17. Future scale

Only add these when the result corpus makes them worthwhile:

- [ ] Client-side columnar data or SQLite/WASM for very large catalogs.
- [ ] Precomputed multidimensional aggregates.
- [ ] Search index for assembly/analysis text.
- [ ] Incremental catalog builds.
- [ ] Separate artifact storage if Git history becomes too large.
- [ ] Public API/static endpoint documentation for third-party consumers.

## Near-term implementation order

1. Merge the schema + Pages foundation.
2. Migrate runtime benchmark producers to `simd-lab-result-v1`.
3. Migrate codegen and FP16 semantic producers.
4. Commit the first real result corpus from controlled runs.
5. Add result-detail pages and baseline-relative comparison.
6. Add interactive charts and shareable comparison URLs.
7. Add assembly/codegen diff views.
8. Add structured automated regression findings.
9. Establish trusted self-hosted performance runners.
10. Build historical compiler/ISA trend views as data accumulates.

Stage 9 is the adapter-only step described above: Linux `perf stat` observed
counter sidecars, bounded llvm-mca static-analysis sidecars, and the common
writer. It does not check the trusted-runner, chart, or UI boxes. See the
[Linux `perf stat` manual](https://man7.org/linux/man-pages/man1/perf-stat.1.html)
and the [LLVM 22.1 `llvm-mca` command guide](https://releases.llvm.org/22.1.0/docs/CommandGuide/llvm-mca.html).

## Definition of done for the results platform

The platform is mature when a result visible on Pages can be traced from a chart or conclusion to a stable result ID, exact source commit, compiler and flags, target/CPU environment, numerical contract, benchmark protocol, raw measurements, and relevant generated artifacts; can be compared only against compatible evidence by default; and can be reproduced or explicitly marked as non-reproducible from the stored provenance.
