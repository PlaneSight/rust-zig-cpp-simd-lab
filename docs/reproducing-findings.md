# Reproducing findings

This page is a command-oriented entry point for reproducing the repository's
reviewed findings. It does not turn every command into performance evidence:
correctness, code generation, static analysis, and runtime measurement remain
different evidence classes.

## Start from a clean checkout

Record the revision before running a finding:

```bash
git rev-parse HEAD
git status --short --untracked-files=all
```

Use the pinned toolchains from CI where possible. The repository currently pins
Rust 1.97.1 and Zig 0.16.0. Record the compiler versions, target profile, CPU
features, optimization mode, and operating system alongside any result that is
kept.

## Correctness smoke checks

The repository shortcut runs the Rust release tests, Zig safety-mode tests, and
C++ release smoke test:

```bash
just test
```

The Python checker and fixture suite can be run independently:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

C++ sanitizer coverage follows the CI configuration. On platforms where the
local sanitizer runtime supports it, use a separate build directory and keep
leak-detection settings explicit:

```bash
cmake -S cpp -B build/cpp-sanitizers \
  -DCMAKE_BUILD_TYPE=Debug \
  -DSIMD_LAB_CPU=baseline \
  -DCMAKE_CXX_FLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all' \
  -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address,undefined'
cmake --build build/cpp-sanitizers --target simd_lab_cpp -j2
ASAN_OPTIONS=halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  ctest --test-dir build/cpp-sanitizers --output-on-failure
```

The hosted CI Miri job is the authoritative Rust UB-focused check; a local
Miri run is not implied by the release test command.

## FP16 semantic finding

Run the raw-bit semantic comparison:

```bash
python3 scripts/compare_fp16_semantics.py
```

The command builds and runs the Rust, C++, and Zig semantic dump executables.
Confirm that the report has the expected 23 corpus outputs, inspect the
`available` field for each strategy, and read `divergence_count` before making
any cross-language claim. Rust and C++ F16C rows require an x86 host with F16C;
ARM64 output is useful semantic evidence for Zig but is not F16C execution
evidence. The corpus covers signed zero, subnormals, infinities, quiet NaNs,
and exact boundary values; it does not establish signaling-NaN or arbitrary
floating-point exception behavior.

The numerical contract and interpretation rules are documented in
[`fp16-semantics.md`](fp16-semantics.md).

## u8 SAD and reviewed x86 codegen

The generic u8 SAD contract and the reviewed `VPSADBW` interpretation are in
[`case-study-sad.md`](case-study-sad.md). To regenerate the x86-64-v3 probe
family on a compatible x86 toolchain:

```bash
just probes target=x86-64-v3
python3 scripts/check_codegen_regressions.py \
  results/codegen/known-good/manifest-x86-64-v3.json \
  results/codegen/manifest-x86-64-v3.json \
  --policy codegen-policy.json
```

The checker is a loose regression policy, not an exact assembly-text comparer.
Inspect bounded exported symbols and the reviewed manifest before promoting a
new baseline. An ARM64 host may be unable to generate the x86 profile; that is a
host/toolchain limitation, not evidence that the x86 lowering exists or fails.

## Runtime benchmark finding

Use a temporary output path unless the result is intentionally reviewed for
commitment:

```bash
python3 scripts/run_benchmarks.py \
  --pretty \
  --cpu baseline \
  --output /tmp/simd-lab-benchmark.json
```

For a useful runtime claim, confirm the result schema contains complete host,
toolchain, target, warmup/sample, working-set, and effective-byte metadata.
The current benchmark binaries validate and run multiple rows in one process;
process-aggregate counters cannot be attributed to an individual row. Use the
existing `--perf` adapter only with an explicit scope and a dedicated workload,
and retain its raw output outside `results/runs`.

The benchmark methodology and result schema provide the full metadata and
evidence-boundary rules:

- [`benchmarking.md`](benchmarking.md)
- [`methodology.md`](methodology.md)
- [`results-schema.md`](results-schema.md)

## Cross-target snapshots

Cross-target code generation is compile-only evidence:

```bash
python3 scripts/generate_cross_codegen.py --target aarch64-neon
python3 scripts/generate_cross_codegen.py --target wasm-simd128
```

These commands do not establish runtime support, instruction throughput, or
native hardware behavior. Keep generated snapshots in the ignored results tree
unless they have been manually reviewed and promoted according to
[`codegen-snapshots.md`](codegen-snapshots.md).

## Cleanup and reporting

Before reporting a finding:

```bash
git diff --check
git status --short --untracked-files=all
```

Remove temporary build outputs, generated candidate snapshots, benchmark JSON,
and scratch logs that are not part of the reviewed result. Report the exact
command, revision, host/toolchain metadata, and whether the result is
correctness, codegen, static-model, or runtime evidence. Never replace a
missing host or runtime measurement with a compile-only artifact.
