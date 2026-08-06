# rust-zig-cpp-simd-lab

A comparative SIMD and low-level optimization laboratory for **Rust**, **Zig 0.16**, and **C++23**.

The goal is not to crown a language from toy examples. It is to implement the **same kernels** in each language, inspect the generated machine code, measure them under the same conditions, and document where each language and toolchain makes high-performance SIMD pleasant, awkward, portable, or fragile.

## Core research question

> How well do Rust, Zig 0.16, and C++23 express and lower common integer and floating-point SIMD idioms across realistic CPU targets?

The lab now explicitly investigates integer and floating-point types including `u8`, `i8`, `u16`, `i16`, `u32`, `i32`, `f16`, `f32`, and `f64`, plus widening and narrowing paths such as `u8 -> u16` and `f16 -> f32 -> f16`.

A major motivating case is [ziglang/zig#19550](https://github.com/ziglang/zig/issues/19550), where vector `f16` clamp code generated dramatically worse machine code than `u8`, `u16`, and `f32` on targets without native FP16 arithmetic. [adworacz/zsmooth#29](https://github.com/adworacz/zsmooth/issues/29) provides a practical video-processing motivation for investigating when native FP16 is viable and when explicit promotion to `f32` is preferable.

See [`docs/type-operation-matrix.md`](docs/type-operation-matrix.md) for the investigation matrix.

## What this repo studies

- scalar baselines and compiler autovectorization
- portable/vector-language abstractions
- common integer and floating-point SIMD idioms
- widening, narrowing, packing, saturation, and conversion
- FP16 native arithmetic versus promotion to FP32
- x86-64 AVX2/FMA/F16C specializations
- AVX-512 and AVX-512 FP16 where available
- runtime ISA dispatch patterns
- inline assembly where intrinsics or vector IR are insufficient
- generated assembly and missed-vectorization diagnostics
- throughput, latency, tail handling, alignment, and reduction strategies
- code clarity and maintenance cost alongside raw speed

## Initial kernels

1. **AXPY**: `dst[i] = a * x[i] + y[i]`
2. **Squared error**: `sum((a[i] - b[i])^2)` — representative of MSE/PSNR workloads

These are intentionally simple enough to understand completely while still exposing FMA, reductions, load/store behavior, vector width, and compiler decisions.

The next common-kernel set covers:

- min / max / clamp
- comparisons and select/blend
- saturating integer arithmetic
- absolute difference and SAD
- horizontal reductions
- integer and floating-point dot products
- widening multiply / accumulate
- narrowing and saturating pack
- `f16 <-> f32` conversion
- short convolution kernels
- weighted pixel blending

## Layout

```text
.
├── cpp/                 # C++23 implementations
├── rust/                # Rust implementations
├── zig/                 # Zig 0.16 implementations
├── docs/                # methodology, experiment matrix, codegen notes
├── scripts/             # helper scripts for repeatable runs
└── .github/workflows/   # compile/test CI
```

## Fair-comparison rules

- Algorithms must be semantically equivalent.
- Scalar reference implementations establish correctness.
- Optimized implementations are compared against the reference before benchmarking.
- Do not compare debug builds.
- Record compiler version, target CPU, enabled ISA features, data size, iteration count, and benchmark methodology.
- Inspect machine code before attributing a result to a language.
- Architecture-specific code is allowed, but must be clearly labeled.
- A faster implementation that changes numerical semantics must say so explicitly.
- Strict FP16 arithmetic and FP16-promoted-to-FP32 arithmetic are separate result classes.
- Never collapse results from materially different ISA feature sets into a single language score.

## FP16 investigation

FP16 is treated as a first-class code-generation experiment rather than just another benchmark type.

We want to compare:

```text
native f16 source arithmetic
optimized / relaxed float mode
f16 -> f32 once -> compute -> f16 once
conversion-heavy f16 <-> f32 negative control
explicit F16C conversion + f32 arithmetic
native AVX-512 FP16
```

A key metric is **conversion hoisting**. For targets without native FP16 arithmetic, good generated code should avoid repeatedly converting every intermediate between `f16` and `f32` when the selected numerical semantics permit the computation to stay widened.

Along with runtime measurements, codegen analysis should count conversion instructions such as `vcvtph2ps` and `vcvtps2ph`, total instructions, spills, vector-width changes, and code size.

## Build

### Rust

```bash
cd rust
cargo test --release
cargo run --release
```

### Zig 0.16

```bash
cd zig
zig build test -Doptimize=ReleaseFast
zig build run -Doptimize=ReleaseFast
```

### C++23

```bash
cmake -S cpp -B build/cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp -j
ctest --test-dir build/cpp --output-on-failure
./build/cpp/simd_lab_cpp
```

## Benchmarking direction

The starter executables are correctness/smoke programs, not a finished benchmarking framework. The next layer should add:

- Criterion-style statistical benchmarking for Rust
- a small shared benchmark protocol for Zig/C++
- CPU pinning where supported
- warmup and multiple sample sizes
- CSV/JSON output
- code-size reporting
- optional Linux `perf stat`
- generated-assembly snapshots
- compiler vectorization diagnostics
- target matrices for AVX2, FMA, F16C, AVX-512, AVX-512 FP16, NEON, and wasm SIMD

## SIMD philosophy

The repo deliberately compares several levels of control:

| Layer | Rust | Zig 0.16 | C++23 |
|---|---|---|---|
| Scalar/autovec | iterators/loops | loops | loops / algorithms |
| Portable explicit vectors | `std::simd` (nightly) / libraries | `@Vector` | `std::experimental::simd` where available / libraries |
| ISA intrinsics | `std::arch` | builtins/extern intrinsics as appropriate | `<immintrin.h>` |
| Inline asm | `asm!` | `asm` | compiler-specific extended asm |

The preferred optimization order is:

1. write a clear scalar reference;
2. see what the optimizer already does;
3. express the algorithm with portable vectors where useful;
4. add ISA-specific intrinsics only when measurements justify them;
5. use inline assembly only when it buys something the compiler cannot express reliably.

## License

MIT
