# rust-zig-cpp-simd-lab

A comparative SIMD and low-level optimization laboratory for **Rust**, **Zig 0.16**, and **C++23**.

The goal is not to crown a language from toy examples. It is to implement the **same kernels** in each language, inspect the generated machine code, measure them under the same conditions, and document where each language and toolchain makes high-performance SIMD pleasant, awkward, portable, or fragile.

## What this repo studies

- scalar baselines and compiler autovectorization
- portable/vector-language abstractions
- x86-64 AVX2/FMA specializations
- runtime ISA dispatch patterns
- inline assembly where intrinsics or vector IR are insufficient
- generated assembly and missed-vectorization diagnostics
- throughput, latency, tail handling, alignment, and reduction strategies
- code clarity and maintenance cost alongside raw speed

## Initial kernels

1. **AXPY**: `dst[i] = a * x[i] + y[i]`
2. **Squared error**: `sum((a[i] - b[i])^2)` — representative of MSE/PSNR workloads

These are intentionally simple enough to understand completely while still exposing FMA, reductions, load/store behavior, vector width, and compiler decisions.

## Layout

```text
.
├── cpp/                 # C++23 implementations
├── rust/                # Rust implementations
├── zig/                 # Zig 0.16 implementations
├── docs/                # methodology and codegen notes
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
