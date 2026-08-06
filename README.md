# rust-zig-cpp-simd-lab

A comparative SIMD and low-level optimization laboratory for **Rust**, **Zig 0.16**, and **C++23**.

The goal is not to crown a language from toy examples. It is to implement the **same kernels** in each language, inspect the generated machine code, measure them under the same conditions, and document where each language and toolchain makes high-performance SIMD pleasant, awkward, portable, or fragile.

A major investigation track is **common integer and floating-point lowering**, motivated in part by the severe `f16` codegen pathology documented in `ziglang/zig#19550` and its practical impact on video-processing code such as `zsmooth`.

> **Project roadmap:** [`docs/roadmap.md`](docs/roadmap.md) is the authoritative backlog for planned types, kernels, ISAs, benchmarks, compiler analysis, numerical validation, image/video case studies, and future inline-assembly work.

## What this repo studies

- scalar baselines and compiler autovectorization
- portable/vector-language abstractions
- common integer types: `u8/i8`, `u16/i16`, `u32/i32`, selected 64-bit cases
- common floating types: `f16`, `f32`, `f64`
- widening, narrowing, packing, and mixed-width arithmetic
- x86-64 AVX2/FMA/F16C specializations
- runtime ISA dispatch patterns
- inline assembly where intrinsics or vector IR are insufficient
- generated assembly and missed-vectorization diagnostics
- throughput, latency, tail handling, alignment, and reduction strategies
- code clarity and maintenance cost alongside raw speed

## Initial kernels

1. **AXPY**: `dst[i] = a * x[i] + y[i]`
2. **Squared error**: `sum((a[i] - b[i])^2)` — representative of MSE/PSNR workloads
3. **Min/max/clamp family** — chosen to expose integer vs floating-point lowering and FP16 conversion behavior

The clamp work includes:

- Zig native `@Vector(16, f16)`
- Zig explicit promote-once `f16 -> f32 -> f16`
- Rust x86 F16C boundary conversion with f32 arithmetic
- C++23 x86 F16C boundary conversion with f32 arithmetic

## Layout

```text
.
├── cpp/                 # C++23 implementations and benchmarks
├── rust/                # Rust implementations and benchmarks
├── zig/                 # Zig 0.16 implementations and benchmarks
├── probes/              # tiny standalone codegen experiments
│   ├── cpp/
│   ├── rust/
│   └── zig/
├── docs/                # methodology, experiment matrix, codegen notes
├── scripts/             # helper scripts for repeatable runs and asm analysis
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
- Language-standard facilities and compiler/ISA extensions must be reported separately.

## Build

### Rust

```bash
cd rust
cargo test --release
cargo run --release
cargo run --release --bin bench
```

### Zig 0.16

```bash
cd zig
zig build test -Doptimize=ReleaseFast
zig build run -Doptimize=ReleaseFast
zig build bench -Doptimize=ReleaseFast
```

### C++23

```bash
cmake -S cpp -B build/cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp -j
ctest --test-dir build/cpp --output-on-failure
./build/cpp/simd_lab_cpp
./build/cpp/simd_lab_cpp_bench
```

## Codegen probes

The files under `probes/` are intentionally minimal exported functions for assembly inspection.

```bash
zig build-obj probes/zig/clamp.zig -O ReleaseFast -femit-asm=zig-clamp.s
rustc -O --crate-type=lib --emit=asm probes/rust/clamp.rs -o rust-clamp.s
clang++ -std=c++23 -O3 -S -masm=intel probes/cpp/clamp.cpp -o cpp-clang-clamp.s
```

Extract simple machine-readable metrics with:

```bash
python scripts/analyze_asm.py --pretty zig-clamp.s rust-clamp.s cpp-clang-clamp.s
```

For the FP16 investigation, conversion counts such as `vcvtph2ps` and `vcvtps2ph` are first-class regression signals.

## Runtime FP16 comparison

The first runtime matrix uses the same exact binary16 values in all three languages and fixed `0.5 .. 2.0` clamp bounds. This deliberately avoids NaNs, subnormals, and rounding-boundary cases until the basic codegen/performance comparison is established.

See:

- `docs/roadmap.md`
- `docs/type-operation-matrix.md`
- `docs/codegen-probes.md`
- `docs/benchmarking.md`
- `docs/fp16-semantics.md`

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
