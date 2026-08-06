# Runtime benchmarking

The runtime harnesses intentionally use the same dataset sizes, warmup count, iteration count, and byte accounting across Rust, Zig, and C++23.

## Current protocol

- elements: `1 << 20`
- warmup iterations: `8`
- measured iterations: `64`
- report: `ns/element` and effective `GiB/s`
- allocations happen outside the timed region
- correctness validation happens before timing
- dispatch checks happen in the timed function only when they are part of the public runtime path being measured

## Current kernel families

### Floating point

- AXPY
- squared error / MSE-style accumulation
- FP16 clamp

### Integer

- `u8` absolute-difference / SAD

The SAD family is particularly useful because it compares a generic source-level idiom against a dedicated x86 instruction family. Rust and C++23 have explicit AVX2 `_mm256_sad_epu8` implementations; Zig expresses `absdiff -> widen u8 to u16 -> reduce` with native vector operations and leaves instruction selection to the backend.

## Shared FP16 dataset

The first FP16 runtime comparison uses identical IEEE-754 binary16 bit patterns in every language:

```text
0x0000  0.0
0x3400  0.25
0x3800  0.5
0x3c00  1.0
0x3e00  1.5
0x4000  2.0
0x4200  3.0
0x4400  4.0
```

The clamp bounds are exactly `0.5 .. 2.0` (`0x3800 .. 0x4000`). These are finite, exactly representable values so the first runtime comparison isolates lowering and conversion cost rather than NaN, signed-zero, or rounding-policy differences.

A separate correctness-only edge corpus lives at `data/fp16-edge-corpus.json`. It includes signed zero, subnormals, values adjacent to clamp boundaries, infinities, negative finite values, and multiple NaN encodings. Edge cases are not mixed into the performance dataset.

## FP16 paths

### Rust / C++23 F16C

Binary16 is stored as `u16`. Eight lanes are widened with F16C, the entire clamp is performed in `f32`, and the result is narrowed once with F16C.

```text
binary16 storage
    -> vcvtph2ps
    -> f32 min/max clamp
    -> vcvtps2ph
    -> binary16 storage
```

Both harnesses validate the resulting half bit patterns against the expected clamp output before timing.

### Zig native f16

The same binary16 bit patterns are materialized as `f16` and processed directly with `@Vector(16, f16)`.

### Zig promote-once

Each vector input is widened once to `@Vector(16, f32)`, the complete clamp is performed in f32, and the result is narrowed once to f16. The harness verifies native and promoted results are bit-identical on the finite exact dataset before timing either path.

## Interpretation

The native-f16 and promote/F16C paths are not assumed to have identical semantics for every possible half value. Runtime speed claims must be paired with the numerical contract used by the implementation. See `docs/fp16-semantics.md` and `data/fp16-edge-corpus.json`.

## Running individual harnesses

Rust:

```bash
cd rust
cargo run --release --bin bench
```

C++23:

```bash
cmake -S cpp -B build/cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp -j
./build/cpp/simd_lab_cpp_bench
```

Zig 0.16:

```bash
cd zig
zig build bench -Doptimize=ReleaseFast
```

CI compiles all benchmark targets but does not use shared-hosted runner timings as performance evidence.

## Machine-readable collection

`scripts/run_benchmarks.py` builds and runs the three harnesses, parses their common text protocol, captures host/toolchain metadata, and emits `simd-lab-benchmark-v1` JSON.

```bash
python scripts/run_benchmarks.py --pretty
python scripts/run_benchmarks.py --pretty --output results/local.json
```

The document contains:

- OS / machine metadata;
- Rust, Cargo, Zig, Clang, GCC, CMake versions when available;
- per-language benchmark metadata;
- normalized `ns_per_element` and `gib_per_second` results;
- skipped ISA-specific cases such as F16C on unsupported hosts.

The JSON collector intentionally sits outside the language harnesses. This keeps benchmark source small and makes it possible to evolve the result schema without maintaining three serialization implementations.
