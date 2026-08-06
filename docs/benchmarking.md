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

### Zig native f16

The same values are materialized as `f16` and processed directly with `@Vector(16, f16)`.

### Zig promote-once

Each vector input is widened once to `@Vector(16, f32)`, the complete clamp is performed in f32, and the result is narrowed once to f16.

## Interpretation

The native-f16 and promote/F16C paths are not assumed to have identical semantics for every possible half value. The current exact finite dataset is deliberately chosen so all paths should agree bit-for-bit. Later semantic tests should add:

- positive and negative zero
- infinities
- quiet/signaling NaNs where representable and meaningful
- subnormals
- values near rounding boundaries
- min/max ordering edge cases

Runtime speed claims must be paired with the numerical contract used by the implementation.

## Running

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

The next step is machine-readable JSON/CSV output so results from multiple compiler/ISA configurations can be aggregated without scraping terminal text.
