# Runtime benchmarking

The runtime harnesses use the same size sweep, sampling policy, byte accounting, and output protocol across Rust, Zig, and C++23. The purpose is to expose cache transitions and run-to-run noise, not to compress a machine into one headline number.

## Protocol

Each kernel runs at six element counts:

| Elements | One f32 array | Typical regime |
|---:|---:|---|
| `1 << 10` | 4 KiB | L1-resident |
| `1 << 13` | 32 KiB | L1 boundary |
| `1 << 16` | 256 KiB | L2-resident |
| `1 << 18` | 1 MiB | L2 boundary on many CPUs |
| `1 << 20` | 4 MiB | shared cache / memory transition |
| `1 << 22` | 16 MiB | shared cache or DRAM |

The actual working set depends on the kernel and is emitted for every result. AXPY has three f32 arrays; squared error has two; SAD has two byte arrays; u8 saturating add has two byte inputs plus one byte output; FP16 clamp has four binary16 arrays. Cache labels are therefore orientation points, not universal classifications.

For every kernel and size:

- three complete warmup samples run before measurement;
- 15 independent timing samples are collected;
- inner iterations target at least `1 << 20` processed elements per sample, with a cap of 4096 iterations;
- allocations and correctness validation remain outside timed regions;
- raw `ns/element` samples are preserved;
- the human and JSON summaries report minimum, median, 95th percentile, and median absolute deviation (MAD);
- effective `GiB/s` is derived from the median sample;
- runtime dispatch stays inside the timed public path when that is the implementation under test.

Median is the primary comparison statistic. Minimum is useful as a low-noise estimate of attainable throughput; p95 exposes slow-tail behavior; MAD describes robust spread without letting one scheduler interruption dominate the result.

## Correctness before timing

Squared error subtracts f32 inputs with f32 semantics, then widens the difference and accumulates products in f64. This makes the scalar implementation a useful numerical oracle at large sizes while keeping the input operation aligned with the SIMD variants.

Deterministic randomized differential tests cover:

- random f32 and u8 inputs;
- lengths on both sides of vector boundaries;
- every remainder modulo eight for F16C rejection;
- every remainder modulo 32 for byte SIMD tails;
- all 65,536 pairs of u8 inputs for the saturating-add contract;
- transactional F16C failure: an unsupported partial block must leave `dst` unchanged.

The fixed smoke datasets remain useful for readable failures, but are no longer the only correctness evidence.

## CPU and ISA tiers

`scripts/run_benchmarks.py --cpu` controls the autovectorization target for all three toolchains and records the selected tier in JSON:

| Tier | C++ | Rust | Zig |
|---|---|---|---|
| `baseline` | default compiler target | default compiler target | default build target |
| `native` | `-march=native` or `/arch:AVX2` | `-C target-cpu=native` | `-Dcpu=native` |
| `x86-64-v3` | `-march=x86-64-v3` or `/arch:AVX2` | `-C target-cpu=x86-64-v3` | `-Dcpu=x86_64_v3` |
| `avx2` | explicit AVX2/FMA/F16C flags | x86-64-v3 CPU tier | x86-64-v3 CPU tier |

Use `baseline` to compare portable scalar/autovec code against runtime-dispatched ISA implementations. Use `x86-64-v3` or `native` when the question is whether source-level autovectorization matches intrinsics at the same ISA level. Do not merge those tiers into one ranking.

C++ GCC and Clang use per-function ISA attributes for portable runtime dispatch. MSVC keeps target-specific code in a separate `/arch:AVX2` translation unit and checks CPUID plus OS YMM-state support in baseline code before entering it. AVX2-only byte kernels are not incorrectly gated on FMA. Every benchmark run emits both the floating-point dispatch tier and the u8 saturating-add tier, so Windows results cannot silently masquerade as SIMD results.

## Byte accounting

Reported bandwidth is effective algorithmic traffic:

- AXPY: two f32 reads plus one f32 write = 12 bytes per element;
- squared error: two f32 reads = 8 bytes per element;
- u8 SAD: two byte reads = 2 bytes per element;
- u8 saturating add: two byte reads plus one byte write = 3 bytes per element;
- FP16 clamp: three binary16 reads plus one binary16 write = 8 bytes per element.

AXPY's 12-byte figure is not necessarily physical memory-bus traffic. A normal cached store can trigger a read-for-ownership/write-allocate transaction for `dst`, making a cold streaming pass closer to 16 bytes per element before eviction writeback details. If `dst` is already resident, the extra read may not reach DRAM; non-temporal stores would change the model again. Treat the reported figure as effective bandwidth and use hardware counters when making physical-bandwidth claims.

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

The bounds are exactly `0.5 .. 2.0` (`0x3800 .. 0x4000`). These finite, exactly representable values isolate lowering and conversion cost. The separate correctness corpus in `data/fp16-edge-corpus.json` covers signed zero, subnormals, adjacent boundary values, infinities, negative finite values, and multiple NaN encodings.

Rust and C++ store binary16 as `u16`, widen eight lanes with F16C, clamp in f32, and narrow once. Zig compares native `@Vector(16, f16)` with a promote-once f32 path. These strategies are not assumed to have identical semantics for every half value; performance claims must name the numerical contract.

## Running benchmarks

The one-command path is:

```bash
just bench
SIMD_LAB_CPU=x86-64-v3 just bench results/local.json
```

The direct equivalent is:

```bash
python3 scripts/run_benchmarks.py --pretty --cpu baseline
python3 scripts/run_benchmarks.py --pretty --cpu x86-64-v3 --output results/local.json
```

Individual harnesses remain available:

```bash
(cd rust && cargo run --release --bin bench)
(cd zig && zig build bench -Doptimize=ReleaseFast)
cmake -S cpp -B build/cpp -DCMAKE_BUILD_TYPE=Release -DSIMD_LAB_CPU=baseline
cmake --build build/cpp -j
./build/cpp/simd_lab_cpp_bench
```

CI compiles every benchmark target but does not treat shared GitHub-hosted runner timings as performance evidence.

## Machine-readable collection

The collector emits `simd-lab-benchmark-v2`. Each result includes:

- implementation name and element count;
- working-set bytes and effective bytes per iteration;
- iterations per timing sample;
- all raw `ns/element` samples;
- minimum, median, p95, MAD, and median effective GiB/s;
- selected CPU tier and exact cross-toolchain target configuration;
- actual dispatch metadata reported by language harnesses;
- host and compiler versions;
- explicit skip records for unavailable ISA paths.

The collector remains outside the language binaries so benchmark timing code stays dependency-free and the JSON contract can evolve without three serialization implementations.
