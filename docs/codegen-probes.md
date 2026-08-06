# Codegen probes

The files under `probes/` are deliberately tiny compiler probes. They are not benchmark harnesses and should not allocate, perform I/O, or hide the interesting operation behind a large abstraction boundary.

## Clamp family

The initial family mirrors the motivating case from `ziglang/zig#19550`: compute the minimum and maximum of eight values and clamp a ninth value to that range.

The first comparison set is:

| Probe | Rust | Zig | C++23 |
|---|---|---|---|
| scalar `u8` | yes | vector `u8x32` | yes |
| scalar `u16` | yes | vector `u16x16` | yes |
| scalar `f32` | yes | vector `f32x8` | yes |
| scalar `f64` | yes | planned | yes |
| vector `f16` native | N/A in stable portable SIMD baseline | `f16x16` | non-standard compiler extensions intentionally excluded from C++23 baseline |
| vector `f16 -> f32 -> f16` | planned with an explicit half representation | `f16x16` promote-once | planned ISA-specific F16C probe |

This table intentionally separates language-standard baselines from architecture/compiler extensions. Adding an extension is useful, but it must be labeled so it is not mistaken for a portable language feature.

## Zig FP16 experiment

`probes/zig/clamp.zig` contains two semantically distinct FP16 implementations:

- `clamp_f16x16_native`: preserve f16 operation semantics throughout.
- `clamp_f16x16_promote_once`: widen all inputs once, perform the complete operation in f32, then narrow once at the output boundary.

The second form is an application-level numerical tradeoff and must not be presented as bit-identical to native f16 arithmetic in the general case.

The primary codegen question on x86 systems without native FP16 arithmetic is whether the native path repeatedly emits `vcvtph2ps` / `vcvtps2ph`, and whether explicit boundary promotion collapses those conversions to a small fixed number.

## Generate assembly

Zig:

```bash
zig build-obj probes/zig/clamp.zig -O ReleaseFast -femit-asm=zig-clamp.s
```

Rust:

```bash
rustc -O --crate-type=lib --emit=asm probes/rust/clamp.rs -o rust-clamp.s
```

Clang C++23:

```bash
clang++ -std=c++23 -O3 -S -masm=intel probes/cpp/clamp.cpp -o cpp-clang-clamp.s
```

GCC C++23:

```bash
g++ -std=c++23 -O3 -S -masm=intel probes/cpp/clamp.cpp -o cpp-gcc-clamp.s
```

For target-specific experiments, record the complete target flags next to the result. Important x86 cases include baseline x86-64, AVX2, AVX2+F16C+FMA, AVX-512, and AVX-512 FP16.

## Extract basic metrics

```bash
python scripts/analyze_asm.py --pretty zig-clamp.s rust-clamp.s cpp-clang-clamp.s
```

The analyzer records total instruction-looking lines, vector-looking instructions, top mnemonics, and selected SIMD families including FP16 conversions. These are regression signals, not a performance model.

For deeper analysis, pair them with:

- runtime ns/element and cycles/element;
- `perf stat` retired instructions and hardware counters;
- `llvm-mca` or uiCA throughput/port-pressure estimates where applicable;
- generated IR when two languages feed LLVM but produce different machine code.

## Result discipline

Every recorded result should include:

1. source revision;
2. compiler and exact version;
3. optimization mode;
4. target triple / CPU / feature flags;
5. relevant floating-point mode;
6. instruction and conversion counts;
7. runtime measurement when available;
8. whether numerical semantics differ from the reference.

The purpose is to distinguish language limitations, frontend lowering decisions, optimizer misses, backend limitations, and deliberate numerical tradeoffs rather than collapsing all of them into "language X is faster".
