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
| vector `f16 -> f32 -> f16` | F16C runtime kernel | `f16x16` promote-once | F16C runtime kernel |

This table intentionally separates language-standard baselines from architecture/compiler extensions. Adding an extension is useful, but it must be labeled so it is not mistaken for a portable language feature.

## Zig FP16 experiment

`probes/zig/clamp.zig` contains two semantically distinct FP16 implementations:

- `clamp_f16x16_native`: preserve f16 operation semantics throughout.
- `clamp_f16x16_promote_once`: widen all inputs once, perform the complete operation in f32, then narrow once at the output boundary.

The second form is an application-level numerical tradeoff and must not be presented as bit-identical to native f16 arithmetic in the general case.

The primary codegen question on x86 systems without native FP16 arithmetic is whether the native path repeatedly emits `vcvtph2ps` / `vcvtps2ph`, and whether explicit boundary promotion collapses those conversions to a small fixed number.

## SAD / widening family

The second probe family targets a very common image/video primitive: sum of absolute differences over `u8` samples.

- Rust: scalar iterator source plus explicit AVX2 `_mm256_sad_epu8`.
- C++23: scalar loop plus explicit AVX2 `_mm256_sad_epu8`.
- Zig: `@Vector(32, u8)` absdiff, explicit widen to `@Vector(32, u16)`, then `@reduce(.Add, ...)`.

The key codegen question is whether generic vector IR recognizes the SAD idiom and selects `VPSADBW` (or an equally efficient sequence), versus lowering it as separate min/max/subtract, zero-extension, and horizontal reduction operations.

`probes/zig/sad.zig` also contains standalone absdiff and widening probes so missed idiom recognition can be separated from widening/reduction cost.

## Dot-product and widening-multiply probes

Stage 7 adds standalone `dot` and `widen_mul` probes for Rust, Zig, and
C++23. Each exports the scalar/autovec operation with raw pointer boundaries;
the Zig probe additionally exports its explicit native-vector forms. Products
are widened before multiplication, including the 32-bit input forms
`u32 * u32 -> u64` and `i32 * i32 -> i64`, and dot reductions use the documented
`f64` or `i64` accumulators. These probes are codegen evidence only: runtime
correctness is exercised by the language harnesses, not by the probe source.

## Mixed-width conversion probes

`probes/rust/mixed_width.rs`, `probes/zig/mixed_width.zig`, and
`probes/cpp/mixed_width.cpp` export the same scalar raw-pointer families:
integer widening, integer-to-f32 conversion, u8 affine conversion, explicit
f32/u16 and f32/u8 narrowing policies, u16/u8 narrowing policies, and
little-endian u8x4/u32 packing. Zig also exports native vector widening and
integer-to-f32 forms with scalar tails. Length is last for all entry points;
packing and unpacking use a group count rather than a byte count.

Inspect the generated output for conversion instructions, widening moves,
vector integer-to-float lowering, and whether scalar tails remain bounded.
The probe set deliberately does not claim dedicated dot-product or narrowing
instructions; those require separate ISA-specific evidence.

## Saturating-add probes

The saturation probe exports the scalar/autovec operation for each fixed-width
type: `u8`, `i8`, `u16`, `i16`, `u32`, `i32`, `u64`, and `i64`. Rust, Zig, and
C++ use raw-pointer entry points with an explicit element count. The runtime
harnesses, rather than these tiny probe functions, own independent
correctness checks for extrema, cancellation, zero-length inputs, and scalar
tails. Widened or checked scalar implementations are compared by semantics;
the generated assembly is evidence of the compiler's lowering, not an exact
instruction fixture.


## Generate assembly

Clamp probes:

```bash
zig build-obj probes/zig/clamp.zig -O ReleaseFast -femit-asm=zig-clamp.s
rustc -O --crate-type=lib --emit=asm probes/rust/clamp.rs -o rust-clamp.s
clang++ -std=c++23 -O3 -S -masm=intel probes/cpp/clamp.cpp -o cpp-clang-clamp.s
g++ -std=c++23 -O3 -S -masm=intel probes/cpp/clamp.cpp -o cpp-gcc-clamp.s
```

SAD probes:

```bash
zig build-obj probes/zig/sad.zig -O ReleaseFast -femit-asm=zig-sad.s
rustc -O --crate-type=lib --emit=asm probes/rust/sad.rs -o rust-sad.s
clang++ -std=c++23 -O3 -S -masm=intel probes/cpp/sad.cpp -o cpp-clang-sad.s
g++ -std=c++23 -O3 -S -masm=intel probes/cpp/sad.cpp -o cpp-gcc-sad.s
```

Generate the Stage 7 probes with the same target-profile pipeline:

```bash
python3 scripts/generate_codegen_snapshots.py --target x86-64-v3
python3 scripts/generate_cross_codegen.py --target aarch64-neon
python3 scripts/generate_cross_codegen.py --target wasm-simd128
```

For target-specific experiments, record the complete target flags next to the result. Important x86 cases include baseline x86-64, AVX2, AVX2+F16C+FMA, AVX-512, and AVX-512 FP16.

## Extract basic metrics

```bash
python scripts/analyze_asm.py --pretty zig-clamp.s rust-clamp.s cpp-clang-clamp.s
python scripts/analyze_asm.py --pretty zig-sad.s rust-sad.s cpp-clang-sad.s
```

The analyzer records total instruction-looking lines, vector-looking instructions, top mnemonics, and selected SIMD families including FP16 conversions. These are regression signals, not a performance model.

For SAD, also inspect specifically for `vpsadbw` / `psadbw`, widening instructions, lane-crossing shuffles, and scalar extraction around reductions.

For deeper analysis, pair codegen metrics with:

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
