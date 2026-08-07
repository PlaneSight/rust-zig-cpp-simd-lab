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

The second probe family targets sum of absolute differences over unsigned
samples. Both widths use an exact `u64` reduction over equal-length inputs and
finish arbitrary tails with the scalar reference operation:

- `u8`: Rust/C++ explicit AVX2 `_mm256_sad_epu8`; Zig
  `@Vector(32, u8)` absdiff widened to `u16` and reduced.
- `u16`: Rust/C++ AVX2 max/min/subtract over 16 lanes, widened to `u32` and
  accumulated into `u64`; Zig `@Vector(16, u16)` uses the same widened
  reduction. There is no u16 SAD instruction analogous to `VPSADBW`.

`probes/zig/sad.zig` exports scalar/vector u8 and u16 entry points plus fixed
lane absdiff probes. Rust and C++ expose scalar and target-gated AVX2 raw
pointer entry points for both widths. The length-last probe contract guards
zero-length inputs before pointer views or loads.

The key u8 codegen question is whether generic vector IR recognizes the SAD
idiom and selects `VPSADBW` (or an equally efficient sequence), versus
lowering it as separate min/max/subtract, zero-extension, and horizontal
reduction operations. For u16, inspect max/min/subtract, zero-extension,
reduction, and scalar-tail code rather than expecting a dedicated mnemonic.

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

## Saturating-subtract probes

The matching probe covers `u8`, `i8`, `u16`, `i16`, `u32`, `i32`, `u64`, and
`i64` with raw-pointer entry points and an explicit element count. Inputs must
have equal lengths; `len == 0` is valid, arbitrary tails are processed, unsigned
underflow clamps to `0`, and signed results clamp to `MIN`/`MAX`. Runtime
harnesses compare every type with independent scalar references. These probes
provide codegen evidence only, so runtime correctness and assembly evidence
remain separate.

Compiler evidence may include x86 `vpsubusb`/`vpsubsb` or other `vpsub*`
lowering, AArch64 `uqsub`/`sqsub`, and wasm `sub_sat` lane opcodes. These are
observations rather than required instruction sequences or thresholds.

## Stage 8 image-kernel probes

The Stage 8 probe family is tagged `image_kernels`. Raw-pointer
entry points with the element count last, matching the public slice
semantics:

```text
blend_u8_scalar(dst, a, b, weight, len)
convolve3_u8_scalar(dst, src, len)
convolve5_u8_scalar(dst, src, len)
```

Rust, Zig, and C++23 expose the three scalar/autovec forms. Zig additionally
exports `blend_u8_vector`, `convolve3_u8_vector`, and `convolve5_u8_vector`.
`weight` is an asserted `u16` in `0..=256`; the blend uses
`(u16(a) * (256 - weight) + u16(b) * weight + 128) >> 8`. The convolution
probes use clamp-to-edge borders, fixed `[1, 2, 1] / 4` and
`[1, 4, 6, 4, 1] / 16` filters, and apply `+2 >> 2` and `+8 >> 4`
rounding before the final u8 store.

Every probe handles `len == 0` without dereferencing, produces all outputs
for lengths `1..5` and arbitrary tails, and keeps the scalar border/tail
semantics identical to the public API. Rust probes must check `len == 0`
before constructing slices from raw pointers. These probes remain tiny
code-generation evidence: they do not allocate, perform I/O, measure
runtime, or claim an ISA-specific instruction sequence. They also do not
expand the contract to alpha blending or generic signed coefficients.



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

Saturating-add probes:

```bash
zig build-obj probes/zig/sat_add.zig -O ReleaseFast -femit-asm=zig-sat_add.s
rustc -O --crate-type=lib --emit=asm probes/rust/sat_add.rs -o rust-sat_add.s
clang++ -std=c++23 -O3 -S -masm=intel probes/cpp/sat_add.cpp -o cpp-clang-sat_add.s
g++ -std=c++23 -O3 -S -masm=intel probes/cpp/sat_add.cpp -o cpp-gcc-sat_add.s
```

Saturating-subtract probes:

```bash
zig build-obj probes/zig/sat_sub.zig -O ReleaseFast -femit-asm=zig-sat_sub.s
rustc -O --crate-type=lib --emit=asm probes/rust/sat_sub.rs -o rust-sat_sub.s
clang++ -std=c++23 -O3 -S -masm=intel probes/cpp/sat_sub.cpp -o cpp-clang-sat_sub.s
g++ -std=c++23 -O3 -S -masm=intel probes/cpp/sat_sub.cpp -o cpp-gcc-sat_sub.s
```

Generate the Stage 7 and Stage 8 probes with the same target-profile pipeline:

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

The analyzer excludes comments, directives, labels, and symbol aliases, then
records total instruction-looking lines, vector-looking instructions, top
mnemonics, and selected SIMD families including FP16 conversions. These are
regression signals, not a performance model.

For SAD, also inspect specifically for `vpsadbw` / `psadbw`, widening instructions, lane-crossing shuffles, and scalar extraction around reductions.

For deeper analysis, keep evidence classes separate:

- `simd-lab-benchmark-v2` wall-clock `ns/element` and bandwidth are runtime
  measurements from the benchmark protocol;
- Linux `perf stat` is observed PMU evidence from the separate collector, with
  process aggregation unless a dedicated workload gives it an attribution
  boundary;
- `llvm-mca` throughput, latency, and resource-pressure values are static
  model estimates, not hardware/runtime evidence;
- generated IR and this analyzer's instruction counts remain codegen evidence.

Do not turn a process aggregate into per-row cycles/element, and do not use a
static estimate alone to justify inline assembly.

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

## Stage 9 evidence adapters

Both adapters use the shared result-bundle arguments (`--id --family --kernel
--implementation --target --cpu` plus the optional metadata documented in
`methodology.md`) and `scripts/result_bundle.py`. The perf surface is:

```text
python3 scripts/collect_perf_stat.py [shared arguments] \
  --perf perf --event cycles --event instructions --event branches \
  --event branch-misses --repeat 5 \
  --scope {process-aggregate,dedicated-workload} --raw-output RAW -- COMMAND [ARGS...]
```

The collector is Linux-only and invokes
`LC_ALL=C perf stat --json-output --no-big-num --output RAW --repeat N
--event cycles,instructions,branches,branch-misses -- COMMAND`. It emits a v1
`counters` observation with numeric metrics, stores the raw output as an
artifact, and keeps the exact command, scope, events, and repeats in
`experiment.parameters`. `<not supported>` and `<not counted>` are errors,
never zero; permission, event, missing-tool, non-Linux, and workload failures
must also fail clearly.

The llvm-mca surface is:

```text
python3 scripts/analyze_mca.py [shared arguments] \
  --llvm-mca llvm-mca --iterations 100 --mattr FEATURE ... \
  [--start-label START --end-label END] [--region REGION] \
  --raw-output RAW ASSEMBLY
```

The input must already contain matched, non-nested LLVM-MCA marker regions or
be explicitly extracted with a paired start/end-label region; an unbounded
multi-function file is never silently analyzed. `--target` and `--cpu` drive
explicit `-mtriple` and `-mcpu`, while `--mattr` and `--iterations` are
provenance. The parser selects one JSON `CodeRegions` entry and requires
`SummaryView` and `ResourcePressureView`. The bundle contains a v1 `analysis`
summary/evidence reference and an artifact for the raw JSON, with model numbers
left in that JSON rather than counter/runtime fields. Record the exact
llvm-mca version, command, triple, CPU, features, and iterations. No
authoritative mca baseline is claimed.

See the [Linux `perf stat` manual](https://man7.org/linux/man-pages/man1/perf-stat.1.html)
and the [LLVM 22.1 `llvm-mca` command guide](https://releases.llvm.org/22.1.0/docs/CommandGuide/llvm-mca.html).
