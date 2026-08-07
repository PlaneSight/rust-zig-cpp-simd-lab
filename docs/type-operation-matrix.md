# Integer and floating-point SIMD investigation matrix

This lab should answer a narrower and more useful question than "which language has faster SIMD?":

> How well do Rust, Zig 0.16, and C++23 express and lower common integer and floating-point SIMD idioms across realistic CPU targets?

The investigation is motivated in part by Zig issue `ziglang/zig#19550`, where a compact `f16` vector clamp produced pathological code generation on targets without native FP16 arithmetic, and by `adworacz/zsmooth#29`, which tracks practical F16 support and optimization in video filters.

## Types

### Integer

- `u8`, `i8`
- `u16`, `i16`
- `u32`, `i32`
- `u64`, `i64` where the operation is representative

### Floating point

- `f16`
- `f32`
- `f64`

### Mixed-width conversions

- `u8 -> u16 -> u8`
- `u16 -> u32 -> u16`
- `i16 -> i32 -> i16`
- `f16 -> f32 -> f16`

## Common kernels

Every kernel starts with a scalar reference and should be implemented with equivalent semantics in each language.

| Family | Kernels / idioms | Why it matters |
|---|---|---|
| Basic arithmetic | add, subtract, multiply | baseline instruction selection and vector width |
| Min/max | min, max, clamp | common image-processing primitive; exposes FP16 lowering |
| Comparisons | compare + select | masks, blends, branch-free conditionals |
| Saturation | saturating add/subtract, clamp-to-range | common integer pixel arithmetic |
| Absolute difference | absdiff, SAD | motion/search/image metrics |
| Reductions | sum, min, max | horizontal operations and accumulator strategy |
| Dot products | integer and float dot | widening, multiply-add, specialized ISA instructions |
| Widening | widen + arithmetic | pixel and coefficient processing |
| Narrowing | pack / saturating pack | output conversion and storage |
| Conversion | integer/float and f16/f32 | conversion throughput and hoisting |
| FMA | `a*b+c` | contraction policy and ISA use |
| Blend | lerp / weighted blend | representative video/image arithmetic |
| Squared error | `(a-b)^2` reduction | MSE/PSNR-style workload |
| Convolution | short 3/5/7/8-tap kernels | realistic multiply-accumulate pressure |

## Dot-product and widening-multiply family

The Stage 7 integer family keeps products in their mathematical widened
storage type before writing results:

| Operation | Output contract | Reduction contract |
|---|---|---|
| `u8 * u8 -> u16` | one widened product per lane | none |
| `i8 * i8 -> i16` | one widened product per lane | none |
| `u16 * u16 -> u32` | one widened product per lane | none |
| `i16 * i16 -> i32` | one widened product per lane | none |
| `f32` dot | none | cast each input to `f64`, multiply, sum in `f64` |
| `f64` dot | none | multiply and sum in `f64` |
| `i16 * i16` dot | none | widen each product and sum in `i64` |
| `u8 * i8` dot | none | preserve signedness and sum products in `i64` |

Every implementation asserts equal input lengths, processes complete native
vector chunks where available, and finishes with a scalar tail. Integer dot
results are exact when the mathematical sum fits `i64`; the benchmark inputs
stay within that contract. No operation allocates in its hot loop. The Rust
and C++ scalar loops are autovectorization baselines; Zig also exposes
explicit native-vector forms. The current Stage 7 implementation intentionally
does not claim dedicated VNNI, dot-product, or widening multiply-accumulate
instructions.

## Implementation layers

Where the language/toolchain permits it, compare:

1. scalar source;
2. compiler autovectorization;
3. portable/native vector abstraction;
4. architecture intrinsics;
5. inline assembly reference, only when it demonstrates a real code-generation limitation.

The inline-assembly variant is not automatically treated as the ideal result. It exists to establish what the ISA can do when the compiler does not find the sequence itself.

## FP16 investigation

FP16 gets a dedicated matrix because its ideal lowering depends strongly on target ISA and numerical semantics.

Test at least these strategies:

1. native `f16` source arithmetic;
2. relaxed/optimized floating-point mode where available;
3. explicit `f16 -> f32` promotion once at the kernel boundary, compute entirely in `f32`, then narrow once;
4. deliberately conversion-heavy `f16 <-> f32` code as a negative control;
5. explicit F16C conversion plus f32 arithmetic on x86 targets without native FP16 arithmetic;
6. native AVX-512 FP16 where supported.

### Conversion-hoisting hypothesis

A central experiment is whether the compiler can transform a chain conceptually like:

```text
f16 -> f32 -> operation -> f16
f16 -> f32 -> operation -> f16
f16 -> f32 -> operation -> f16
```

into:

```text
load f16
  -> convert once to f32
  -> perform the whole kernel in f32
  -> convert once to f16
  -> store
```

For each generated function, count at minimum:

- `vcvtph2ps` / equivalent widening conversions;
- `vcvtps2ph` / equivalent narrowing conversions;
- total instructions;
- vector-width changes;
- spills and reloads.

## CPU target matrix

Initial x86-64 targets should distinguish capabilities rather than using only `-march=native`:

- baseline x86-64;
- AVX2;
- AVX2 + FMA + F16C;
- AVX-512F where useful;
- AVX-512 FP16 when supported.

Later targets:

- AArch64 + NEON;
- AArch64 FP16-capable targets;
- wasm32 + simd128.

Results from different target feature sets must never be collapsed into one "language" result.

## What to measure

### Runtime

- ns / element;
- cycles / element;
- effective GB/s;
- throughput over multiple working-set sizes;
- tail-handling cost;
- alignment sensitivity where relevant.

### Code generation

- instruction count;
- code size;
- vector width;
- conversion instruction count;
- loads / stores;
- spills;
- branches;
- relevant uop counts where tooling permits.

### Compiler evidence

Record:

- compiler and version;
- optimization mode;
- target triple;
- CPU / ISA features;
- floating-point mode / flags;
- vectorization diagnostics where available;
- generated assembly.

## Numerical-semantics rule

Performance results are only comparable when semantics are comparable.

In particular, promoting an `f16` computation to `f32` can change rounding behavior. A promoted implementation must therefore be labeled separately from strict per-operation `f16` semantics. Relaxed floating-point modes must also be recorded explicitly.

The lab should measure both when useful rather than pretending the distinction does not exist.

## Cross-language equivalence

For each test case, keep the source structure as close as practical while respecting idiomatic language constructs. If apparently equivalent Rust, Zig, and C++ source lowers differently, inspect IR/assembly before attributing the result to the language itself.

The useful output is not just a winner. It is a catalogue of:

- optimizations each compiler reliably discovers;
- source patterns that inhibit SIMD;
- type-specific codegen pathologies;
- cases where manual widening/narrowing wins;
- cases where intrinsics beat portable vector code;
- cases where inline assembly proves a missed compiler opportunity.
