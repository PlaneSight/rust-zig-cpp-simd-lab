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

## Stage 8: `u8` blend and short horizontal filters

Stage 8 uses one shared integer contract in Rust, Zig 0.16, and C++23. Each
public slice API requires equal logical lengths, accepts length zero, and writes
every output for every non-zero length. All transforms are out-of-place:
destinations must not overlap inputs; Rust enforces this through borrowing,
while Zig and C++ make it a caller precondition. The arithmetic is unsigned
and widened explicitly; no signed intermediate or overflowing narrow
accumulator is part of the contract.

### Mathematical contract

For `n > 0`, `blend_u8_scalar(dst, a, b, weight)` computes, for every
`0 <= i < n`:

```text
dst[i] = (u16(a[i]) * (256 - weight) + u16(b[i]) * weight + 128) >> 8
```

`weight` is a `u16` value asserted to be in `0..=256`. The displayed `u16`
casts define the lane widening; the products and their sum must be held in a
sufficiently wide unsigned accumulator (`u32` is sufficient) before the
right shift and final `u8` store. The `+128` term gives integer
round-to-nearest with upward ties for the denominator 256. The result is
already in the `u8` range for the permitted inputs; this is not an alpha-blend
or generic signed-coefficient operation.

For a non-empty source, define the clamp-to-edge index
`edge(j, n) = min(max(j, 0), n - 1)`. The 3-tap operation is:

```text
convolve3_u8_scalar(dst, src):
dst[i] = (u32(src[edge(i - 1, n)])
        + 2 * u32(src[edge(i, n)])
        + u32(src[edge(i + 1, n)]) + 2) >> 2
```

The 5-tap operation is:

```text
convolve5_u8_scalar(dst, src):
dst[i] = (u32(src[edge(i - 2, n)])
        + 4 * u32(src[edge(i - 1, n)])
        + 6 * u32(src[edge(i, n)])
        + 4 * u32(src[edge(i + 1, n)])
        + u32(src[edge(i + 2, n)]) + 8) >> 4
```

The fixed coefficient vectors are `[1, 2, 1] / 4` and `[1, 4, 6, 4, 1] /
16`. `+2` and `+8` are applied before the shifts, respectively, so both
filters use non-negative integer round-to-nearest with upward ties. The
clamp-to-edge rule writes the first and last pixels as well as the interior;
for `n = 1..4` repeated edge samples are intentional.

### Length, traffic, and implementation tiers

All three operations require equal-length input and output slices. Length zero
returns without dereferencing a pointer. Lengths `1..5`, vector-boundary
lengths, and arbitrary remainders are valid; no element may be skipped,
over-read, or left unwritten. Convolution borders use the same formula as the
interior, with the index clamp above, and scalar tails use exactly the same
rounding and narrowing semantics as vectorized chunks.

The fixed filters use explicit unsigned widening for their fixed taps. This
scope does not introduce a generic signed-coefficient API or claim arbitrary
filter-kernel support.

| Operation | Rust | Zig 0.16 | C++23 | Effective traffic |
|---|---|---|---|---:|
| weighted blend / lerp | scalar/autovec | scalar/autovec and `@Vector` | scalar/autovec | `3*n` bytes |
| 3-tap horizontal convolution | scalar/autovec | scalar/autovec and `@Vector` | scalar/autovec | `2*n` bytes |
| 5-tap horizontal convolution | scalar/autovec | scalar/autovec and `@Vector` | scalar/autovec | `2*n` bytes |

Blend traffic is two u8 reads plus one u8 write. Convolution traffic is one
u8 read plus one u8 write; the fixed coefficient constants are excluded from
the traffic and working-set figures. The Rust and C++ rows are ordinary
scalar-source/autovectorization baselines. Zig's explicit vector rows retain
scalar border and tail handling. No ISA-specific implementation tier is
claimed by Stage 8.

## Saturating-add family

The fixed-width saturation family computes each output from the mathematical
sum of two equal-length input streams:

| Input type | Output type | Boundary behavior |
|---|---|---|
| `u8`, `u16`, `u32`, `u64` | same unsigned type | clamp sums above `MAX` to `MAX` |
| `i8`, `i16`, `i32`, `i64` | same signed type | clamp sums below `MIN` or above `MAX` |

Rust and Zig use their defined saturating-add operations in the scalar
baseline. C++ widens operands where the next integer width is sufficient and
uses checked boundary comparisons for `u64` and `i64`, so no signed
intermediate can overflow. All three languages assert equal buffer lengths,
leave zero-length inputs valid, and process arbitrary lengths without a
separate tail contract. Runtime rows use two input streams plus one output
stream, with traffic of `3 * sizeof(T)` bytes per element.

## Saturating-subtract family

The fixed-width saturating-subtract family computes each output from the
mathematical difference of two equal-length input streams:

| Input type | Output type | Boundary behavior |
|---|---|---|
| `u8`, `u16`, `u32`, `u64` | same unsigned type | underflow clamps to `0` |
| `i8`, `i16`, `i32`, `i64` | same signed type | clamp below `MIN` to `MIN` and above `MAX` to `MAX` |

All languages assert equal lengths, accept empty inputs, and process arbitrary
lengths with scalar tails. Runtime correctness uses independent scalar
references for every fixed-width type, including extrema, cancellation, zero
length, and tail cases. Runtime rows use two input streams plus one output
stream, with traffic of `3 * sizeof(T)` bytes per element. Runtime results and
codegen snapshots are separate evidence; assembly shows compiler lowering and
does not replace the reference checks.


## Dot-product and widening-multiply family

The integer widening family keeps products in their mathematical widened
storage type before writing results:

| Operation | Output contract | Reduction contract |
|---|---|---|
| `u8 * u8 -> u16` | one widened product per lane | none |
| `i8 * i8 -> i16` | one widened product per lane | none |
| `u16 * u16 -> u32` | one widened product per lane | none |
| `i16 * i16 -> i32` | one widened product per lane | none |
| `u32 * u32 -> u64` | one widened product per lane | none |
| `i32 * i32 -> i64` | one widened product per lane | none |
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

## Mixed-width conversion family

The mixed-width APIs keep widening explicit and make every narrowing policy
observable:

| Operation | Output contract |
|---|---|
| `u8 -> u16`, `u8 -> u32` | zero-extend each lane |
| `i8 -> i16`, `i16 -> i32` | sign-extend each lane |
| `u16 -> u32` | zero-extend each lane |
| `u16 -> f32`, `i16 -> f32` | exactly represent each integer in `f32` |
| `u8 -> f32` affine | compute `f32(u8) * scale + bias` |
| `f32 -> u16` saturation | NaN/non-positive -> `0`; values at or above `65535` -> `65535`; otherwise truncate |
| `f32 -> u8` truncation/rounding | finite `[0, 255]` precondition; truncate or round `floor(x + 0.5)` |
| `f32 -> u8` saturation | NaN/non-positive -> `0`; values at or above `255` -> `255`; otherwise truncate |
| `u16 -> u8` truncation | retain the low eight bits |
| `u16 -> u8` rounding | full-range integer round-to-nearest: `(u16 + 128) / 257` |
| `u16 -> u8` saturation | clamp values above `255` to `255` |
| `u8x4 <-> u32` | logical little-endian packing and unpacking |

All operations require equal logical lengths, accept zero-length buffers, and
process arbitrary tails without reading or writing outside the supplied
buffers. Packing uses a group count: four source bytes map to one `u32`.
Rust and C++ expose scalar-source autovectorization baselines. Zig additionally
exposes native `@Vector` widening and integer-to-float forms with scalar tails.
The standalone probes use raw pointers, explicit length-last entry points, and
the same arithmetic contracts; they are code-generation evidence, not runtime
benchmarks.


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
