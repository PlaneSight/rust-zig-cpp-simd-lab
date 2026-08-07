# FP16 numerical semantics

Performance results are only comparable when the numerical contract is explicit.

The first runtime benchmark intentionally uses finite, exactly representable half values and therefore expects bit-identical output from:

- Zig native `f16`
- Zig promote-once `f16 -> f32 -> f16`
- Rust F16C boundary conversion
- C++23 F16C boundary conversion

## Phase 1: exact finite values

Shared input bit patterns:

```text
0000  0.0
3400  0.25
3800  0.5
3c00  1.0
3e00  1.5
4000  2.0
4200  3.0
4400  4.0
```

Bounds: `0x3800 .. 0x4000` (`0.5 .. 2.0`).

For positive finite half values the integer encoding is monotonic, so the Rust/C++ harnesses can validate this initial clamp directly in storage representation. Zig validates native and promoted results by comparing the resulting half bit patterns.

## Phase 2: executable semantic edge corpus

`data/fp16-edge-corpus.json` defines the correctness-only corpus. It includes:

- `+0` and `-0`
- positive and negative infinity
- positive and negative finite values
- smallest/largest positive subnormals
- smallest positive normal
- values immediately around `0.5` and `1.0`
- largest finite positive/negative values
- multiple quiet NaN encodings

The semantic clamp uses `-0.5 .. +2.0` bounds so negative values and signed zero are exercised as well as the positive range used by the performance corpus.

Each implementation has a tiny semantic dump executable:

```text
Rust    languages/rust/src/bin/fp16_semantics.rs
C++23   languages/cpp/src/fp16_semantics.cpp
Zig     languages/zig/src/fp16_semantics.zig
```

They output the resulting binary16 bit patterns as JSON lines. This is deliberately lower-level than comparing floating-point values: signed zero, NaN payload changes, and quieting/canonicalization remain visible.

Run the cross-language comparison with:

```bash
python scripts/compare_fp16_semantics.py
```

The resulting `simd-lab-fp16-semantics-v1` JSON contains every available strategy, the output bits for each corpus index, an `all_equal` flag per case, and a total divergence count.

A divergence is a **measurement**, not automatically a failure. Native f16 and widen-to-f32 strategies can legitimately have different contracts. The report exists to make those differences explicit before performance claims are made.

## Interpretation checklist

For each divergent case determine:

1. output bit pattern;
2. whether the input/result is zero, finite, infinity, or NaN;
3. whether NaN payload/sign is preserved, changed, or canonicalized;
4. whether signed zero is preserved;
5. whether native f16 and promoted-f32 semantics differ;
6. whether the difference comes from min/max semantics, conversion semantics, or selected float mode;
7. whether the difference is allowed by the documented implementation contract.

Exception/flag behavior is intentionally a later layer because it needs platform-specific floating-point environment handling and should not be inferred from value bits alone.

## Rule

A promoted-f32 implementation is not labelled a drop-in native-f16 replacement merely because it is faster. The repository reports it as a distinct implementation strategy and documents any semantic divergence alongside performance results.

## Current evidence boundary

The checked-in corpus contains 23 cases. The Rust and C++ F16C entry points
require complete eight-element blocks, so their semantic dump executables use
an internal 24-element padded scratch buffer and emit only the 23 corpus
results. The padding is not part of the reported contract.

The local comparison was run on Darwin arm64 (Apple M5). Rust F16C and C++ F16C
reported `available: false` because the host has no x86 F16C path; those rows
were excluded rather than treated as equal. Zig native-f16 and promote-f32
both returned 23 outputs with zero divergence under ReleaseSafe and
ReleaseFast. This is ARM semantic evidence only, not F16C evidence.

The reviewed x86-64-v3 codegen artifacts show a conversion-hoisting
difference: the Zig native clamp has 384 `vcvtph2ps` and 256 `vcvtps2ph`
operations in its bounded symbol, while the promote-once path has 18 and 2.
These are compile/codegen counts from a reviewed artifact, not runtime
measurements and not proof that one numerical contract is preferable.

The hosted x86 F16C run now executes the Rust/C++ edge corpus and compares
signed zero, subnormal, infinity, NaN payload/sign, and rounding behavior.
The existing hosted x86 codegen job by itself does not run the semantic
executables; the runtime evidence is recorded in the dedicated run below.

### Hosted x86 F16C run

An isolated GitHub Actions run on `ubuntu-latest` executed the current semantic
comparison on a real x86 runner. The runner's `/proc/cpuinfo` reported both
`f16c` and `avx2`; the run is
`https://github.com/PlaneSight/rust-zig-cpp-simd-lab/actions/runs/31167176176`.
Rust F16C, C++ F16C, Zig native-f16, and Zig promote-f32 were all available,
each emitted 23 results, and the comparison reported zero divergences.

The four strategies preserved `+0` (`0000`) and `-0` (`8000`), retained the
positive subnormal boundaries (`0001`, `03ff`, `0400`) and the negative
subnormal (`8001`), and agreed at the values immediately around `0.5` and
`1.0`. The clamp contract mapped positive and negative finite extrema,
infinities, and quiet NaN payload/sign cases to the documented `-0.5`/`+2.0`
bounds where applicable. The hosted output therefore directly confirms the
documented signed-zero, subnormal, infinity, NaN, and boundary-rounding behavior
for this 23-case clamp corpus; it does not generalize to arbitrary FP16
arithmetic or floating-point exception flags.

This run used the optimized Zig path (`ReleaseFast`). The ARM64
`ReleaseSafe`/`ReleaseFast` comparison remains the direct fast-mode comparison:
both modes produced the same 23 bits. The reviewed x86 conversion-hoisting
counts remain compile/codegen evidence rather than runtime throughput claims.

### ARM64 edge-observation matrix

The current Apple M5 run compared Zig `ReleaseSafe` and `ReleaseFast`; both
strategies emitted the same 23 output bits:

| Semantic class | Corpus inputs | Observed output behavior |
|---|---|---|
| Signed zero | `0000`, `8000` | Both signs were preserved (`0000`, `8000`). |
| Positive subnormal/normal boundary | `0001`, `03ff`, `0400` | Values remained unchanged. |
| Negative subnormal | `8001` | The value remained unchanged. |
| Finite values around `0.5` and `1.0` | `37ff`, `3800`, `3801`, `3bff`, `3c00`, `3c01` | All remained at their input half bit patterns. |
| Positive/negative infinity | `7c00`, `fc00` | Positive infinity clamped to `4000`; negative infinity clamped to `b800`. |
| Quiet NaN payload/sign cases | `7e01`, `7fff`, `fe01` | All selected the upper bound `4000`; payload and sign were not preserved by this clamp policy. |
| Rounding-sensitive boundary | Values immediately around `0.5` and `1.0` | No rounding difference was observed; these inputs and outputs are exactly representable. |

This matrix documents the native/promoted ARM64 result only. It is not a
substitute for executing the Rust/C++ F16C paths on the user's x86 host, and it
does not generalize NaN or rounding behavior to arbitrary FP16 arithmetic.