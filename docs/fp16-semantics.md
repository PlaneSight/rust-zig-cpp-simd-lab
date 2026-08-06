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
Rust    rust/src/bin/fp16_semantics.rs
C++23   cpp/src/fp16_semantics.cpp
Zig     zig/src/fp16_semantics.zig
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
