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

## Phase 2: semantic edge cases

These must not be folded into performance datasets until behavior is documented for every implementation:

- `+0` and `-0`
- positive and negative infinity
- positive and negative finite values
- smallest/largest subnormals
- smallest normal
- largest finite value
- values adjacent to half rounding boundaries
- quiet NaNs with multiple payloads
- signaling NaNs where the language/compiler/hardware contract makes testing meaningful

For each case record:

1. output bit pattern;
2. exception/flag behavior if observable;
3. whether NaN payload/sign is preserved;
4. whether signed zero is preserved;
5. whether native f16 and promoted-f32 semantics differ;
6. whether differences are permitted by the selected fast/optimized float mode.

## Rule

A promoted-f32 implementation is not labelled a drop-in native-f16 replacement merely because it is faster. The repository reports it as a distinct implementation strategy and documents any semantic divergence alongside performance results.
