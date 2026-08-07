# Case study: generic `u8` SAD versus `VPSADBW`

This study compares the same unsigned-byte sum-of-absolute-differences contract
across scalar source, compiler autovectorization, native/vector source, and an
explicit x86 AVX2 implementation.

## Contract

For equal-length inputs `a` and `b`, the result is:

```text
SAD(a, b) = sum(i = 0 .. n) abs(int(a[i]) - int(b[i]))
```

The result is accumulated in `u64`. Every implementation accepts `n == 0`,
processes complete vector blocks where applicable, and computes the remaining
scalar tail with the same unsigned absolute-difference operation. Inputs must
have equal lengths. The Rust and C++ out-of-place APIs use slice/span contracts;
the raw-pointer probes use a length-last entry point and are only called with
valid buffers.

## Implementation tiers

| Language | Scalar/autovec | Native/vector source | Explicit x86 path |
|---|---|---|---|
| Rust | `sad_u8_scalar` | scalar source used by `sad_u8_best` fallback | `sad_u8_avx2`, selected by AVX2 runtime dispatch |
| C++23 | `sad_u8_scalar` | scalar source used by `sad_u8_best` fallback | `sad_u8_avx2`, selected by GCC/Clang or MSVC dispatch |
| Zig 0.16 | `sadU8Scalar` | `sadU8Vector`, 32 `u8` lanes with widening reduction | native vector lowering, not an x86-only API |

The Rust and C++ AVX2 paths load 32 bytes, use the hardware byte SAD
operation, accumulate partial sums in 64-bit lanes, horizontally reduce them,
and finish with a scalar tail. Zig's vector source widens the 32 absolute
differences to `u16` before reduction.

## Reviewed x86 codegen

The reviewed manifest is
`results/codegen/known-good/manifest-x86-64-v3.json`. It is a compile/codegen
baseline from Linux x86-64-v3, not runtime performance evidence. Symbol-local
inspection of the associated assembly found:

| Compiler/path | `VPSADBW` evidence | Interpretation |
|---|---:|---|
| Rust AVX2 | 1 | One compact instruction in the vector body, followed by reduction and tail handling. |
| Clang AVX2 | 9 | Eight steady-state unrolled operations plus one residual operation; not nine different algorithms. |
| GCC AVX2 | 1 | Compact byte-SAD lowering. |
| Zig `sad_u8x32` | 1 | Native vector absdiff/widen/reduce lowers to the same byte-SAD idiom. |

The raw reviewed assembly is under
`results/artifacts/2026-08-07-codegen-x86-64-v3/`. The corresponding
saturating-add study uses `VPADDUSB`; it is not conflated with SAD.

The conclusion is about instruction selection, not language-wide speed:
explicit AVX2 paths and the Zig fixed-vector probe all select `VPSADBW`, while
Clang's larger count is explained by unrolling and residual handling.

## Runtime benchmark evidence

The checked-in runtime bundle is
`results/artifacts/2026-08-07-baseline/benchmark-v2.json`. It records the host,
compiler versions, target tier, warmup/sample policy, raw samples, median,
p95, MAD, working-set bytes, and effective bytes per iteration. Its host is
Darwin arm64, where Rust and C++ report scalar dispatch and Zig reports its
native-vector implementation. Those rows are valid arm64 baseline observations;
they must not be presented as measurements of x86 `VPSADBW`.

To collect a new matched result on a compatible host, use the existing runner:

```bash
python3 scripts/run_benchmarks.py --pretty --cpu baseline --output results/local.json
```

For an x86-64-v3 runtime comparison, use an actual x86-64-v3 host and record
the resulting bundle's complete host and toolchain metadata. Shared CI compile
and codegen jobs do not establish runtime throughput.

## Correctness coverage

The language test suites compare vector/best implementations with independent
scalar references over zero, short, vector-boundary, odd, prime, and randomized
lengths. The C++ smoke test also exercises the same dispatch contract. The raw
probes are codegen inputs, not substitutes for runtime differential tests.

## Limits of the result

This case study does not claim that `VPSADBW` always wins: memory behavior,
array length, dispatch overhead, compiler unrolling, and target CPU all matter.
It establishes a fair semantic contract and shows that the reviewed x86 paths
already select the compact instruction, so inline assembly is not justified by
this evidence.
