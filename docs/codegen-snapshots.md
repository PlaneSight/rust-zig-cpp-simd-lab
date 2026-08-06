# Codegen snapshots

The repository treats generated assembly as versioned experiment evidence.

## Stable target profile

CI currently generates snapshots for **x86-64-v3**. This gives Rust, Zig, Clang, and GCC a common AVX2-class target without depending on the GitHub runner's exact host CPU.

```bash
python3 scripts/generate_codegen_snapshots.py --target x86-64-v3
```

Other supported profiles are `x86-64` and `native`. `native` is useful for local investigation but should not be used for cross-machine comparisons without recording the CPU model.

## Output

Snapshots are written under `results/codegen/`:

```text
manifest-x86-64-v3.json
rust-clamp-x86-64-v3.s
rust-sad-x86-64-v3.s
zig-clamp-x86-64-v3.s
zig-sad-x86-64-v3.s
clang-clamp-x86-64-v3.s
clang-sad-x86-64-v3.s
gcc-clamp-x86-64-v3.s
gcc-sad-x86-64-v3.s
```

The manifest uses schema `simd-lab-codegen-v1` and records:

- exact compile command;
- compiler first-line version string;
- target profile;
- host metadata;
- assembly path;
- instruction count;
- vector-instruction count;
- top mnemonics;
- tracked SIMD idioms.

Tracked idioms include FP16 conversions (`vcvtph2ps`, `vcvtps2ph`), SAD (`vpsadbw`/`psadbw`), and widening moves (`vpmovzx*` / `vpmovsx*`).

## Questions the snapshots should answer

### SAD

For a u8 absolute-difference reduction:

- does the portable/vector implementation become `VPSADBW`?
- if not, does it lower to min/max/subtract + widening + horizontal adds?
- how many instructions and widening operations are required?
- do Rust and C++ autovec discover the idiom without explicit intrinsics?

### FP16 clamp

For the clamp family:

- how many `vcvtph2ps` / `vcvtps2ph` conversions are emitted?
- are conversions hoisted to the boundary or repeated around operations?
- does target ISA materially change the lowering?
- how does native Zig f16 differ from explicit promote-once code?

## CI

The `codegen` CI job generates the x86-64-v3 snapshot set and uploads it as the `codegen-x86-64-v3` artifact. Shared-hosted CI is suitable for deterministic compilation/codegen evidence; runtime timings from shared runners are not treated as performance evidence.

A future step should compare the manifest against a checked-in expectation or previous artifact and flag large instruction/conversion regressions without requiring exact assembly-text equality.
