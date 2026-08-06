# Codegen snapshots

The repository treats generated assembly as versioned experiment evidence.

## Target profiles

CI currently generates snapshots for **x86-64-v3**. This gives Rust, Zig, Clang, and GCC a common AVX2-class target without depending on the GitHub runner's exact host CPU.

```bash
python3 scripts/generate_codegen_snapshots.py --target x86-64-v3
```

Supported profiles:

- `x86-64`: conservative baseline.
- `x86-64-v3`: AVX2-class comparison baseline used in CI.
- `sapphirerapids`: AVX-512/AVX-512-FP16-class investigation target; useful for checking whether native half arithmetic removes F16C conversion traffic.
- `native`: local host specialization; do not compare across machines without CPU metadata.

The Sapphire Rapids profile is intentionally not a required CI job yet. Run it to probe compiler support and lowering independently of whether the runner CPU can execute the resulting code; these are compile-only snapshots.

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
- does the Sapphire Rapids profile replace conversion-heavy F16C-style lowering with native FP16 instructions?
- how does native Zig f16 differ from explicit promote-once code?

## Comparing snapshots

Raw assembly text is intentionally not used as the regression contract. Compare manifests instead:

```bash
python3 scripts/compare_codegen.py \
  results/codegen/manifest-x86-64-v3-before.json \
  results/codegen/manifest-x86-64-v3.json \
  --pretty
```

The `simd-lab-codegen-diff-v1` output reports per-probe instruction-count, vector-instruction-count, and tracked-mnemonic deltas. This makes large regressions visible while tolerating harmless register allocation and label changes.

## CI

The `codegen` CI job generates the x86-64-v3 snapshot set and uploads it as the `codegen-x86-64-v3` artifact. Shared-hosted CI is suitable for deterministic compilation/codegen evidence; runtime timings from shared runners are not treated as performance evidence.

The next regression step is to retain a known-good manifest or previous CI artifact and establish thresholds for meaningful conversion/instruction-count changes rather than requiring byte-for-byte assembly identity.
