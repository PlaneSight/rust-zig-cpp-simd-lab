# Codegen snapshots

The repository treats generated assembly as versioned experiment evidence.

## x86 target profiles

```bash
python3 scripts/generate_codegen_snapshots.py --target x86-64-v3
```

Supported x86 profiles:

- `x86-64`: conservative baseline.
- `x86-64-v3`: AVX2-class comparison baseline used in CI.
- `sapphirerapids`: AVX-512/AVX-512-FP16-class compile-only investigation target.
- `native`: local host specialization; do not compare across machines without CPU metadata.

The Sapphire Rapids profile is intentionally not an execution requirement on shared CI. Its purpose is to inspect whether native FP16-class targeting removes conversion-heavy F16C-style lowering.

## Cross-target profiles

Architecture-neutral and Zig-native-vector probes can also be cross-compiled:

```bash
python3 scripts/generate_cross_codegen.py --target aarch64-neon
python3 scripts/generate_cross_codegen.py --target aarch64-fp16
python3 scripts/generate_cross_codegen.py --target wasm-simd128
```

Profiles:

- `aarch64-neon`: baseline AArch64 AdvSIMD lowering.
- `aarch64-fp16`: Armv8.2-A FP16/AdvSIMD investigation profile.
- `wasm-simd128`: WebAssembly SIMD128 lowering.

The C++ cross-target SAD source deliberately avoids standard-library and ISA headers so Clang can compile it without a target sysroot. The `_Float16` C++ probe is explicitly a **Clang extension experiment**, not a C++23 portable facility. Stable Rust is currently omitted from the portable cross-target vector comparison because `std::simd` remains outside the stable language surface used by this repository; architecture-specific Rust probes can be added separately when useful.

## Output

Snapshots are written under `results/codegen/`. Every manifest uses schema `simd-lab-codegen-v1` and records:

- exact compile command;
- compiler first-line version string;
- target profile;
- host metadata;
- assembly path;
- instruction count;
- vector-instruction count;
- top mnemonics;
- tracked SIMD idioms.

Tracked x86 signals include FP16 conversions (`vcvtph2ps`, `vcvtps2ph`), SAD (`vpsadbw`/`psadbw`), and widening moves (`vpmovzx*` / `vpmovsx*`). AArch64 signals include `uabd`, widening/reduction operations such as `uaddlp`/`uaddlv`/`uadalp`, FP min/max and conversion operations. WebAssembly tracking recognizes the `v128`, integer-lane, and floating-lane SIMD opcode families.

## Questions the snapshots should answer

### u8 SAD

- Does Zig's generic vector implementation become the target's compact SAD idiom?
- On x86, is `VPSADBW` selected?
- On AArch64, does lowering use `UABD` plus efficient widening/reduction instructions?
- On wasm-simd128, how many lane-extension and horizontal-reduction operations are required?
- How do portable/autovec paths compare with explicit ISA references?

### FP16 clamp

- How many half/f32 conversions are emitted on x86 without native FP16 arithmetic?
- Are conversions hoisted to the boundary or repeated around operations?
- Does Sapphire Rapids targeting replace conversion-heavy lowering with native FP16 instructions?
- Does AArch64 FP16 targeting use native half arithmetic rather than promotion?
- How does Zig native f16 differ from explicit promote-once code across those targets?

## Comparing snapshots

Raw assembly text is intentionally not the regression contract. Compare manifests instead:

```bash
python3 scripts/compare_codegen.py \
  results/codegen/manifest-x86-64-v3-before.json \
  results/codegen/manifest-x86-64-v3.json \
  --pretty
```

For automated regression checks, use the looser policy layer:

```bash
python3 scripts/check_codegen_regressions.py \
  results/codegen/known-good/manifest-x86-64-v3.json \
  results/codegen/manifest-x86-64-v3.json \
  --policy codegen-policy.json
```

`codegen-policy.json` intentionally uses broad instruction/vector growth limits plus tighter high-signal rules for FP16 conversion explosions and widening growth. This avoids treating harmless register allocation or instruction scheduling changes as regressions.

A known-good manifest should only be promoted after manual inspection of compiler versions, semantic equivalence, and the generated assembly. Policy thresholds are safeguards, not a substitute for reviewing a new baseline.

## CI

CI produces three deterministic compile/codegen artifact families:

- `codegen-x86-64-v3`
- `codegen-aarch64-neon`
- `codegen-wasm-simd128`

Shared-hosted CI is suitable for deterministic compilation/codegen evidence. Runtime timings from shared runners are **not** treated as performance evidence. AArch64 FP16 and Sapphire Rapids profiles remain explicit compile-only investigation runs until their compiler support and expected lowering are established well enough to make them required jobs.
