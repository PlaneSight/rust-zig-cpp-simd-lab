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
On an AArch64 Darwin host, the installed Apple Clang may reject x86
`-march=x86-64*` values; that is a toolchain limitation rather than evidence
that an x86 lowering exists or does not exist. Keep x86 claims tied to the
reviewed CI manifest, and use the AArch64 profiles above for locally generated
Stage 7 evidence.

Cross-target manifests for Stage 8, including `aarch64-neon`,
`aarch64-fp16`, and `wasm-simd128`, are generated evidence pending manual
review. They are not known-good baselines or regression baselines, and no
Stage 8 cross-target lowering is authoritative. The only authoritative
reviewed known-good x86 baseline remains
`results/codegen/known-good/manifest-x86-64-v3.json`.

### Stage 8 generated interpretation

The 2026-08-07 local generation used Homebrew Clang 22.1.8 and Zig 0.16.0
on `macOS-26.6-arm64`. The integer-only image probes lowered identically in
the `aarch64-neon` and `aarch64-fp16` profiles, as expected:

- Clang C++: 402 whole-file instructions, 15 classified vector instructions,
  with `ushll`, `uaddl`, `umlal`, and `rshrn` widening/MAC/narrowing signals.
- Zig: 574 whole-file instructions, 14 classified vector instructions, with
  `ushll`, `uaddl`, `umull`/`umlal`, and `rshrn` signals.
- wasm-simd128: Clang C++ emitted 988 whole-file instructions with 84
  classified vector instructions; Zig emitted 1624 with 201. Both contain
  `i8x16`/`i16x8` families, and Zig's u32 blend path also contains `i32x4`.

The AArch64 assembly therefore provides evidence that both frontends select
widening arithmetic and rounded narrowing for these contracts. The larger
Zig and wasm whole-file counts warrant per-function investigation; they are
not runtime-performance conclusions. These counts cover all exports in each
probe file, not individual kernels, and do not establish an ISA-specialized
production implementation or a reviewed baseline.

## Output

Snapshots are written under `results/codegen/`. Every manifest uses schema `simd-lab-codegen-v1` and records:

- exact compile command;
- compiler first-line version string;
- target profile;
- source revision;
- host metadata;
- assembly path;
- instruction count;
- vector-instruction count;
- top mnemonics;
- tracked SIMD idioms.

Tracked x86 signals include FP16 conversions (`vcvtph2ps`, `vcvtps2ph`), SAD (`vpsadbw`/`psadbw`), widening moves (`vpmovzx*` / `vpmovsx*`), multiply-add idioms (`vpmaddwd`, `vpmaddubsw`, and VNNI `vpdp*` families), and vector multiply families. AArch64 signals include `uabd`, widening/reduction operations such as `uaddlp`/`uaddlv`/`uadalp`, integer widening/narrowing (`ushll`/`sshll`/`smull`/`umull`/`xtn`/`uqxtn`), integer-to-float conversion (`scvtf`/`ucvtf`), dot-product (`sdot`/`udot`/`usdot`), FP min/max and conversion operations. WebAssembly tracking recognizes the `v128`, integer-lane, floating-lane, widening multiply, and dot-product opcode families.

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


### Dot product and widening multiply

The Stage 7 probes expose scalar/autovec products and reductions without
allocations or I/O. The portable x86 baseline is intentionally separate from
future dedicated instruction studies: `vpmaddwd`, VNNI `vpdp*`, AArch64
`sdot`/`udot`, and wasm widening/dot opcodes are tracked as evidence, but are
not promoted to an expected lowering until a generated target-profile
snapshot demonstrates the idiom and its semantics match.

### Mixed-width conversion

The mixed-width probes are architecture-neutral semantic baselines. Review
integer widening moves, integer-to-f32 conversion sequences, f32/u16 and
f32/u8 narrowing, u16/u8 narrowing, and little-endian packing. Zig's native
vector exports provide a portable-vector comparison; Rust and C++ scalar
sources test ordinary autovectorization. Do not promote a specific conversion
or pack instruction from one compiler snapshot without checking tails and
the documented saturation/rounding policy.

### Stage 8 blend and short convolution

Stage 8 snapshots may record the scalar/autovec blend and fixed 3-/5-tap
u8 convolution probes for all three languages, plus Zig's explicit
`@Vector` forms. Review the generated output for explicit unsigned widening,
fixed-coefficient multiply/add structure, the `+128 >> 8`, `+2 >> 2`, and
`+8 >> 4` rounding points, clamp-to-edge border handling, and bounded scalar
tails. These are semantic/code-generation checks, not requirements for a
particular x86, Arm, or wasm instruction.

The Stage 8 contract supplies no ISA-specific implementation, measured
benchmark result bundle, or reviewed baseline. A generated manifest can be
compared as evidence only; it must not be promoted to a known-good baseline
until compiler metadata, semantic equivalence, and assembly have been
manually reviewed.



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

## Reviewed x86-64-v3 probe baseline

The x86-64-v3 snapshot from CI run `31142383190` (source revision
`0055cd3b74017746d501d8973b3441490b154625`) was manually reviewed on
2026-08-07. The host was `Linux-6.17.0-1020-azure-x86_64`; the recorded
toolchains were rustc 1.97.1, Clang 18.1.3, and GCC 13.3.0. The manifest
contains Zig jobs and assembly, but records Zig as `unavailable` because the
pre-fix generator queried `zig --version` instead of Zig's `zig version`
command. Its assembly is retained as evidence, but that artifact's Zig
compiler version is not treated as known.

The reviewed manifest is checked in at
`results/codegen/known-good/manifest-x86-64-v3.json`. It is a loose baseline,
not an exact-assembly fixture. Symbol-local counts below were obtained by
restricting the existing analyzer to each exported function body; the checked
manifest still contains whole-file aggregates and therefore does not claim to
separate vector loops from tails.

| Family | Reviewed symbol evidence | Interpretation |
|---|---|---|
| FP16 clamp | Zig `clamp_f16x16_native`: 384 `vcvtph2ps`, 256 `vcvtps2ph`; Zig `clamp_f16x16_promote_once`: 18 and 2 | Native half arithmetic repeatedly bounces through f32 on x86-64-v3. Promote-once reduces conversion traffic to boundary conversions, but it remains a deliberate numerical-semantics tradeoff. The x86 probe does not include Rust or C++ F16C production kernels, so FP16 coverage remains incomplete. |
| u8 SAD | Rust AVX2: 1 `vpsadbw`; Clang AVX2: 9 (`8` unrolled steady-state plus `1` residual); GCC AVX2: 1; Zig fixed-vector `sad_u8x32`: 1 | Explicit AVX2 paths select `VPSADBW`. Clang's extra count is unrolling, not a different operation. Scalar tails and Zig's standalone absdiff/widen probes remain separate evidence. |
| u8 saturating add | Rust AVX2: 6 `vpaddusb`; Clang AVX2: 5; GCC AVX2: 1; Zig `sat_add_u8x32`: 1 | Explicit vector paths select `VPADDUSB`; counts differ because of unrolling and residual handling. Generic scalar-source lowering is compiler-dependent and must not be conflated with the intrinsic reference. |

The baseline is accepted for the SAD and saturating-add probe families. The
FP16 conversion finding is recorded but is not promoted to a complete
cross-language baseline until Rust/C++ F16C and the Clang extension probe are
included in the same target-profile workflow. Regenerate and compare it with:

```bash
python3 scripts/generate_codegen_snapshots.py --target x86-64-v3
python3 scripts/check_codegen_regressions.py \
  results/codegen/known-good/manifest-x86-64-v3.json \
  results/codegen/manifest-x86-64-v3.json \
  --policy codegen-policy.json
```

## CI

CI produces three deterministic compile/codegen artifact families:

- `codegen-x86-64-v3`
- `codegen-aarch64-neon`
- `codegen-wasm-simd128`

Shared-hosted CI is suitable for deterministic compilation/codegen evidence. Runtime timings from shared runners are **not** treated as performance evidence. AArch64 FP16 and Sapphire Rapids profiles remain explicit compile-only investigation runs until their compiler support and expected lowering are established well enough to make them required jobs.
