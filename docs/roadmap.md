# Roadmap

This document is the authoritative backlog for `rust-zig-cpp-simd-lab`.

The project is not intended to stop at a handful of toy SIMD examples. The long-term goal is to build a reproducible laboratory for comparing how Rust, Zig 0.16, and C++23 express and lower common integer, floating-point, mixed-width, reduction, image-processing, and low-level CPU kernels across multiple ISAs and compilers.

The comparison should always separate language ergonomics, frontend lowering, optimizer behavior, backend instruction selection, runtime dispatch, numerical semantics, and hand-written ISA specialization. Faster code is only considered better when the semantic contract is explicit and correctness is preserved for the intended workload.

## Principles

- [x] Keep scalar reference implementations for correctness.
- [x] Compare semantically equivalent kernels.
- [x] Record compiler version, target CPU/features, optimization mode, and benchmark parameters.
- [x] Inspect generated code before attributing performance to a language.
- [x] Treat architecture-specific code as a separate implementation tier.
- [x] Track numerical-semantic differences explicitly.
- [x] Prefer vector IR/intrinsics over inline assembly unless evidence shows a real compiler limitation.
- [ ] Maintain reviewed known-good codegen manifests and benchmark baselines.
- [ ] Add regression thresholds only for high-signal metrics rather than exact assembly text.

# 1. Type matrix

## Integer types

- [x] `u8`
- [x] `i8`
- [x] `u16` probes
- [x] `i16`
- [x] `u32`
- [x] `i32`
- [x] selected `u64`
- [x] selected `i64`

## Floating-point types

- [x] `f16` investigation track
- [x] `f32`
- [x] `f64` scalar/codegen probes
- [x] broaden `f64` runtime/vector coverage
- [ ] investigate `bf16` where language/toolchain/ISA support is practical

## Mixed-width pipelines

- [ ] `u8 -> u16`
- [ ] `u8 -> u32`
- [ ] `i8 -> i16`
- [ ] `u16 -> u32`
- [ ] `i16 -> i32`
- [x] `f16 -> f32 -> f16`
- [ ] mixed integer/float conversion pipelines used in image processing
- [ ] narrowing with truncation
- [ ] narrowing with rounding
- [ ] narrowing with saturation
- [ ] packing and unpacking multiple lanes

# 2. Core arithmetic kernels

For each useful kernel, aim for the following implementation tiers where the language/toolchain permits them:

1. scalar reference;
2. ordinary scalar source intended for autovectorization;
3. portable/native language vector form;
4. ISA-specific intrinsic implementation;
5. inline-assembly reference only when justified by measured/codegen evidence.

## Basic arithmetic

- [x] add/multiply/FMA through AXPY
- [ ] vector add/subtract for all common integer widths
- [ ] vector multiply for integer and float widths
- [ ] division/reciprocal behavior for `f32` and `f64`
- [ ] reciprocal-estimate + refinement experiments
- [ ] square/squared-difference kernels
- [ ] absolute value and absolute difference
- [ ] negation/sign manipulation

## Min/max/clamp

- [x] `u8` clamp probe
- [x] `u16` clamp probe
- [x] `f32` clamp probe
- [x] `f64` clamp probe
- [x] Zig native `f16` clamp
- [x] Zig promote-once `f16 -> f32 -> f16` clamp
- [x] Rust F16C boundary-promotion clamp
- [x] C++ F16C boundary-promotion clamp
- [ ] signed integer min/max/clamp matrix
- [ ] larger vector-array runtime clamp kernels beyond tiny codegen probes
- [ ] NaN-sensitive float min/max variants with clearly documented semantics

## Saturating arithmetic

- [x] `u8` saturating-add codegen probes
- [x] `u8` saturating-add runtime benchmarks
- [x] `i8` saturating-add runtime/probe coverage
- [x] `u16` saturating-add runtime/probe coverage
- [x] `i16` saturating-add runtime/probe coverage
- [x] `u32` saturating-add runtime/probe coverage
- [x] `i32` saturating-add runtime/probe coverage
- [x] `u64` saturating-add runtime/probe coverage
- [x] `i64` saturating-add runtime/probe coverage
- [ ] saturating subtract for the fixed-width matrix
- [ ] explicit widening + clamp fallback implementations for comparison

## Shifts and bit operations

- [ ] constant left/right shifts
- [ ] variable shifts
- [ ] arithmetic vs logical shifts
- [ ] rotates
- [ ] bitwise AND/OR/XOR/NOT
- [ ] mask creation and mask application
- [ ] population count where vector support exists
- [ ] leading/trailing-zero idioms where meaningful

# 3. Reductions and dot-product families

## Reductions

- [x] squared-error reduction
- [x] SAD reduction
- [ ] integer sum reductions by width
- [ ] float sum reductions
- [ ] min/max reductions
- [ ] pairwise/tree vs linear reductions
- [ ] multiple accumulators to hide dependency latency
- [ ] reproducible vs reassociated floating-point reductions
- [ ] Kahan/compensated reduction baseline for numerical comparison

## Absolute difference / SAD

- [x] scalar `u8` SAD
- [x] Rust AVX2 `VPSADBW` reference
- [x] C++ AVX2 `VPSADBW` reference
- [x] Zig native-vector absdiff -> widen -> reduce
- [x] architecture-neutral C++ autovec probe
- [ ] `u16` SAD
- [ ] signed absolute-difference variants
- [ ] block SAD representative of motion estimation / image matching

## Dot products

- [x] `f32` dot product
- [x] `f64` dot product
- [x] `i16 * i16 -> i64` accumulation dot product
- [x] `u8/i8` mixed signedness dot products
- [ ] x86 `VPMADDWD` idioms
- [ ] x86 VNNI / AVX-VNNI dot-product variants
- [ ] AArch64 dot-product extension variants
- [ ] wasm SIMD equivalents where available
- [ ] compare explicit dot-product instructions against generic widening multiply + reduction

# 4. Widening multiply and multiply-accumulate

- [x] `u8 * u8 -> u16`
- [x] `i8 * i8 -> i16`
- [x] `u16 * u16 -> u32`
- [x] `i16 * i16 -> i32`
- [x] `u32 * u32 -> u64`
- [x] `i32 * i32 -> i64`
- [ ] widening multiply-accumulate
- [ ] pairwise multiply-add idioms
- [ ] coefficient/filter kernels that reuse widened data
- [ ] measure pack/unpack overhead around widened arithmetic

# 5. Floating-point investigation

## FP16 semantics and lowering

- [x] native Zig `f16` path
- [x] promote-once Zig path
- [x] Rust F16C boundary conversion
- [x] C++ F16C boundary conversion
- [x] shared exact finite benchmark corpus
- [x] shared edge-case corpus
- [x] per-strategy semantic dump executables
- [x] cross-strategy raw-bit comparator
- [ ] run and interpret edge cases on a host with F16C
- [ ] document signed-zero behavior
- [ ] document infinity behavior
- [ ] document subnormal behavior
- [ ] document NaN payload/sign behavior
- [ ] document signaling-NaN behavior where meaningful
- [ ] document rounding differences between native-half and promote-to-f32 computation
- [ ] test optimized/fast float modes explicitly
- [ ] quantify conversion-hoisting wins
- [ ] check whether native AVX-512 FP16 removes conversion-heavy lowering
- [ ] check AArch64 native FP16 lowering

## `f32` / `f64`

- [ ] precise vs fast-math variants
- [ ] FMA contraction on/off
- [ ] reassociation on/off
- [ ] denormal/FTZ/DAZ-sensitive experiments where observable
- [ ] min/max NaN semantics across language/library forms
- [ ] conversion to/from integer with truncation and rounding variants

# 6. Data rearrangement

- [ ] broadcast/splat
- [ ] shuffle/permutation
- [ ] lane extract/insert
- [ ] interleave/deinterleave
- [ ] zip/unzip
- [ ] transpose 4x4
- [ ] transpose 8x8 where useful
- [ ] byte shuffle/table lookup
- [ ] RGBA interleaved -> planar decomposition
- [ ] planar -> interleaved composition
- [ ] gather/scatter experiments where hardware support exists
- [ ] contiguous-load alternatives to gather

Important x86 instruction-selection cases to watch include `VPSHUFB`, `VPERM*`, `VUNPCK*`, and pack/narrow instruction families. Equivalent AArch64 table/zip/unzip operations should be tracked separately rather than forcing x86 terminology onto other ISAs.

# 7. Image/video-oriented kernels

These are the bridge from microbenchmarks to realistic workloads.

## Pixel arithmetic

- [ ] pixel scaling / normalization
- [ ] clamp-to-range
- [ ] add/subtract with saturation
- [ ] weighted blend / lerp
- [ ] alpha blend
- [ ] average / rounded average
- [ ] absdiff image kernel

## Error/quality kernels

- [x] squared error
- [x] SAD foundation
- [ ] MSE
- [ ] PSNR accumulation path
- [ ] variance / covariance primitives
- [ ] SSIM window primitives
- [ ] local mean/variance kernels

## Convolution / filtering

- [ ] 3-tap horizontal convolution
- [ ] 5-tap horizontal convolution
- [ ] 7/8-tap filter representative of resampling
- [ ] short vertical convolution
- [ ] separable 2D convolution
- [ ] integer coefficient convolution with widening accumulation
- [ ] float convolution with FMA
- [ ] edge/tail strategies for filters

## Color/pixel conversion

- [ ] `u8 <-> f32` normalization
- [ ] RGB <-> planar transforms
- [ ] RGB <-> YUV matrix multiply
- [ ] limited/full-range scaling
- [ ] packed RGB(A) channel extraction
- [ ] integer matrix approximations vs float matrix paths
- [ ] eventually add representative transfer-function math where SIMD-friendly

# 8. Implementation-style matrix

## Rust

- [x] scalar/autovec source
- [x] stable `std::arch` AVX2/FMA/F16C examples
- [x] runtime `is_x86_feature_detected!` dispatch
- [ ] nightly `std::simd` comparison track
- [ ] evaluate mature portable-SIMD crates where useful
- [ ] AArch64 `std::arch` NEON implementations
- [ ] wasm SIMD implementations
- [ ] generic compile-time abstractions that avoid duplicating whole kernels
- [ ] measure abstraction/codegen cost of iterator vs index vs chunked loops

## Zig 0.16

- [x] native `@Vector`
- [x] `@reduce`
- [x] vector widening / `@floatCast`
- [x] saturating vector operator probe
- [ ] `@shuffle` kernels
- [ ] `@select` kernels
- [ ] generic vector-length helpers
- [ ] explicit target-feature specialization
- [ ] architecture-specific intrinsic/extern approaches where useful
- [ ] inline asm reference cases
- [ ] optimized float-mode comparisons

## C++23

- [x] scalar/autovec source
- [x] x86 intrinsics
- [x] runtime CPU detection on GCC/Clang
- [x] MSVC runtime dispatch with an isolated `/arch:AVX2` translation unit
- [ ] compare GCC vs Clang consistently
- [ ] investigate `std::experimental::simd` / implementation status separately from ISO C++23 baseline
- [ ] portable wrapper/library comparison if worthwhile
- [ ] NEON intrinsics
- [ ] wasm SIMD intrinsics
- [ ] compiler vector extensions as a separately labelled tier

# 9. ISA / target matrix

## x86-64

- [x] baseline x86-64 snapshot profile
- [x] x86-64-v3 / AVX2-class snapshot profile
- [x] AVX2
- [x] FMA
- [x] F16C
- [ ] SSE2/SSE4 fallback examples where useful
- [ ] AVX-only cases where relevant
- [ ] AVX-512 foundation
- [ ] AVX-512BW integer kernels
- [ ] AVX-512VNNI dot products
- [ ] AVX-512 FP16 native-half investigation
- [x] Sapphire Rapids compile profile
- [ ] execute Sapphire Rapids/FP16 benchmarks on compatible hardware

## AArch64

- [x] NEON/AdvSIMD cross-target snapshot pipeline
- [x] FP16 cross-target profile
- [ ] runtime NEON kernels
- [ ] runtime native FP16 kernels
- [ ] AArch64 dot-product extension
- [ ] SVE/SVE2 as an advanced/future track

## WebAssembly

- [x] wasm-simd128 cross-target snapshot pipeline
- [ ] runtime wasm-simd128 benchmark harness
- [ ] browser benchmark harness
- [ ] Node/WASI benchmark option for reproducibility
- [ ] compare scalar wasm vs SIMD wasm
- [ ] document operations with poor/no wasm SIMD equivalent

# 10. Runtime dispatch

- [x] Rust AVX2/FMA dispatch example
- [x] Rust F16C availability checks
- [x] C++ GCC/Clang x86 runtime dispatch examples
- [ ] common dispatch taxonomy across languages
- [ ] scalar -> SSE/AVX2 -> AVX-512 tiers
- [ ] AArch64 feature dispatch where needed
- [ ] function-pointer dispatch selected once vs per-call detection
- [ ] cached dispatch
- [ ] compile-time target-only builds vs fat binaries
- [ ] quantify dispatch overhead for small kernels

# 11. Memory behavior

- [ ] aligned vs unaligned loads/stores
- [ ] non-temporal stores where justified
- [ ] prefetch experiments only for workloads where memory latency matters
- [ ] contiguous vs strided access
- [ ] structure-of-arrays vs array-of-structures
- [ ] interleaved RGB/RGBA access
- [x] cache-resident vs streaming datasets
- [x] working-set sweeps across L1/L2/L3/DRAM
- [ ] memory-bandwidth roofline comparisons
- [ ] aliasing/restrict/noalias effects
- [x] allocation-free benchmark discipline

# 12. Tail handling

- [x] scalar tails
- [ ] masked tails
- [ ] overread-safe padded buffers where appropriate
- [ ] loop peeling for alignment
- [ ] compare vector widths on short arrays
- [x] test pathological lengths: 0, 1, lanes-1, lanes, lanes+1, odd sizes, prime sizes
- [ ] benchmark tail overhead separately from steady-state throughput

# 13. Benchmark methodology

- [x] matched data size, warmup, and iteration counts
- [x] allocation outside timed region
- [x] correctness checks before timing
- [x] `ns/element`
- [x] effective GiB/s
- [x] machine-readable benchmark JSON
- [ ] cycles/element
- [x] multiple samples and robust statistics
- [x] median / p95 / median absolute deviation
- [ ] confidence intervals where useful
- [ ] CPU pinning guidance
- [ ] governor/frequency guidance
- [ ] warm/cold cache modes
- [ ] randomized benchmark order
- [ ] compiler/ISA matrix runner
- [ ] persistent result history
- [ ] compare benchmark result sets and flag regressions
- [ ] optional Criterion-style richer benchmarking without making the core lab framework-dependent

# 14. Hardware-counter analysis

- [ ] Linux `perf stat` integration
- [ ] retired instructions
- [ ] cycles
- [ ] IPC
- [ ] branches / branch misses
- [ ] cache misses where relevant
- [ ] vector instruction counts where counters permit
- [ ] frontend/backend stall indicators on supported systems
- [ ] optional platform equivalents for macOS/Windows where practical

# 15. Static performance analysis

- [x] assembly instruction-count analyzer
- [x] tracked mnemonic families
- [x] codegen manifest schema
- [x] manifest diff tool
- [x] loose regression policy
- [ ] `llvm-mca` integration
- [ ] uiCA integration for supported x86 chips
- [ ] llvm-exegesis experiments for isolated instruction behavior if useful
- [ ] code-size metrics per exported kernel
- [ ] basic-block level instruction counts
- [ ] estimated throughput / critical path
- [ ] port-pressure reports

# 16. Compiler-analysis tooling

- [ ] capture LLVM IR for Rust
- [ ] capture LLVM IR for Zig where applicable
- [ ] capture LLVM IR for Clang C++
- [ ] compare IR for equivalent source constructs
- [ ] GCC GIMPLE/vectorizer dump track
- [ ] Clang optimization remarks (`-Rpass`, `-Rpass-missed`)
- [ ] Rust LLVM optimization/vectorization remarks where practical
- [ ] Zig LLVM diagnostics where exposed
- [ ] missed-vectorization report collection
- [ ] identify frontend-lowering vs middle-end vs backend failures
- [ ] minimized reproducer generation for compiler bug reports

# 17. Codegen snapshot coverage

- [x] Rust x86 snapshots
- [x] Zig x86 snapshots
- [x] Clang x86 snapshots
- [x] GCC x86 snapshots
- [x] AArch64 cross-target snapshots
- [x] wasm cross-target snapshots
- [x] inspect and document x86-64-v3 SAD lowering
- [x] inspect and document x86-64-v3 saturation lowering
- [ ] inspect and document x86 FP16 conversion counts
- [ ] inspect Sapphire Rapids native-half lowering
- [ ] inspect AArch64 SAD idiom lowering (`UABD`/widening/reduction)
- [ ] inspect AArch64 saturation lowering (`UQADD`)
- [ ] inspect AArch64 half arithmetic
- [ ] inspect wasm saturating add opcode selection
- [ ] inspect wasm widening/reduction cost
- [x] promote manually reviewed manifests to known-good baselines
- [x] store a concise human-readable interpretation beside each reviewed baseline

# 18. Numerical-correctness framework

- [x] scalar references
- [x] exact finite FP16 validation
- [x] FP16 raw-bit edge corpus
- [x] reusable random differential tests across language implementations
- [x] deterministic seeded datasets
- [ ] integer exhaustive testing for small-vector kernels where feasible
- [ ] float ULP-distance reporting
- [ ] absolute/relative error reporting
- [ ] NaN-aware equality policies
- [ ] signed-zero-aware policies
- [ ] overflow/saturation contract tests
- [ ] property testing for min/max/clamp, widening, narrowing, and packs

# 19. Inline assembly

Inline assembly is an escalation tier, not the default implementation strategy.

- [ ] identify compiler/intrinsic cases with demonstrated bad codegen
- [ ] write a minimal Rust `asm!` reference when justified
- [ ] write a minimal Zig `asm` reference when justified
- [ ] write a minimal GCC/Clang extended-asm C++ reference when justified
- [ ] document complete register/clobber/memory contracts
- [ ] compare asm against intrinsics and vector source
- [ ] reject asm versions that win only through changed semantics or unfair assumptions
- [ ] document maintenance/portability cost alongside speedup

Potential candidates should emerge from evidence—for example repeated FP16 conversion traffic or a missed compact integer idiom—not from a desire to hand-write assembly for its own sake.

# 20. Compiler/toolchain matrix

- [x] pinned stable rustc baseline
- [x] Zig 0.16 stable toolchain pin in CI
- [x] GCC
- [x] Clang
- [ ] track exact LLVM major version underneath Rust/Zig/Clang where useful
- [ ] latest stable vs nightly Rust comparison for portable SIMD
- [ ] multiple GCC major versions when codegen differs materially
- [ ] multiple Clang/LLVM versions for regression/bisect work
- [ ] Zig release/dev snapshots around known SIMD bugs
- [ ] compiler-bisect automation for important regressions

# 21. Reporting and visualization

- [x] JSON benchmark output
- [x] JSON codegen output
- [ ] CSV export
- [ ] Markdown summary tables
- [ ] benchmark comparison tables by kernel/type/ISA/language
- [ ] codegen tables showing instruction counts and key mnemonic counts
- [ ] plots for throughput vs data size
- [ ] plots for speedup over scalar
- [ ] plots for instruction count vs runtime
- [ ] FP16 conversion-count visualization
- [ ] architecture comparison reports
- [ ] checked-in result summaries only when hardware/toolchain metadata is complete

# 22. CI

- [x] Rust build/test
- [x] Zig build/test
- [x] C++ build/test
- [x] benchmark targets compile in CI
- [x] x86 codegen artifact generation
- [x] AArch64 codegen artifact generation
- [x] wasm codegen artifact generation
- [ ] run codegen regression policy against reviewed baselines
- [ ] separate correctness CI from performance evidence
- [ ] sanitizers for C++ where relevant
- [ ] Miri/UB-focused Rust checks where relevant
- [x] Zig safety-mode correctness job in addition to ReleaseFast
- [ ] compiler matrix jobs for selected releases
- [ ] artifact retention/versioning strategy

# 23. Documentation

- [x] benchmark methodology
- [x] FP16 semantic policy
- [x] codegen probe methodology
- [x] codegen snapshot methodology
- [x] type/operation matrix
- [x] this complete roadmap
- [ ] per-kernel documentation describing mathematical operation and semantics
- [ ] per-ISA implementation notes
- [ ] compiler pathology case studies
- [ ] "how to reproduce" pages for important findings
- [ ] glossary for autovec, portable SIMD, intrinsics, ISA, vector width, reduction, saturation, widening, FMA, F16C, etc.
- [ ] contributor guide for adding a new kernel fairly

# 24. Candidate case studies

The repo should eventually contain a small set of polished studies rather than only a large benchmark matrix.

- [ ] Zig `f16` clamp conversion explosion vs promote-once strategy
- [ ] generic `u8` SAD vs x86 `VPSADBW`
- [ ] generic saturating add vs `VPADDUSB` / `UQADD` / wasm saturating add
- [ ] dot product: generic widening multiply vs dedicated dot-product ISA instructions
- [ ] short convolution: generic vectors vs hand-tuned ISA intrinsics
- [ ] shuffle-heavy RGB/RGBA deinterleave across x86/NEON/wasm
- [ ] runtime dispatch overhead and implementation selection

# 25. Definition of a completed kernel family

A kernel family should only be considered substantially complete when most applicable boxes below are satisfied:

- [ ] scalar reference exists;
- [ ] correctness tests exist;
- [ ] ordinary/autovec source exists;
- [ ] portable/native vector implementation exists where the language supports it;
- [ ] useful ISA-specific implementation exists where justified;
- [ ] runtime benchmark exists;
- [ ] machine-readable result exists;
- [ ] codegen probe exists;
- [ ] numerical semantics are documented;
- [ ] relevant architecture targets are inspected;
- [ ] performance result has compiler/CPU metadata;
- [ ] any inline-asm version is justified by evidence;
- [ ] findings are summarized in human-readable documentation.

# Near-term priority

The most useful next sequence is:

1. run and interpret the existing FP16 edge-case semantics suite;
2. inspect existing x86-64-v3 codegen artifacts for FP16, SAD, and saturating add;
3. generate/inspect Sapphire Rapids FP16 snapshots;
4. inspect AArch64 NEON/FP16 and wasm-simd128 snapshots;
5. promote reviewed manifests to known-good baselines;
6. add runtime saturating-add benchmarks;
7. [x] add dot product and widening-multiply families

Stage 7 evidence covers scalar/autovec Rust and C++ APIs, scalar plus
`@Vector` Zig APIs, independent edge/pathological/random correctness checks,
benchmark rows, and raw-pointer probes for the four dot products and six
widening products. AArch64 NEON/FP16 cross-target manifests were generated and
inspected for widening and floating reduction mnemonics. Dedicated VNNI,
`sdot`/`udot`, `vpmadd*`, wasm dot, and widening multiply-accumulate variants
remain intentionally unchecked.
8. add blend and short-convolution kernels;
9. integrate `perf` and `llvm-mca`;
10. only then add inline-assembly reference implementations where the evidence says they are warranted.
