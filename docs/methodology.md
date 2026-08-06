# Benchmark methodology

This repository is intended to compare implementations, not marketing claims.

## Correctness first

Every optimized kernel must be checked against a scalar reference over:

- lengths smaller than one vector
- exact vector-width lengths
- non-multiple tails
- large buffers
- representative and adversarial numeric inputs

Floating-point implementations may differ because of FMA contraction, reassociation, reduction order, or fast-math settings. Tests should therefore use an explicitly documented error tolerance rather than silently demanding bit identity.

## Benchmark dimensions

For every result record:

- CPU model and microarchitecture
- operating system
- compiler and exact version
- optimization mode/flags
- target CPU / ISA features
- kernel and implementation name
- element type
- buffer size
- alignment if controlled
- number of warmup and measured iterations
- median and dispersion, not only the best sample

Useful derived metrics include ns/element, elements/s, bytes/s, and cycles/element when reliable cycle counters are available.

## Avoid common benchmark traps

Do not include allocation, random-data generation, logging, or validation in timed regions unless they are specifically what is being measured.

Prevent dead-code elimination by consuming outputs. Keep input generation deterministic. Use enough work that timer resolution is irrelevant, but also test several working-set sizes so L1/L2/L3/DRAM behavior is visible.

Run on an otherwise quiet machine. For serious measurements on Linux, consider CPU affinity, a performance governor, and `perf stat` counters. Record these choices rather than pretending the machine is noise-free.

## Compare generated code

When results differ, inspect assembly before inventing a language-level explanation. Useful questions:

1. Did the scalar loop autovectorize?
2. Which vector width was selected?
3. Was FMA emitted?
4. Was the loop unrolled?
5. How is the tail handled?
6. Did alias analysis inhibit vectorization?
7. Is the reduction dependency-bound?
8. Are there unexpected spills, calls, bounds checks, or conversions?
9. Are the implementations actually using the same floating-point semantics?

## Optimization tiers

Use consistent labels:

- `scalar`: intentionally straightforward reference code
- `autovec`: ordinary source written to allow compiler vectorization
- `portable`: language/library vector abstraction independent of a specific ISA
- `avx2`, `avx512`, `neon`, `wasm-simd128`: explicitly ISA-oriented implementation
- `asm`: inline/standalone assembly implementation

A language comparison should normally show more than one tier. Comparing Zig `@Vector` to deliberately scalar Rust/C++ would not answer a useful question.
