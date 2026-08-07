# Contributing a kernel fairly

This repository compares Rust, Zig 0.16, and C++23 implementations under a
shared semantic contract. A contribution should make the comparison more
reproducible, not merely add another fast-looking loop.

## 1. Define the contract first

Before writing an optimized tier, record:

- the mathematical operation and accumulator type;
- equal-length, empty-input, short-input, and tail behavior;
- overflow, saturation, rounding, NaN, signed-zero, and aliasing rules where
  applicable;
- effective input/output bytes and the benchmark working-set convention.

Keep an independent scalar reference. If two implementations intentionally use
different numerical semantics, give them distinct names and document the
allowed difference.

## 2. Follow the repository layout

| Concern | Location |
|---|---|
| Rust implementation and tests | `rust/src/` |
| Zig implementation and tests | `zig/src/` |
| C++ implementation and smoke tests | `cpp/src/`, `cpp/include/` |
| Minimal codegen inputs | `probes/rust/`, `probes/zig/`, `probes/cpp/` |
| Shared input corpus | `data/` |
| Repeatable tooling | `scripts/` |
| Python checker tests | `tests/` |
| Human-readable findings | `docs/` |
| Reviewed manifests and result bundles | `results/` |
| Schemas and examples | `schema/` |

Do not place generated assembly, local benchmark output, compiler caches, or
scratch binaries beside source files. Use the ignored build directories or a
temporary directory for transient output.

## 3. Add correctness coverage before optimization claims

Compare every vector, dispatch, and ISA-specific tier with the scalar reference
using the established pathological lengths: zero, short, vector-width minus
one, vector-width, vector-width plus one, odd, prime, and randomized sizes.
Include extrema and semantic edge cases relevant to the type. Raw-pointer
probes are codegen inputs; they do not replace language-level differential
tests.

Use the existing repository checks where applicable:

```bash
just test
python3 -m unittest discover -s tests -p 'test_*.py'
```

The complete command and evidence protocol is in
[`docs/reproducing-findings.md`](docs/reproducing-findings.md).

## 4. Add codegen probes as small experiments

A probe should expose the operation of interest with minimal setup, no I/O, and
no benchmark harness. Keep the exported symbol and length convention consistent
with neighboring probes. Guard zero length before constructing a slice or span
from raw pointers. Label target-feature requirements explicitly.

Generate the appropriate target profile, inspect the bounded symbol, and record
compiler versions and flags. A mnemonic count or static model is codegen
evidence, not a runtime throughput result. Promote a manifest to
`results/codegen/known-good/` only after manual review of semantics, tails,
compiler metadata, and assembly.

## 5. Add runtime results only with complete provenance

Run correctness before timing and keep allocation, validation, and data creation
outside the timed region. Record the CPU, operating system, compiler versions,
optimization flags, target tier, working set, warmups, samples, aggregation
policy, units, and effective bytes. Preserve raw samples when a result is
reviewed.

Put authoritative bundles under `results/runs/` and raw evidence under the
referenced `results/artifacts/<result-id>/` directory. Do not commit a local
machine result merely because it is convenient; explain its scope and hardware
limitations in the corresponding study.

## 6. Document the finding and its limits

Update the relevant case study or methodology page with:

- the contract and implementation tiers;
- the observed codegen and runtime evidence separately;
- the exact reproduction command;
- compiler, target, and host boundaries;
- tail and semantic behavior;
- what the result does not establish.

Do not claim a language-wide speedup from one kernel, a runtime win from
assembly inspection, or native execution from a cross-target compile.

## 7. Final review checklist

Before committing a kernel contribution:

- [ ] scalar reference and contract are present;
- [ ] all applicable tiers agree with the reference;
- [ ] zero, boundary, tail, extrema, and randomized cases are covered;
- [ ] probe symbols and target features follow repository conventions;
- [ ] generated code was inspected at the intended target profile;
- [ ] runtime results, if any, have complete metadata;
- [ ] documentation separates correctness, codegen, static-model, and runtime evidence;
- [ ] no compiler cache, scratch output, or machine-specific artifact is left in
      the worktree;
- [ ] `git diff --check` and the relevant existing checks pass.
