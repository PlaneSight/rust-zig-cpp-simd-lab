# Glossary

This glossary defines terms as they are used in `rust-zig-cpp-simd-lab`. It is
a guide to the repository's comparison and evidence conventions, not a
replacement for a language specification or ISA manual. See the
[methodology](methodology.md), [runtime protocol](benchmarking.md),
[codegen-probe guide](codegen-probes.md), [snapshot guide](codegen-snapshots.md),
[FP16 policy](fp16-semantics.md), and [contributor guide](../CONTRIBUTING.md) for
the complete contracts.

## Alignment

The relationship between a memory address and a byte boundary. An aligned load
or store meets the boundary required or preferred by its implementation; an
unaligned operation does not assume that boundary. Alignment is part of a
kernel contract only when it is stated and validated.

## Architecture

A processor family and execution model, such as x86-64, AArch64, or
WebAssembly. An architecture can contain several ISAs or optional instruction
extensions, so an architecture name alone does not identify the instructions
available to a kernel.

## Autovectorization

A compiler optimization that turns ordinary scalar-looking source into vector
operations without an explicit vector API in that source. The repository labels
this tier `autovec`. The label describes the source form and optimization
intent; generated code must be inspected before claiming that vector
instructions were emitted.

## Backend

The compiler component that selects, schedules, and emits instructions for a
target. Backend instruction selection is kept distinct from language frontend
lowering and optimizer transformations when explaining generated code.

## Baseline

A comparison reference with an explicit scope. A scalar baseline is the
correctness implementation; a target baseline fixes compiler and CPU features;
a reviewed codegen baseline is a known-good manifest. A baseline is not
necessarily the fastest implementation or a runtime performance result.

## Benchmark

A timed execution under the repository's runtime protocol. Benchmark rows are
wall-clock runtime evidence only when their implementation, data size, target,
toolchain, samples, and host metadata are recorded. Compilation, assembly
inspection, and static models are not benchmarks.

## Bfloat16 (BF16)

A 16-bit floating-point format with an eight-bit exponent and seven stored
fraction bits. It differs from IEEE binary16 in range, precision, conversions,
and ISA support. The roadmap treats BF16 as a separate investigation track;
`f16` results do not apply to it.

## Binary16 (FP16)

The IEEE 754 16-bit floating-point format with a five-bit exponent and ten
stored fraction bits, written `f16` where a language exposes that type. This
repository records raw 16-bit encodings when signed zero, subnormals,
infinities, NaNs, or conversion rounding must remain observable.

## Code generation (codegen)

The lowering from source through compiler intermediate forms to target machine
instructions. A codegen claim concerns emitted code for named compiler flags,
versions, and targets; it is not by itself a claim about runtime speed.

## Codegen manifest

The machine-readable summary emitted by the snapshot tooling. It records
provenance, whole-file instruction metrics, tracked mnemonic families, and
artifact paths under the `simd-lab-codegen-v1` schema. Counts must be scoped to
the recorded file or bounded symbol before they are interpreted.

## Codegen probe

A tiny exported function under `probes/` designed to expose one lowering
question with minimal setup, no timing, and no I/O. A probe supplies compile and
codegen evidence. It does not replace a production implementation,
correctness test, or benchmark.

## Codegen snapshot

A versioned set of generated assembly and its manifest for a fixed target
profile. Generated snapshots remain evidence pending review. Only manually
reviewed, semantically matched snapshots are promoted to known-good baselines.

## Compiler frontend

The compiler component that parses a language and lowers its constructs into an
intermediate representation. Differences visible in final assembly can begin in
the frontend, the optimizer, or the backend; the repository avoids attributing
every difference directly to the source language.

## Compiler middle end

The target-independent optimizer between frontend lowering and backend
instruction selection. Vectorization, reassociation, common-subexpression
elimination, and loop transformations commonly occur here, although exact
compiler boundaries vary.

## Dot product

A reduction of pairwise products, such as `sum(a[i] * b[i])`. Its contract must
name input signedness and width, multiplication width, accumulator type,
overflow behavior, and floating-point reassociation policy. A dedicated ISA dot
instruction and a generic widening-multiply reduction are separate
implementation tiers.

## Effective bandwidth

Algorithmic bytes processed per unit time, computed from the documented input
and output streams. It is not necessarily physical memory-bus traffic because
caches, write allocation, eviction, and non-temporal operations can change
actual transfers.

## F16C

The x86 conversion extension that provides packed conversions between IEEE
binary16 storage and `f32`. F16C is not native general FP16 arithmetic: the
Rust and C++ paths in this repository widen at a boundary, compute in `f32`, and
narrow once, with that numerical strategy stated explicitly.

## Fast math

A compiler or language mode that permits some floating-point transformations
beyond strict default semantics. The exact permissions vary by toolchain and
flags. Results using fast math must name the mode and must not be assumed
bit-identical to precise-mode results.

## Floating-point contraction

Combining separate floating-point multiply and add operations into one fused
operation when the language and compiler mode permit it. Contraction changes
rounding from two operations to one and therefore belongs to the numerical
contract, not only to performance tuning.

## FMA

Fused multiply-add: an operation that computes a product and sum with one final
rounding. FMA may be written explicitly or introduced through permitted
contraction. A benchmark or codegen comparison must state whether contraction
is required, allowed, or disabled.

## Implementation tier

The repository's level of implementation control: `scalar`, `autovec`,
`portable`, an explicit ISA tier such as `avx2`, `avx512`, `neon`, or
`wasm-simd128`, and finally `asm`. Comparisons name tiers so portable source is
not confused with an ISA-specific implementation.

## Intrinsic

A compiler-provided function or operation that exposes target or vector
facilities in source, such as Rust `std::arch` or C++ `<immintrin.h>` APIs. An
intrinsic can have ISA feature requirements, and it is not universally
portable. It usually guides instruction selection but does not guarantee a
one-source-call-to-one-instruction mapping.

## ISA

Instruction set architecture: the machine-visible instructions, registers,
data types, and execution rules available to code. AVX2, F16C, NEON/AdvSIMD,
and WebAssembly SIMD128 are tracked as distinct ISA capabilities or extension
tiers rather than as language features.

## Lane

One element position in a vector value or operation. Lane count depends on both
element type and vector size: a 128-bit vector can hold sixteen `u8` lanes but
four `f32` lanes.

## Latency

The time or cycle dependency from an operation's inputs to its usable result.
Latency differs from throughput. Static tools can estimate it for a modeled CPU,
but that estimate is not a hardware timing measurement.

## Mask

A per-lane predicate or bit pattern used to select, suppress, or combine vector
operations. Mask representation and masked-load/store fault behavior are
ISA-specific and must be stated when they affect memory safety or tails.

## Narrowing

Converting elements to a smaller representation, such as `u16 -> u8` or
`f32 -> f16`. The contract must distinguish truncation, rounding, saturation,
overflow, and packing order; "narrow" alone does not select one policy.

## Native vector

An explicit vector type or operation supplied directly by the language or
compiler, such as Zig `@Vector`. In this repository, `native-vector` describes
the source form. It does not claim a particular hardware instruction, ISA, or
vector register width without codegen evidence.

## NEON / AdvSIMD

AArch64's fixed-width Advanced SIMD facility, historically and commonly called
NEON. It is an ISA implementation tier distinct from portable vector source and
from scalable SVE/SVE2 vectors.

## Numerical semantics

The observable rules for values and exceptional cases: operation and
accumulator precision, overflow, saturation, rounding, reassociation, NaNs,
infinities, signed zero, subnormals, and floating-point modes. Faster code is
comparable only after this contract is made explicit.

## Portable SIMD

A language or library vector abstraction intended to express lane-wise work
without naming one hardware ISA. Portability describes the API level, not a
guarantee of identical instructions, vector widths, availability, or
performance on every target.

## Reduction

Combining multiple input or vector-lane values into fewer values, usually one,
using an operation such as sum, minimum, maximum, SAD, or dot product. Integer
overflow and floating-point order/reassociation are part of the reduction
contract.

## Runtime dispatch

Selecting an implementation after detecting host CPU features, for example
choosing scalar or AVX2 code. The detection can occur per call or once and be
cached. A fat binary must not enter a target-specific function until both CPU
and operating-system state requirements are satisfied.

## SAD

Sum of absolute differences: `sum(abs(a[i] - b[i]))`. The input widths,
signedness, accumulator type, equal-length rule, and overflow behavior are part
of the contract. x86 `VPSADBW` is one specialized unsigned-byte lowering, not
the definition of SAD itself.

## Saturating arithmetic

Arithmetic that clamps an out-of-range result to the destination type's minimum
or maximum instead of wrapping, trapping, or invoking undefined behavior.
Signed and unsigned bounds differ, so the type is part of every saturation
contract.

## Scalar reference

The intentionally straightforward implementation used as the correctness
oracle for optimized tiers. `Scalar` describes its role and source form; a
compiler may still autovectorize it unless the experiment explicitly prevents
that transformation.

## SIMD

Single instruction, multiple data: one operation applies to multiple lanes.
The term can describe an ISA instruction, compiler vector IR, or a source-level
vector abstraction, so this repository qualifies which layer is meant.

## Static performance analysis

A model-based analysis of instructions without timing the kernel on hardware.
For example, `llvm-mca` estimates throughput, latency, and resource pressure for
a named CPU model. Static estimates are neither PMU observations nor runtime
benchmark evidence.

## Tail handling

Processing elements left after complete vector-sized chunks, including arrays
shorter than one chunk. Strategies include scalar cleanup, masked operations,
loop peeling, or contractually safe padding. Zero, boundary, odd, prime, and
non-multiple lengths are correctness cases, not optional benchmark details.

## Target feature

One compiler-enabled CPU capability such as `avx2`, `fma`, `f16c`, or
`+simd128`. Compiling with a feature permits the compiler to emit corresponding
instructions; it does not prove the current host can execute them.

## Target profile

A reproducible bundle of target triple, CPU model, and enabled or disabled
features used to compile snapshots or binaries. Profiles such as `x86-64-v3`,
`aarch64-neon`, and `wasm-simd128` make codegen comparisons more precise than an
architecture name alone.

## Throughput

The steady-state rate at which work completes, reported here with units such as
`ns/element`, elements/s, or effective `GiB/s`. Throughput is distinct from
single-operation latency and must be associated with a dataset size and
benchmark protocol.

## Vector IR

A compiler intermediate-representation operation over multiple lanes. It can
come from autovectorization, a portable SIMD API, or a native vector type. The
backend may split, widen, scalarize, or combine it when selecting target
instructions.

## Vector length

The number of lanes in a source value, loop chunk, or scalable-vector runtime
instance. Always name the element type and layer: sixteen `u8` lanes, for
example, does not by itself identify a hardware register.

## Vector width

The total bit width of a vector value or machine operation, such as 128, 256,
or 512 bits. Source vector width, IR width, and emitted register width can
differ, so generated code is required to connect them.

## Wasm SIMD128

WebAssembly's fixed 128-bit SIMD extension. It is a cross-target ISA tier with
its own operations and limitations; compiling a wasm snapshot is not evidence
that it executed in a browser, Node, or WASI runtime.

## Widening

Converting elements to a larger representation before arithmetic or storage,
such as `u8 -> u16` or `i32 * i32 -> i64`. The contract must preserve signedness
and specify whether widening happens before the operation, since widening only
the already-overflowed result is not equivalent.

## Working set

The memory footprint actively touched by a benchmark row, derived from its
input and output arrays for a stated element count. Working-set labels help
orient cache regimes, but actual cache residency depends on the host and kernel.
