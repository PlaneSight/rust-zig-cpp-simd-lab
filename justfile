set shell := ["bash", "-euo", "pipefail", "-c"]

cpu := env_var_or_default("SIMD_LAB_CPU", "baseline")

default:
    @just --list

test:
    cd languages/rust && cargo test --release
    cd languages/zig && zig build test -Doptimize=ReleaseSafe
    cmake -S languages/cpp -B build/cpp -DCMAKE_BUILD_TYPE=Release -DSIMD_LAB_CPU={{cpu}}
    cmake --build build/cpp -j2
    ctest --test-dir build/cpp --output-on-failure

bench output="":
    python3 scripts/run_benchmarks.py --pretty --cpu {{cpu}} {{if output == "" { "" } else { "--output " + output }}}

probes target="x86-64-v3":
    python3 scripts/generate_codegen_snapshots.py --target {{target}}
