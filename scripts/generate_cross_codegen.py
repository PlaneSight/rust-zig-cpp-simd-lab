#!/usr/bin/env python3
"""Generate cross-target SIMD codegen evidence for AArch64 and WebAssembly.

This complements generate_codegen_snapshots.py, which is x86-oriented. The
cross-target pipeline intentionally uses architecture-neutral probes plus Zig's
native vector probes; ISA-specific x86 intrinsic sources are excluded.
"""
from __future__ import annotations

import argparse
import json
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "codegen"

PROFILES = {
    "aarch64-neon": {
        "clang_target": "aarch64-unknown-linux-gnu",
        "clang_flags": ["-march=armv8-a+simd"],
        "zig_target": "aarch64-linux-gnu",
        "zig_cpu": "baseline+neon",
    },
    "aarch64-fp16": {
        "clang_target": "aarch64-unknown-linux-gnu",
        "clang_flags": ["-march=armv8.2-a+fp16+simd"],
        "zig_target": "aarch64-linux-gnu",
        "zig_cpu": "baseline+neon+fullfp16",
    },
    "wasm-simd128": {
        "clang_target": "wasm32-unknown-unknown",
        "clang_flags": ["-msimd128"],
        "zig_target": "wasm32-freestanding",
        "zig_cpu": "baseline+simd128",
    },
}


def run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, cwd=ROOT, check=False, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        print(f"command failed: {shlex.join(cmd)}", file=sys.stderr)
        print(p.stdout, file=sys.stderr, end="" if p.stdout.endswith("\n") else "\n")
        p.check_returncode()
    return p.stdout.strip()


def version(exe: str) -> str:
    args = ["version"] if exe == "zig" else ["--version"]
    try:
        return run([exe, *args]).splitlines()[0]
    except Exception:
        return "unavailable"

def source_revision() -> str:
    try:
        return run(["git", "rev-parse", "HEAD"])
    except Exception:
        return "unavailable"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=sorted(PROFILES))
    args = ap.parse_args()
    profile = PROFILES[args.target]
    OUT.mkdir(parents=True, exist_ok=True)
    jobs: list[dict[str, object]] = []

    if shutil.which("clang++"):
        sources = [
            ("dot", ROOT / "probes/cpp/dot.cpp"),
            ("image_kernels", ROOT / "probes/cpp/image_kernels.cpp"),
            ("sad-portable", ROOT / "probes/cpp/sad_portable.cpp"),
            ("sat-add-portable", ROOT / "probes/cpp/sat_add_portable.cpp"),
            ("widen-mul", ROOT / "probes/cpp/widen_mul.cpp"),
            ("mixed-width", ROOT / "probes/cpp/mixed_width.cpp"),
        ]
        sat_sub_source = ROOT / "probes/cpp/sat_sub_portable.cpp"
        if sat_sub_source.exists():
            sources.insert(4, ("sat-sub-portable", sat_sub_source))
        if args.target == "aarch64-fp16":
            sources.append(("fp16-clang-ext", ROOT / "probes/cpp/fp16_clang.cpp"))
        for probe, source in sources:
            out = OUT / f"clang-{probe}-{args.target}.s"
            cmd = ["clang++", "-std=c++23", "-O3", "-S",
                   "--target=" + profile["clang_target"],
                   *profile["clang_flags"], str(source), "-o", str(out)]
            run(cmd)
            jobs.append({"toolchain": "clang", "probe": probe,
                         "assembly": str(out.relative_to(ROOT)), "command": cmd})

    if shutil.which("zig"):
        for probe in ["clamp", "dot", "image_kernels", "mixed_width", "sad", "sat_add", "sat_sub", "widen_mul"]:
            source = ROOT / f"probes/zig/{probe}.zig"
            if not source.exists():
                continue
            out = OUT / f"zig-{probe}-{args.target}.s"
            obj = OUT / f"zig-{probe}-{args.target}.o"
            cmd = ["zig", "build-obj", str(source), "-O", "ReleaseFast",
                   "-target", profile["zig_target"], "-mcpu=" + profile["zig_cpu"],
                   f"-femit-asm={out}", f"-femit-bin={obj}"]
            run(cmd)
            jobs.append({"toolchain": "zig", "probe": probe,
                         "assembly": str(out.relative_to(ROOT)), "command": cmd})

    asm_files = [str(ROOT / j["assembly"]) for j in jobs]
    metrics = {}
    if asm_files:
        metrics = json.loads(run(["python3", str(ROOT / "scripts/analyze_asm.py"), *asm_files]))

    metadata = {
        "schema": "simd-lab-codegen-v1",
        "target_profile": args.target,
        "source_revision": source_revision(),
        "host": platform.platform(),
        "versions": {"zig": version("zig"), "clang": version("clang++")},
        "jobs": jobs,
        "metrics": metrics,
        "notes": {
            "cpp_fp16": "_Float16 probe is a Clang extension, not a C++23 portable facility",
            "rust": "stable Rust cross-target portable-SIMD probe omitted until std::simd is stable or an explicit crate policy is chosen",
        },
    }
    path = OUT / f"manifest-{args.target}.json"
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
