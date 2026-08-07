#!/usr/bin/env python3
"""Generate matched compiler assembly snapshots and machine-readable metrics.

The script records command lines and compiler versions next to each assembly
file so snapshots are evidence, not anonymous compiler output.
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
PROBES = ["clamp", "dot", "image_kernels", "mixed_width", "sad", "sat_add", "sat_add_widened", "sat_sub", "sat_sub_widened", "widen_mul"]


def run(cmd: list[str], cwd: Path = ROOT) -> str:
    p = subprocess.run(cmd, cwd=cwd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
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
    ap.add_argument(
        "--target",
        default="x86-64-v3",
        choices=["x86-64", "x86-64-v3", "sapphirerapids", "native"],
    )
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    target_flags = {
        "x86-64": {"clang": ["-march=x86-64"], "gcc": ["-march=x86-64"], "rust": ["-C", "target-cpu=x86-64"], "zig": ["-mcpu=x86_64"]},
        "x86-64-v3": {"clang": ["-march=x86-64-v3"], "gcc": ["-march=x86-64-v3"], "rust": ["-C", "target-cpu=x86-64-v3"], "zig": ["-mcpu=x86_64_v3"]},
        "sapphirerapids": {"clang": ["-march=sapphirerapids"], "gcc": ["-march=sapphirerapids"], "rust": ["-C", "target-cpu=sapphirerapids"], "zig": ["-mcpu=sapphirerapids"]},
        "native": {"clang": ["-march=native"], "gcc": ["-march=native"], "rust": ["-C", "target-cpu=native"], "zig": ["-mcpu=native"]},
    }[args.target]

    jobs: list[dict[str, object]] = []
    if shutil.which("rustc"):
        for probe in PROBES:
            out = OUT / f"rust-{probe}-{args.target}.s"
            cmd = ["rustc", "--edition=2021", "-O", "--crate-type=lib", "--emit=asm", *target_flags["rust"], str(ROOT / f"probes/rust/{probe}.rs"), "-o", str(out)]
            run(cmd)
            jobs.append({"toolchain": "rustc", "probe": probe, "assembly": str(out.relative_to(ROOT)), "command": cmd})

    for exe, key in [("clang++", "clang"), ("g++", "gcc")]:
        if not shutil.which(exe):
            continue
        for probe in PROBES:
            out = OUT / f"{key}-{probe}-{args.target}.s"
            cmd = [exe, "-std=c++23", "-O3", "-S", "-masm=intel", *target_flags[key], str(ROOT / f"probes/cpp/{probe}.cpp"), "-o", str(out)]
            run(cmd)
            jobs.append({"toolchain": key, "probe": probe, "assembly": str(out.relative_to(ROOT)), "command": cmd})

    if shutil.which("zig"):
        for probe in PROBES:
            out = OUT / f"zig-{probe}-{args.target}.s"
            obj = OUT / f"zig-{probe}-{args.target}.o"
            cmd = ["zig", "build-obj", str(ROOT / f"probes/zig/{probe}.zig"), "-O", "ReleaseFast", *target_flags["zig"], f"-femit-asm={out}", f"-femit-bin={obj}"]
            run(cmd)
            jobs.append({"toolchain": "zig", "probe": probe, "assembly": str(out.relative_to(ROOT)), "command": cmd})

    asm_files = [str(ROOT / j["assembly"]) for j in jobs]
    metrics = {}
    if asm_files:
        raw = run(["python3", str(ROOT / "scripts/analyze_asm.py"), *asm_files])
        metrics = json.loads(raw)

    metadata = {
        "schema": "simd-lab-codegen-v1",
        "target_profile": args.target,
        "source_revision": source_revision(),
        "host": platform.platform(),
        "versions": {k: version(v) for k, v in {"rustc": "rustc", "zig": "zig", "clang": "clang++", "gcc": "g++"}.items()},
        "jobs": jobs,
        "metrics": metrics,
    }
    path = OUT / f"manifest-{args.target}.json"
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
