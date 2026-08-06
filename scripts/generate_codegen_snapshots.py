#!/usr/bin/env python3
"""Generate matched compiler assembly snapshots and machine-readable metrics.

The script records command lines and compiler versions next to each assembly
file so snapshots are evidence, not anonymous compiler output.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "codegen"


def run(cmd: list[str], cwd: Path = ROOT) -> str:
    p = subprocess.run(cmd, cwd=cwd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return p.stdout.strip()


def version(exe: str) -> str:
    try:
        return run([exe, "--version"]).splitlines()[0]
    except Exception:
        return "unavailable"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="x86-64-v3", choices=["x86-64", "x86-64-v3", "native"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    target_flags = {
        "x86-64": {"clang": ["-march=x86-64"], "gcc": ["-march=x86-64"], "rust": ["-C", "target-cpu=x86-64"], "zig": ["-mcpu=x86_64"]},
        "x86-64-v3": {"clang": ["-march=x86-64-v3"], "gcc": ["-march=x86-64-v3"], "rust": ["-C", "target-cpu=x86-64-v3"], "zig": ["-mcpu=x86_64_v3"]},
        "native": {"clang": ["-march=native"], "gcc": ["-march=native"], "rust": ["-C", "target-cpu=native"], "zig": ["-mcpu=native"]},
    }[args.target]

    jobs: list[dict[str, object]] = []
    if shutil.which("rustc"):
        for probe in ["clamp", "sad"]:
            out = OUT / f"rust-{probe}-{args.target}.s"
            cmd = ["rustc", "-O", "--crate-type=lib", "--emit=asm", *target_flags["rust"], str(ROOT / f"probes/rust/{probe}.rs"), "-o", str(out)]
            run(cmd)
            jobs.append({"toolchain": "rustc", "probe": probe, "assembly": str(out.relative_to(ROOT)), "command": cmd})

    for exe, key in [("clang++", "clang"), ("g++", "gcc")]:
        if not shutil.which(exe):
            continue
        for probe in ["clamp", "sad"]:
            out = OUT / f"{key}-{probe}-{args.target}.s"
            cmd = [exe, "-std=c++23", "-O3", "-S", "-masm=intel", *target_flags[key], str(ROOT / f"probes/cpp/{probe}.cpp"), "-o", str(out)]
            run(cmd)
            jobs.append({"toolchain": key, "probe": probe, "assembly": str(out.relative_to(ROOT)), "command": cmd})

    if shutil.which("zig"):
        for probe in ["clamp", "sad"]:
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
