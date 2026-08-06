#!/usr/bin/env python3
"""Run FP16 semantic dump tools and compare output bit patterns."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def run_json_lines(cmd: list[str], cwd: Path) -> list[dict]:
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    text = proc.stdout + ("\n" if proc.stdout and proc.stderr else "") + proc.stderr
    if proc.returncode:
        sys.stderr.write(text)
        raise SystemExit(proc.returncode)
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            rows.append(json.loads(line))
    return rows


def main() -> int:
    root = Path(__file__).resolve().parents[1]

    builds = [
        (["cargo", "build", "--release", "--bin", "fp16_semantics"], root / "rust"),
        (["cmake", "-S", "cpp", "-B", "build/cpp", "-DCMAKE_BUILD_TYPE=Release"], root),
        (["cmake", "--build", "build/cpp", "--target", "simd_lab_cpp_fp16_semantics", "-j2"], root),
        (["zig", "build", "-Doptimize=ReleaseFast"], root / "zig"),
    ]
    for cmd, cwd in builds:
        proc = subprocess.run(cmd, cwd=cwd)
        if proc.returncode:
            return proc.returncode

    rows: list[dict] = []
    rows += run_json_lines(["cargo", "run", "--release", "--bin", "fp16_semantics"], root / "rust")
    rows += run_json_lines([str(root / "build" / "cpp" / "simd_lab_cpp_fp16_semantics")], root)
    rows += run_json_lines(["zig", "build", "fp16-semantics", "-Doptimize=ReleaseFast"], root / "zig")

    active = {row["strategy"]: row for row in rows if row.get("available")}
    names = sorted(active)
    if not names:
        print(json.dumps({"schema": "simd-lab-fp16-semantics-v1", "strategies": rows, "comparisons": []}, indent=2))
        return 0

    count = min(len(active[name].get("outputs", [])) for name in names)
    comparisons = []
    for i in range(count):
        values = {name: active[name]["outputs"][i] for name in names}
        comparisons.append({
            "index": i,
            "outputs": values,
            "all_equal": len(set(values.values())) == 1,
        })

    result = {
        "schema": "simd-lab-fp16-semantics-v1",
        "strategies": rows,
        "comparisons": comparisons,
        "divergence_count": sum(not item["all_equal"] for item in comparisons),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
