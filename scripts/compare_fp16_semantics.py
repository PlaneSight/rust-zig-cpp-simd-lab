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


def collect_available_strategies(
    rows: list[dict], expected_length: int
) -> tuple[dict[str, dict], list[dict]]:
    active: dict[str, dict] = {}
    invalid_results: list[dict] = []
    for row in rows:
        if not row.get("available"):
            continue

        outputs = row.get("outputs")
        actual_length = len(outputs) if isinstance(outputs, list) else None
        if actual_length != expected_length:
            invalid_results.append(
                {
                    "strategy": row.get("strategy"),
                    "available": True,
                    "valid": False,
                    "reason": "output-length-mismatch",
                    "expected_length": expected_length,
                    "actual_length": actual_length,
                }
            )
            continue

        active[row["strategy"]] = row
    return active, invalid_results


def main() -> int:
    root = Path(__file__).resolve().parents[1]

    builds = [
        (["cargo", "build", "--release", "--bin", "fp16_semantics"], root / "languages" / "rust"),
        (["cmake", "-S", "languages/cpp", "-B", "build/cpp", "-DCMAKE_BUILD_TYPE=Release"], root),
        (["cmake", "--build", "build/cpp", "--target", "simd_lab_cpp_fp16_semantics", "-j2"], root),
        (["zig", "build", "-Doptimize=ReleaseFast"], root / "languages" / "zig"),
    ]
    for cmd, cwd in builds:
        proc = subprocess.run(cmd, cwd=cwd)
        if proc.returncode:
            return proc.returncode

    rows: list[dict] = []
    rows += run_json_lines(["cargo", "run", "--release", "--bin", "fp16_semantics"], root / "languages" / "rust")
    rows += run_json_lines([str(root / "build" / "cpp" / "simd_lab_cpp_fp16_semantics")], root)
    rows += run_json_lines(["zig", "build", "fp16-semantics", "-Doptimize=ReleaseFast"], root / "languages" / "zig")

    corpus = json.loads(
        (root / "data" / "fp16-edge-corpus.json").read_text()
    )
    corpus_length = len(corpus["cases"])
    active, invalid_results = collect_available_strategies(rows, corpus_length)
    names = sorted(active)
    if not names:
        result = {
            "schema": "simd-lab-fp16-semantics-v1",
            "strategies": rows,
            "comparisons": [],
        }
        if invalid_results:
            result["invalid_results"] = invalid_results
        print(json.dumps(result, indent=2))
        return 1 if invalid_results else 0

    comparisons = []
    for i in range(corpus_length):
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
    if invalid_results:
        result["invalid_results"] = invalid_results
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if invalid_results else 0


if __name__ == "__main__":
    raise SystemExit(main())
