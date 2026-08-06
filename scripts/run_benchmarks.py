#!/usr/bin/env python3
"""Run SIMD lab benchmarks and emit one machine-readable JSON document.

The language benchmark binaries intentionally keep a human-readable output for
interactive use. This runner normalizes that output without coupling the three
implementations to a shared benchmark library.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

RESULT_RE = re.compile(
    r"^(?P<name>\S(?:.*?\S)?)\s+(?P<ns>[0-9]+(?:\.[0-9]+)?)\s+ns/elem\s+"
    r"(?P<gib>[0-9]+(?:\.[0-9]+)?)\s+GiB/s$"
)
META_RE = re.compile(r"^N=(?P<n>\d+)\s+warmup=(?P<warmup>\d+)\s+iterations=(?P<iterations>\d+)$")


def run(cmd: list[str], cwd: Path) -> tuple[str, str, int]:
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    return proc.stdout, proc.stderr, proc.returncode


def version(cmd: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)
    except FileNotFoundError:
        return None
    text = (out.stdout or out.stderr).strip().splitlines()
    return text[0] if text else None


def parse_output(language: str, text: str) -> dict:
    results: list[dict] = []
    meta: dict[str, int] = {}
    skipped: list[str] = []

    for raw in text.splitlines():
        line = raw.strip()
        m = RESULT_RE.match(line)
        if m:
            results.append(
                {
                    "name": m.group("name"),
                    "ns_per_element": float(m.group("ns")),
                    "gib_per_second": float(m.group("gib")),
                }
            )
            continue
        m = META_RE.match(line)
        if m:
            meta = {k: int(v) for k, v in m.groupdict().items()}
            continue
        if "skipped" in line.lower():
            skipped.append(line)

    return {"language": language, "metadata": meta, "results": results, "skipped": skipped}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()

    commands = {
        "rust": (["cargo", "run", "--release", "--bin", "bench"], root / "rust"),
        "cpp23": ([str(root / "build" / "cpp" / "simd_lab_cpp_bench")], root),
        "zig": (["zig", "build", "bench", "-Doptimize=ReleaseFast"], root / "zig"),
    }

    if not args.no_build:
        builds = [
            (["cargo", "build", "--release", "--bin", "bench"], root / "rust"),
            (["cmake", "-S", "cpp", "-B", "build/cpp", "-DCMAKE_BUILD_TYPE=Release"], root),
            (["cmake", "--build", "build/cpp", "--target", "simd_lab_cpp_bench", "-j2"], root),
            (["zig", "build", "-Doptimize=ReleaseFast"], root / "zig"),
        ]
        for cmd, cwd in builds:
            stdout, stderr, rc = run(cmd, cwd)
            if rc:
                sys.stderr.write(stdout)
                sys.stderr.write(stderr)
                return rc

    runs: list[dict] = []
    for language, (cmd, cwd) in commands.items():
        stdout, stderr, rc = run(cmd, cwd)
        # Zig writes std.debug.print to stderr.
        text = stdout + ("\n" if stdout and stderr else "") + stderr
        if rc:
            sys.stderr.write(text)
            return rc
        runs.append(parse_output(language, text))

    document = {
        "schema": "simd-lab-benchmark-v1",
        "timestamp_unix": int(time.time()),
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "toolchains": {
            "rustc": version(["rustc", "--version"], root),
            "cargo": version(["cargo", "--version"], root),
            "zig": version(["zig", "version"], root),
            "clang++": version(["clang++", "--version"], root),
            "g++": version(["g++", "--version"], root),
            "cmake": version(["cmake", "--version"], root),
        },
        "runs": runs,
    }

    rendered = json.dumps(document, indent=2 if args.pretty else None, sort_keys=args.pretty)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
