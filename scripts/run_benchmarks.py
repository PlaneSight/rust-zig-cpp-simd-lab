#!/usr/bin/env python3
"""Build, run, and normalize the cross-language benchmark protocol."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

CPU_TIERS: Final = ("baseline", "native", "x86-64-v3", "avx2")
INTEGER_FIELDS: Final = {
    "n",
    "working_set_bytes",
    "effective_bytes_per_iteration",
    "iterations_per_sample",
    "sample_count",
}
FLOAT_FIELDS: Final = {
    "min_ns_per_element",
    "median_ns_per_element",
    "p95_ns_per_element",
    "mad_ns_per_element",
    "median_gib_per_second",
}


@dataclass(frozen=True)
class Command:
    argv: list[str]
    cwd: Path


@dataclass(frozen=True)
class TargetTier:
    name: str
    rustflags: str | None
    cmake_value: str
    zig_args: tuple[str, ...]

    @classmethod
    def from_name(cls, name: str) -> "TargetTier":
        rust_cpu = {
            "baseline": None,
            "native": "native",
            "x86-64-v3": "x86-64-v3",
            "avx2": "x86-64-v3",
        }[name]
        zig_cpu = {
            "baseline": (),
            "native": ("-Dcpu=native",),
            "x86-64-v3": ("-Dcpu=x86_64_v3",),
            "avx2": ("-Dcpu=x86_64_v3",),
        }[name]
        rustflags = None if rust_cpu is None else f"-C target-cpu={rust_cpu}"
        return cls(name=name, rustflags=rustflags, cmake_value=name, zig_args=zig_cpu)

    def environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment["SIMD_LAB_CPU_TIER"] = self.name
        if self.rustflags:
            existing = environment.get("RUSTFLAGS", "").strip()
            environment["RUSTFLAGS"] = " ".join(
                value for value in (existing, self.rustflags) if value
            )
        return environment


def run(command: Command, environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command.argv,
            cwd=command.cwd,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"required executable not found: {command.argv[0]}") from exc


def version(argv: list[str], cwd: Path, environment: dict[str, str]) -> str | None:
    try:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    lines = (completed.stdout or completed.stderr).strip().splitlines()
    return lines[0] if lines else None


def fields_from(line: str, prefix: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in shlex.split(line.removeprefix(prefix).strip()):
        if "=" not in token:
            raise ValueError(f"malformed protocol token {token!r} in {line!r}")
        key, value = token.split("=", 1)
        fields[key] = value
    return fields


def parse_result(line: str) -> dict[str, object]:
    fields = fields_from(line, "RESULT")
    missing = {"name", "raw_ns_per_element"} | INTEGER_FIELDS | FLOAT_FIELDS
    if absent := missing.difference(fields):
        raise ValueError(f"benchmark result is missing fields: {sorted(absent)}")

    samples = [float(value) for value in fields.pop("raw_ns_per_element").split(",")]
    result: dict[str, object] = {"name": fields.pop("name")}
    for key in INTEGER_FIELDS:
        result[key] = int(fields.pop(key))
    result["statistics"] = {key: float(fields.pop(key)) for key in sorted(FLOAT_FIELDS)}
    result["samples"] = {"unit": "ns/element", "values": samples}
    if len(samples) != result["sample_count"]:
        raise ValueError(
            f"{result['name']}: declared {result['sample_count']} samples, got {len(samples)}"
        )
    if fields:
        result["extra"] = fields
    return result


def parse_metadata(line: str) -> dict[str, object]:
    metadata: dict[str, object] = {}
    for key, value in fields_from(line, "META").items():
        metadata[key] = int(value) if value.isdecimal() else value
    return metadata


def parse_output(language: str, text: str) -> dict[str, object]:
    results: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []
    metadata: dict[str, object] = {}

    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("RESULT "):
            results.append(parse_result(line))
        elif line.startswith("SKIP "):
            skipped.append(fields_from(line, "SKIP"))
        elif line.startswith("META "):
            metadata.update(parse_metadata(line))

    if not results:
        raise ValueError(f"{language} benchmark produced no RESULT records")
    return {
        "language": language,
        "metadata": metadata,
        "results": results,
        "skipped": skipped,
    }


def checked_run(command: Command, environment: dict[str, str]) -> str:
    completed = run(command, environment)
    text = completed.stdout + ("\n" if completed.stdout and completed.stderr else "") + completed.stderr
    if completed.returncode:
        sys.stderr.write(text)
        rendered = shlex.join(command.argv)
        raise RuntimeError(f"command failed with status {completed.returncode}: {rendered}")
    return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--cpu", choices=CPU_TIERS, default="baseline")
    args = parser.parse_args()

    root = args.root.resolve()
    tier = TargetTier.from_name(args.cpu)
    environment = tier.environment()
    zig_release_args = ["-Doptimize=ReleaseFast", *tier.zig_args]

    builds = [
        Command(["cargo", "build", "--release", "--bin", "bench"], root / "rust"),
        Command(
            [
                "cmake",
                "-S",
                "cpp",
                "-B",
                "build/cpp",
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DSIMD_LAB_CPU={tier.cmake_value}",
            ],
            root,
        ),
        Command(
            ["cmake", "--build", "build/cpp", "--target", "simd_lab_cpp_bench", "-j2"],
            root,
        ),
        Command(["zig", "build", *zig_release_args], root / "zig"),
    ]
    commands = {
        "rust": Command(["cargo", "run", "--release", "--bin", "bench"], root / "rust"),
        "cpp23": Command([str(root / "build" / "cpp" / "simd_lab_cpp_bench")], root),
        "zig": Command(["zig", "build", "bench", *zig_release_args], root / "zig"),
    }

    try:
        if not args.no_build:
            for command in builds:
                checked_run(command, environment)

        runs = [
            parse_output(language, checked_run(command, environment))
            for language, command in commands.items()
        ]
    except (RuntimeError, ValueError) as exc:
        print(f"benchmark collection failed: {exc}", file=sys.stderr)
        return 1

    document = {
        "schema": "simd-lab-benchmark-v2",
        "timestamp_unix": int(time.time()),
        "protocol": {
            "target_tier": tier.name,
            "aggregation": "median",
            "spread": "median_absolute_deviation",
            "tail_percentile": 95,
            "raw_samples_preserved": True,
        },
        "target_configuration": {
            "rustflags": tier.rustflags,
            "cmake_simd_lab_cpu": tier.cmake_value,
            "zig_args": list(tier.zig_args),
        },
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "toolchains": {
            "rustc": version(["rustc", "--version"], root, environment),
            "cargo": version(["cargo", "--version"], root, environment),
            "zig": version(["zig", "version"], root, environment),
            "clang++": version(["clang++", "--version"], root, environment),
            "g++": version(["g++", "--version"], root, environment),
            "cmake": version(["cmake", "--version"], root, environment),
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
