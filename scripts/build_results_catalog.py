#!/usr/bin/env python3
"""Build a UI-oriented catalog from authoritative result bundles.

Only `simd-lab-result-v1` files under results/runs are authoritative. The
catalog is derived output and can be regenerated at any time.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = ROOT / "results" / "runs"
DEFAULT_OUT = ROOT / "site" / "data" / "catalog.json"


def load_bundle(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "simd-lab-result-v1":
        raise ValueError(f"{path}: unsupported schema {data.get('schema')!r}")
    for key in ("id", "created_at", "provenance", "environment", "experiment", "observations"):
        if key not in data:
            raise ValueError(f"{path}: missing required key {key!r}")
    return data


def metric_map(observations: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for obs in observations:
        if obs.get("kind") != kind:
            continue
        for metric in obs.get("metrics", []):
            out[metric["name"]] = {
                "value": metric["value"],
                "unit": metric.get("unit"),
                "aggregation": metric.get("aggregation"),
                "lower": metric.get("lower"),
                "upper": metric.get("upper"),
                "samples": metric.get("samples"),
            }
    return out


def summarize(path: Path, bundle: dict[str, Any]) -> dict[str, Any]:
    observations = bundle["observations"]
    codegen = next((x for x in observations if x.get("kind") == "codegen"), None)
    semantics = next((x for x in observations if x.get("kind") == "semantics"), None)
    analyses = [x for x in observations if x.get("kind") == "analysis"]
    counter_observations = [x for x in observations if x.get("kind") == "counters"]
    toolchain = bundle.get("toolchain") or {}
    experiment = bundle["experiment"]
    environment = bundle["environment"]
    resolved_path = path.resolve()
    try:
        source = str(resolved_path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        source = str(resolved_path).replace("\\", "/")

    return {
        "id": bundle["id"],
        "created_at": bundle["created_at"],
        "source": source,
        "commit": bundle["provenance"]["commit"],
        "language": toolchain.get("language"),
        "compiler": toolchain.get("compiler"),
        "compiler_version": toolchain.get("version"),
        "optimization": toolchain.get("optimization"),
        "target": environment.get("target"),
        "arch": environment.get("arch"),
        "cpu": environment.get("cpu"),
        "family": experiment.get("family"),
        "kernel": experiment.get("kernel"),
        "implementation": experiment.get("implementation"),
        "variant": experiment.get("variant"),
        "data_type": experiment.get("data_type"),
        "isa": experiment.get("isa"),
        "semantics_contract": experiment.get("semantics"),
        "dataset": experiment.get("dataset"),
        "runtime": metric_map(observations, "runtime"),
        "counters": metric_map(observations, "counters"),
        "counter_observations": counter_observations,
        "codegen": None if not codegen else {
            "instruction_count": codegen.get("instruction_count"),
            "vector_instruction_count": codegen.get("vector_instruction_count"),
            "unique_mnemonics": codegen.get("unique_mnemonics"),
            "tracked_mnemonics": codegen.get("tracked_mnemonics", {}),
            "assembly_path": codegen.get("assembly_path"),
            "ir_path": codegen.get("ir_path"),
        },
        "semantics": None if not semantics else {
            "status": semantics.get("status"),
            "reference": semantics.get("reference"),
            "cases": semantics.get("cases"),
            "mismatches": semantics.get("mismatches"),
            "details_path": semantics.get("details_path"),
            "summary": semantics.get("summary"),
        },
        "artifacts": bundle.get("artifacts", []),
        "analysis": analyses,
        "tags": bundle.get("tags", []),
        "notes": bundle.get("notes"),
    }


def values(rows: list[dict[str, Any]], key: str) -> list[str]:
    return sorted({str(row[key]) for row in rows if row.get(key) not in (None, "")})


def build(runs_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    if runs_dir.exists():
        for path in sorted(runs_dir.rglob("*.json")):
            try:
                rows.append(summarize(path, load_bundle(path)))
            except Exception as exc:
                errors.append({"path": str(path), "error": str(exc)})

    kinds = Counter()
    for row in rows:
        if row["runtime"]:
            kinds["runtime"] += 1
        if row["codegen"]:
            kinds["codegen"] += 1
        if row["semantics"]:
            kinds["semantics"] += 1
        if row["counter_observations"]:
            kinds["counters"] += 1
        if row["analysis"]:
            kinds["analysis"] += 1

    return {
        "schema": "simd-lab-catalog-v1",
        "result_schema": "simd-lab-result-v1",
        "count": len(rows),
        "errors": errors,
        "facets": {
            key: values(rows, key)
            for key in (
                "language",
                "compiler",
                "target",
                "arch",
                "family",
                "kernel",
                "implementation",
                "data_type",
                "isa",
                "dataset",
            )
        },
        "observation_counts": dict(sorted(kinds.items())),
        "results": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    catalog = build(args.runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({catalog['count']} results, {len(catalog['errors'])} errors)")
    if catalog["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
