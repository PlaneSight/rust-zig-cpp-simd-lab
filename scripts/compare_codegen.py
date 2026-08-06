#!/usr/bin/env python3
"""Compare two simd-lab-codegen-v1 manifests without diffing raw assembly."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def key_for(path: str) -> str:
    name = Path(path).name
    parts = name.split("-")
    return "-".join(parts[:2]) if len(parts) >= 2 else name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("before", type=Path)
    ap.add_argument("after", type=Path)
    ap.add_argument("--pretty", action="store_true")
    args = ap.parse_args()

    before = json.loads(args.before.read_text())
    after = json.loads(args.after.read_text())
    if before.get("schema") != "simd-lab-codegen-v1" or after.get("schema") != "simd-lab-codegen-v1":
        raise SystemExit("unsupported manifest schema")

    bmetrics = {key_for(k): v for k, v in before.get("metrics", {}).items()}
    ametrics = {key_for(k): v for k, v in after.get("metrics", {}).items()}
    rows = []
    for key in sorted(set(bmetrics) | set(ametrics)):
        b = bmetrics.get(key)
        a = ametrics.get(key)
        if b is None or a is None:
            rows.append({"probe": key, "status": "added" if b is None else "removed"})
            continue
        bt = b.get("tracked", {})
        at = a.get("tracked", {})
        tracked_delta = {
            name: at.get(name, 0) - bt.get(name, 0)
            for name in sorted(set(bt) | set(at))
            if at.get(name, 0) != bt.get(name, 0)
        }
        rows.append({
            "probe": key,
            "instruction_delta": a["instruction_count"] - b["instruction_count"],
            "vector_instruction_delta": a["vector_instruction_count"] - b["vector_instruction_count"],
            "tracked_delta": tracked_delta,
        })

    output = {
        "schema": "simd-lab-codegen-diff-v1",
        "before_target": before.get("target_profile"),
        "after_target": after.get("target_profile"),
        "comparisons": rows,
    }
    print(json.dumps(output, indent=2 if args.pretty else None, sort_keys=True))


if __name__ == "__main__":
    main()
