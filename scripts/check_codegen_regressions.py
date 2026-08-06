#!/usr/bin/env python3
"""Check a codegen manifest against a known-good baseline using loose policy.

The goal is to catch compiler/code changes that cause large instruction-count or
high-signal mnemonic regressions without requiring exact assembly equality.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_key(path: str) -> str:
    return Path(path).name


def matching_rule(policy: dict, key: str) -> dict | None:
    for rule in policy.get("rules", []):
        if rule.get("match", "") in key:
            return rule
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--policy", type=Path, default=Path("codegen-policy.json"))
    args = ap.parse_args()

    base = load(args.baseline)
    cand = load(args.candidate)
    policy = load(args.policy)
    defaults = policy["defaults"]

    base_metrics = {metric_key(k): v for k, v in base.get("metrics", {}).items()}
    cand_metrics = {metric_key(k): v for k, v in cand.get("metrics", {}).items()}
    failures: list[dict] = []
    observations: list[dict] = []

    for key, new in sorted(cand_metrics.items()):
        old = base_metrics.get(key)
        if old is None:
            observations.append({"file": key, "status": "new"})
            continue

        rule = matching_rule(policy, key) or {}
        max_i = rule.get("max_instruction_growth_ratio", defaults["max_instruction_growth_ratio"])
        max_v = rule.get("max_vector_instruction_growth_ratio", defaults["max_vector_instruction_growth_ratio"])

        old_i = max(int(old.get("instruction_count", 0)), 1)
        new_i = int(new.get("instruction_count", 0))
        old_v = max(int(old.get("vector_instruction_count", 0)), 1)
        new_v = int(new.get("vector_instruction_count", 0))

        item = {
            "file": key,
            "instruction_ratio": new_i / old_i,
            "vector_instruction_ratio": new_v / old_v,
            "tracked_deltas": {},
        }

        if new_i / old_i > max_i:
            failures.append({"file": key, "kind": "instruction_growth", "old": old_i, "new": new_i, "limit_ratio": max_i})
        if new_v / old_v > max_v:
            failures.append({"file": key, "kind": "vector_instruction_growth", "old": old_v, "new": new_v, "limit_ratio": max_v})

        old_t = old.get("tracked", {})
        new_t = new.get("tracked", {})
        for mnemonic, allowed_growth in rule.get("tracked_max_growth", {}).items():
            delta = int(new_t.get(mnemonic, 0)) - int(old_t.get(mnemonic, 0))
            item["tracked_deltas"][mnemonic] = delta
            if delta > int(allowed_growth):
                failures.append({
                    "file": key,
                    "kind": "tracked_mnemonic_growth",
                    "mnemonic": mnemonic,
                    "delta": delta,
                    "allowed_growth": allowed_growth,
                })
        observations.append(item)

    result = {
        "schema": "simd-lab-codegen-regression-v1",
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "failures": failures,
        "observations": observations,
    }
    print(json.dumps(result, indent=2))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
