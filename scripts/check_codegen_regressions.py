#!/usr/bin/env python3
"""Check a codegen manifest against a known-good baseline using loose policy.

The goal is to catch compiler/code changes that cause large instruction-count or
high-signal mnemonic regressions without requiring exact assembly equality.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

MANIFEST_SCHEMA = "simd-lab-codegen-v1"
POLICY_SCHEMA = "simd-lab-codegen-policy-v1"
RESULT_SCHEMA = "simd-lab-codegen-regression-v1"


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")
    return value


def metric_key(path: str) -> str:
    return path.replace("\\", "/").rsplit("/", 1)[-1]


def non_negative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def positive_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{field} must be a positive number")
    return float(value)


def validate_manifest(manifest: dict, label: str) -> tuple[str, dict[str, dict]]:
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"{label}: schema must be {MANIFEST_SCHEMA!r}")

    target_profile = manifest.get("target_profile")
    if not isinstance(target_profile, str) or not target_profile:
        raise ValueError(f"{label}: target_profile must be a non-empty string")

    raw_metrics = manifest.get("metrics")
    if not isinstance(raw_metrics, dict) or not raw_metrics:
        raise ValueError(f"{label}: metrics must be a non-empty object")

    metrics: dict[str, dict] = {}
    for path, raw_metric in raw_metrics.items():
        if not isinstance(path, str) or not path:
            raise ValueError(f"{label}: metric paths must be non-empty strings")
        if not isinstance(raw_metric, dict):
            raise ValueError(f"{label}: metric {path!r} must be an object")

        key = metric_key(path)
        if not key:
            raise ValueError(f"{label}: metric path {path!r} has no file name")
        if key in metrics:
            raise ValueError(f"{label}: duplicate normalized metric key {key!r}")

        non_negative_int(
            raw_metric.get("instruction_count"),
            f"{label}: metric {key!r} instruction_count",
        )
        non_negative_int(
            raw_metric.get("vector_instruction_count"),
            f"{label}: metric {key!r} vector_instruction_count",
        )

        tracked = raw_metric.get("tracked", {})
        if not isinstance(tracked, dict):
            raise ValueError(f"{label}: metric {key!r} tracked must be an object")
        for mnemonic, count in tracked.items():
            if not isinstance(mnemonic, str) or not mnemonic:
                raise ValueError(
                    f"{label}: metric {key!r} tracked names must be non-empty strings"
                )
            non_negative_int(
                count,
                f"{label}: metric {key!r} tracked {mnemonic!r}",
            )

        metrics[key] = raw_metric

    return target_profile, metrics


def validate_policy(policy: dict) -> None:
    if policy.get("schema") != POLICY_SCHEMA:
        raise ValueError(f"policy: schema must be {POLICY_SCHEMA!r}")

    defaults = policy.get("defaults")
    if not isinstance(defaults, dict):
        raise ValueError("policy: defaults must be an object")
    for field in (
        "max_instruction_growth_ratio",
        "max_vector_instruction_growth_ratio",
    ):
        positive_number(defaults.get(field), f"policy: defaults.{field}")

    rules = policy.get("rules", [])
    if not isinstance(rules, list):
        raise ValueError("policy: rules must be an array")
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise ValueError(f"policy: rules[{index}] must be an object")
        match = rule.get("match")
        if not isinstance(match, str) or not match:
            raise ValueError(f"policy: rules[{index}].match must be a non-empty string")
        for field in (
            "max_instruction_growth_ratio",
            "max_vector_instruction_growth_ratio",
        ):
            if field in rule:
                positive_number(rule[field], f"policy: rules[{index}].{field}")
        tracked_growth = rule.get("tracked_max_growth", {})
        if not isinstance(tracked_growth, dict):
            raise ValueError(
                f"policy: rules[{index}].tracked_max_growth must be an object"
            )
        for mnemonic, growth in tracked_growth.items():
            if not isinstance(mnemonic, str) or not mnemonic:
                raise ValueError(
                    f"policy: rules[{index}] tracked names must be non-empty strings"
                )
            non_negative_int(
                growth,
                f"policy: rules[{index}].tracked_max_growth[{mnemonic!r}]",
            )


def matching_rule(policy: dict, key: str) -> dict | None:
    for rule in policy.get("rules", []):
        if rule["match"] in key:
            return rule
    return None


def compare_manifests(baseline: dict, candidate: dict, policy: dict) -> dict:
    validate_policy(policy)
    base_profile, base_metrics = validate_manifest(baseline, "baseline")
    cand_profile, cand_metrics = validate_manifest(candidate, "candidate")
    if cand_profile != base_profile:
        raise ValueError(
            "candidate: target_profile "
            f"{cand_profile!r} does not match baseline {base_profile!r}"
        )

    defaults = policy["defaults"]
    failures: list[dict] = []
    observations: list[dict] = []

    for key in sorted(base_metrics.keys() - cand_metrics.keys()):
        failures.append({"file": key, "kind": "missing_metric"})

    for key, new in sorted(cand_metrics.items()):
        old = base_metrics.get(key)
        if old is None:
            observations.append({"file": key, "status": "new"})
            continue

        rule = matching_rule(policy, key) or {}
        max_i = rule.get(
            "max_instruction_growth_ratio",
            defaults["max_instruction_growth_ratio"],
        )
        max_v = rule.get(
            "max_vector_instruction_growth_ratio",
            defaults["max_vector_instruction_growth_ratio"],
        )

        old_i = max(old["instruction_count"], 1)
        new_i = new["instruction_count"]
        old_v = max(old["vector_instruction_count"], 1)
        new_v = new["vector_instruction_count"]

        item = {
            "file": key,
            "instruction_ratio": new_i / old_i,
            "vector_instruction_ratio": new_v / old_v,
            "tracked_deltas": {},
        }

        if new_i / old_i > max_i:
            failures.append(
                {
                    "file": key,
                    "kind": "instruction_growth",
                    "old": old_i,
                    "new": new_i,
                    "limit_ratio": max_i,
                }
            )
        if new_v / old_v > max_v:
            failures.append(
                {
                    "file": key,
                    "kind": "vector_instruction_growth",
                    "old": old_v,
                    "new": new_v,
                    "limit_ratio": max_v,
                }
            )

        old_tracked = old.get("tracked", {})
        new_tracked = new.get("tracked", {})
        for mnemonic, allowed_growth in rule.get("tracked_max_growth", {}).items():
            delta = new_tracked.get(mnemonic, 0) - old_tracked.get(mnemonic, 0)
            item["tracked_deltas"][mnemonic] = delta
            if delta > allowed_growth:
                failures.append(
                    {
                        "file": key,
                        "kind": "tracked_mnemonic_growth",
                        "mnemonic": mnemonic,
                        "delta": delta,
                        "allowed_growth": allowed_growth,
                    }
                )
        observations.append(item)

    return {"failures": failures, "observations": observations}


def run(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--policy", type=Path, default=Path("codegen-policy.json"))
    args = ap.parse_args(argv)

    try:
        comparison = compare_manifests(
            load(args.baseline),
            load(args.candidate),
            load(args.policy),
        )
    except (OSError, UnicodeError, ValueError) as exc:
        comparison = {
            "failures": [{"kind": "invalid_input", "message": str(exc)}],
            "observations": [],
        }
        status = 2
    else:
        status = 1 if comparison["failures"] else 0

    result = {
        "schema": RESULT_SCHEMA,
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        **comparison,
    }
    print(json.dumps(result, indent=2))
    return status


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
