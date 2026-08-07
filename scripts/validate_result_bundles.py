#!/usr/bin/env python3
"""Validate authoritative result bundles against the versioned JSON Schema."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

try:
    from jsonschema.exceptions import SchemaError
    from jsonschema.validators import validator_for
except ImportError as exc:  # pragma: no cover - exercised by CI dependency setup.
    print(
        "validate_result_bundles: jsonschema is required (install jsonschema==4.25.1)",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA = ROOT / "schema" / "result-bundle-v1.schema.json"
DEFAULT_EXAMPLE = ROOT / "schema" / "example-result-v1.json"
DEFAULT_RUNS = ROOT / "results" / "runs"

def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant {value!r}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result



def _load_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc


def _default_instances() -> list[Path]:
    paths = [DEFAULT_EXAMPLE]
    if DEFAULT_RUNS.exists():
        paths.extend(sorted(DEFAULT_RUNS.rglob("*.json")))
    return paths


def _json_location(error: Any) -> str:
    parts = [str(part) for part in error.absolute_path]
    return "$" if not parts else "$." + ".".join(parts)


def validate(schema_path: Path, instances: Sequence[Path]) -> list[str]:
    schema = _load_json(schema_path)
    validator_type = validator_for(schema)
    try:
        validator_type.check_schema(schema)
    except SchemaError as exc:
        return [f"{schema_path}: invalid JSON Schema: {exc.message}"]

    validator = validator_type(schema)
    failures: list[str] = []
    for path in instances:
        try:
            instance = _load_json(path)
        except ValueError as exc:
            failures.append(str(exc))
            continue
        for error in sorted(
            validator.iter_errors(instance),
            key=lambda item: tuple(str(part) for part in item.absolute_path),
        ):
            failures.append(f"{path}:{_json_location(error)}: {error.message}")
    return failures


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate result bundles against result-bundle-v1.schema.json"
    )
    parser.add_argument("paths", nargs="*", type=Path, help="bundle paths")
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    instances = args.paths or _default_instances()
    failures = validate(args.schema, instances)
    if failures:
        for failure in failures:
            print(f"validate_result_bundles: {failure}", file=sys.stderr)
        return 1
    print(f"validated {len(instances)} result bundle(s) against {args.schema}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
