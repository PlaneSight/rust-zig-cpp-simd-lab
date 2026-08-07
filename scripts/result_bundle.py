#!/usr/bin/env python3
"""Shared helpers for producing ``simd-lab-result-v1`` bundles.

The collector scripts keep their measurement-specific parsing separate from
this module.  This module owns the common envelope, command-line metadata,
and artifact bookkeeping so every producer emits the same shape.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "PlaneSight/rust-zig-cpp-simd-lab"
_VIRTUALIZED = {"yes": True, "no": False, "unknown": None}
_LANGUAGES = ("rust", "zig", "cpp", "other")


def _strict_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant {value!r}")


def _parse_parameter_spec(spec: str) -> tuple[str, Any]:
    """Parse one ``KEY=JSON`` parameter specification.

    ``json.loads`` is deliberately used instead of treating the right-hand
    side as text: parameter values may be numbers, strings, arrays, objects,
    booleans, or null.  Rejecting non-standard constants keeps the emitted
    document valid JSON rather than accepting Python's NaN/Infinity spellings.
    """

    if not isinstance(spec, str):
        raise ValueError("parameter must be KEY=JSON")
    key, separator, encoded = spec.partition("=")
    key = key.strip()
    if not separator or not key:
        raise ValueError("parameter must use a non-empty KEY=JSON form")
    try:
        value = json.loads(encoded, parse_constant=_strict_json_constant)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid JSON for parameter {key!r}") from exc
    return key, value


class _ParameterAction(argparse.Action):
    """Argparse action that parses and de-duplicates ``--parameter`` values."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str,
        option_string: str | None = None,
    ) -> None:
        try:
            key, value = _parse_parameter_spec(values)
        except ValueError as exc:
            parser.error(str(exc))
            return

        current = getattr(namespace, self.dest, None)
        if current is None:
            current = {}
        if not isinstance(current, dict):
            parser.error("internal error: --parameter destination is not a mapping")
            return
        if key in current:
            parser.error(f"duplicate parameter key: {key!r}")
        current = dict(current)
        current[key] = value
        setattr(namespace, self.dest, current)


def _positive_lanes(value: str) -> int:
    try:
        lanes = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("lanes must be an integer") from exc
    if lanes < 1:
        raise argparse.ArgumentTypeError("lanes must be at least 1")
    return lanes


def add_result_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the common result-bundle options to *parser* and return it."""

    parser.add_argument("--id", required=True, help="stable identifier for this result bundle")
    parser.add_argument("--family", required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--implementation", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--cpu", required=True)

    parser.add_argument("--variant")
    parser.add_argument("--data-type", dest="data_type")
    parser.add_argument("--lanes", type=_positive_lanes)
    parser.add_argument("--isa")
    parser.add_argument("--semantics")
    parser.add_argument("--dataset")
    parser.add_argument(
        "--parameter",
        dest="parameters",
        action=_ParameterAction,
        default=None,
        metavar="KEY=JSON",
        help="repeatable experiment parameter (JSON value)",
    )

    parser.add_argument("--source", dest="source_files", action="append", default=[])
    parser.add_argument("--cpu-feature", dest="cpu_features", action="append", default=[])
    parser.add_argument("--runner")
    parser.add_argument("--virtualized", choices=tuple(_VIRTUALIZED), default="unknown")

    parser.add_argument("--language", choices=_LANGUAGES)
    parser.add_argument("--compiler")
    parser.add_argument("--compiler-version", dest="compiler_version")
    parser.add_argument(
        "--compiler-flag",
        dest="compiler_flags",
        action="append",
        default=[],
    )
    parser.add_argument("--optimization")
    parser.add_argument("--notes")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--tag", dest="tags", action="append", default=[])
    return parser


def paths_alias(first: str | os.PathLike[str], second: str | os.PathLike[str]) -> bool:
    """Return whether two paths resolve to the same existing or future file."""

    first_path = Path(first)
    second_path = Path(second)
    if first_path.resolve() == second_path.resolve():
        return True
    try:
        return os.path.samefile(first_path, second_path)
    except OSError:
        return False


def validate_artifact_output_paths(
    raw_output: str | os.PathLike[str],
    result_output: str | os.PathLike[str] | None,
    *,
    tool: str,
) -> None:
    """Enforce non-aliasing and repository persistence paths for raw evidence."""

    raw_path = Path(raw_output)
    if result_output is not None and paths_alias(raw_path, result_output):
        raise ValueError(f"{tool} raw output and result output must be different files")

    raw_resolved = raw_path.resolve()
    runs_dir = (ROOT / "results" / "runs").resolve()
    artifacts_dir = (ROOT / "results" / "artifacts").resolve()
    if raw_resolved.is_relative_to(runs_dir):
        raise ValueError(f"{tool} raw output must not be stored under results/runs")

    if result_output is None:
        return
    result_resolved = Path(result_output).resolve()
    if result_resolved.is_relative_to(runs_dir) and not raw_resolved.is_relative_to(
        artifacts_dir
    ):
        raise ValueError(
            f"{tool} result bundles under results/runs require raw output "
            "under results/artifacts"
        )


def artifact_record(
    path: str | os.PathLike[str],
    kind: str,
    description: str | None,
) -> dict[str, Any]:
    """Return a schema-shaped artifact record with a SHA-256 content hash."""

    artifact_path = Path(path)
    digest = hashlib.sha256()
    with artifact_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    record = {
        "kind": kind,
        "path": str(path),
        "sha256": digest.hexdigest(),
    }
    if description is not None:
        record["description"] = description
    return record


def _git_value(*arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def _first_attr(args: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if hasattr(args, name):
            value = getattr(args, name)
            if value is not None:
                return value
    return default


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, Path)):
        return [value]
    return list(value)


def _parameter_mapping(value: Any) -> dict[str, Any]:
    """Normalize parsed or direct-call parameters and reject bad keys."""

    if value is None:
        return {}
    if isinstance(value, Mapping):
        items = value.items()
    else:
        if isinstance(value, (str, bytes)):
            value = [value]
        try:
            items = ((item, None) for item in value)
        except TypeError as exc:
            raise ValueError("parameters must be a mapping or KEY=JSON iterable") from exc

    result: dict[str, Any] = {}
    for item, supplied_value in items:
        if supplied_value is None and isinstance(item, str) and "=" in item:
            key, parsed_value = _parse_parameter_spec(item)
        elif supplied_value is not None or isinstance(value, Mapping):
            key = item
            parsed_value = supplied_value
        else:
            raise ValueError("parameter must be KEY=JSON")
        if not isinstance(key, str) or not key.strip():
            raise ValueError("parameter keys must be non-empty strings")
        key = key.strip()
        if key in result:
            raise ValueError(f"duplicate parameter key: {key!r}")
        result[key] = parsed_value
    return result


def _merge_parameters(
    args: Any,
    explicit: Mapping[str, Any] | Iterable[str] | None,
) -> dict[str, Any]:
    cli_value = _first_attr(args, "parameters", "parameter", default=None)
    merged = _parameter_mapping(cli_value)
    if explicit is None:
        return merged
    for key, value in _parameter_mapping(explicit).items():
        if key in merged:
            raise ValueError(f"duplicate parameter key: {key!r}")
        merged[key] = value
    return merged


def _virtualized_value(value: Any) -> bool | None:
    if isinstance(value, bool) or value is None:
        return value
    try:
        return _VIRTUALIZED[str(value).lower()]
    except KeyError as exc:
        raise ValueError("virtualized must be one of yes, no, or unknown") from exc


def _host_os() -> str:
    system = platform.system()
    release = platform.release()
    value = " ".join(part for part in (system, release) if part)
    return value or platform.platform()


def _required(args: Any, name: str) -> Any:
    value = getattr(args, name, None)
    if value is None or value == "":
        raise ValueError(f"missing required result argument: --{name.replace('_', '-')}")
    return value


def build_result_bundle(
    args: Any,
    observations: Iterable[Mapping[str, Any]],
    *,
    parameters: Mapping[str, Any] | Iterable[str] | None = None,
    artifacts: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a ``simd-lab-result-v1`` document from common CLI arguments."""

    observation_list = list(observations)
    if not observation_list:
        raise ValueError("a result bundle requires at least one observation")

    merged_parameters = _merge_parameters(args, parameters)
    lanes = getattr(args, "lanes", None)
    if lanes is not None:
        try:
            lanes = int(lanes)
        except (TypeError, ValueError) as exc:
            raise ValueError("lanes must be an integer") from exc
        if lanes < 1:
            raise ValueError("lanes must be at least 1")

    commit = _git_value("rev-parse", "HEAD") or os.environ.get("GITHUB_SHA") or "unknown"
    branch = _git_value("branch", "--show-current")
    if branch is None:
        branch = os.environ.get("GITHUB_REF_NAME") or os.environ.get("GITHUB_HEAD_REF")
    workflow_run = os.environ.get("GITHUB_RUN_ID")

    source_files = [
        str(item)
        for item in _as_list(_first_attr(args, "source_files", "source", default=[]))
    ]
    cpu_features = [
        str(item)
        for item in _as_list(_first_attr(args, "cpu_features", "cpu_feature", default=[]))
    ]
    tags = [
        str(item)
        for item in _as_list(_first_attr(args, "tags", "tag", default=[]))
    ]

    toolchain_values: dict[str, Any] = {}
    toolchain_sources = (
        ("language", getattr(args, "language", None)),
        ("compiler", getattr(args, "compiler", None)),
        ("version", _first_attr(args, "compiler_version", "version", default=None)),
        ("optimization", getattr(args, "optimization", None)),
        (
            "flags",
            _as_list(
                _first_attr(
                    args,
                    "compiler_flags",
                    "compiler_flag",
                    "flags",
                    default=[],
                )
            ),
        ),
    )
    for key, value in toolchain_sources:
        if value is None or value == "" or value == []:
            continue
        toolchain_values[key] = _as_list(value) if key == "flags" else value
    created_at = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )

    document: dict[str, Any] = {
        "schema": "simd-lab-result-v1",
        "id": _required(args, "id"),
        "created_at": created_at,
        "provenance": {
            "repository": REPOSITORY,
            "commit": commit,
            "branch": branch,
            "workflow_run": workflow_run,
            "source_files": source_files,
        },
        "environment": {
            "os": _host_os(),
            "arch": platform.machine(),
            "cpu": _required(args, "cpu"),
            "cpu_features": cpu_features,
            "target": _required(args, "target"),
            "runner": getattr(args, "runner", None),
            "virtualized": _virtualized_value(getattr(args, "virtualized", "unknown")),
        },
        **({"toolchain": toolchain_values} if toolchain_values else {}),
        "experiment": {
            "family": _required(args, "family"),
            "kernel": _required(args, "kernel"),
            "implementation": _required(args, "implementation"),
            "variant": getattr(args, "variant", None),
            "data_type": _first_attr(args, "data_type", "data_type_name", default=None),
            "lanes": lanes,
            "isa": getattr(args, "isa", None),
            "semantics": getattr(args, "semantics", None),
            "dataset": getattr(args, "dataset", None),
            "parameters": merged_parameters,
        },
        "observations": observation_list,
        "artifacts": list(artifacts) if artifacts is not None else [],
        "tags": tags,
        "notes": getattr(args, "notes", None),
    }

    return document


def write_result(document: Mapping[str, Any], args: Any) -> None:
    indent = 2 if bool(getattr(args, "pretty", False)) else None
    rendered = json.dumps(document, indent=indent, allow_nan=False) + "\n"

    output = getattr(args, "output", None)
    if output is None:
        sys.stdout.write(rendered)
        return

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")


__all__ = [
    "add_result_arguments",
    "artifact_record",
    "build_result_bundle",
    "paths_alias",
    "validate_artifact_output_paths",
    "write_result",
]
