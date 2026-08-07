#!/usr/bin/env python3
"""Collect Linux ``perf stat`` counters into a result bundle.

The parser deliberately accepts only perf's JSON-lines counter records.  It
never turns perf's ``<not counted>`` or ``<not supported>`` markers into zeroes,
and the raw output file supplied by the caller remains the source artifact for
all successful and unsuccessful parses.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import subprocess
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:  # Running as ``python scripts/collect_perf_stat.py``.
    from result_bundle import (
        add_result_arguments,
        artifact_record,
        build_result_bundle,
        validate_artifact_output_paths,
        write_result,
    )
except ImportError:  # Imported as ``scripts.collect_perf_stat`` in tests.
    from scripts.result_bundle import (
        add_result_arguments,
        artifact_record,
        build_result_bundle,
        validate_artifact_output_paths,
        write_result,
    )


DEFAULT_EVENTS: tuple[str, ...] = (
    "cycles",
    "instructions",
    "branches",
    "branch-misses",
)
_MAX_COUNTER_VALUE = (1 << 64) - 1

_SENTINEL_VALUES = frozenset({"<not supported>", "<not counted>"})
_NON_WORD_RE = re.compile(r"[^0-9A-Za-z]+")


def _reject_json_constant(value: str) -> None:
    raise PerfParseError(f"invalid JSON constant {value!r}")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PerfParseError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


class PerfCollectionError(RuntimeError):
    """An execution, validation, or result-writing error from perf collection."""


class PerfParseError(PerfCollectionError, ValueError):
    """The perf JSON output is malformed or contains unusable counters."""


def normalize_event_name(event: str) -> str:
    """Return the stable snake_case form used for counter metric names."""

    if not isinstance(event, str):
        raise PerfParseError("perf event name must be a string")
    normalized = _NON_WORD_RE.sub("_", event.strip().lower()).strip("_")
    if not normalized:
        raise PerfParseError("perf event name must not be empty")
    return normalized


def _validated_repeat(repeat: int | str) -> int:
    """Validate and normalize the repeat count used by perf and metric samples."""

    if isinstance(repeat, bool):
        raise PerfParseError("perf repeat must be a positive integer")
    if isinstance(repeat, str):
        value = repeat.strip()
        if not value or not value.isdecimal():
            raise PerfParseError("perf repeat must be a positive integer")
        repeat = int(value)
    elif isinstance(repeat, int):
        repeat = int(repeat)
    else:
        raise PerfParseError("perf repeat must be a positive integer")
    if repeat < 1:
        raise PerfParseError("perf repeat must be a positive integer")
    if repeat > 100:
        raise PerfParseError("perf repeat must not exceed 100")
    return repeat


def _positive_repeat(value: str) -> int:
    """argparse converter for ``--repeat``."""

    try:
        repeat = _validated_repeat(value)
    except PerfParseError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return repeat


def _counter_number(value: Any, *, line_number: int) -> int | float:
    """Convert a perf counter value without rounding integral PMU evidence."""

    if isinstance(value, bool):
        raise PerfParseError(
            f"perf record on line {line_number} has a nonnumeric counter value"
        )
    if isinstance(value, str):
        text = value.strip()
        if text in _SENTINEL_VALUES:
            raise PerfParseError(
                f"perf record on line {line_number} reports {text}"
            )
        if not text:
            raise PerfParseError(
                f"perf record on line {line_number} has an empty counter value"
            )
        try:
            number = Decimal(text)
        except InvalidOperation as exc:
            raise PerfParseError(
                f"perf record on line {line_number} has a nonnumeric counter value"
            ) from exc
    elif isinstance(value, int):
        number = Decimal(value)
    elif isinstance(value, Decimal):
        number = value
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise PerfParseError(
                f"perf record on line {line_number} has a non-finite counter value"
            )
        number = Decimal(str(value))
    else:
        raise PerfParseError(
            f"perf record on line {line_number} has a nonnumeric counter value"
        )

    if not number.is_finite():
        raise PerfParseError(
            f"perf record on line {line_number} has a non-finite counter value"
        )
    if number < 0:
        raise PerfParseError(
            f"perf record on line {line_number} has a negative counter value"
        )
    if number > _MAX_COUNTER_VALUE:
        raise PerfParseError(
            f"perf record on line {line_number} has an out-of-range counter value"
        )

    integral = number.to_integral_value()
    if number == integral:
        return int(integral)

    rounded = float(number)
    if not math.isfinite(rounded) or Decimal(str(rounded)) != number:
        raise PerfParseError(
            f"perf record on line {line_number} loses precision as a JSON number"
        )
    return rounded


def _parse_perf_records(
    text: str, repeat: int | str
) -> tuple[list[dict[str, object]], set[str]]:
    """Parse raw records, returning raw metrics and their normalized event names.

    ``perf stat --json-output`` emits one JSON object per line.  Blank lines are
    harmless, but every non-blank line must be a JSON object containing an event
    and a numeric ``counter-value``.  Optional perf metadata (unit, runtime,
    running percentage, and variance) is intentionally not copied into the
    schema metric: the untouched line is retained in the raw artifact instead.
    """

    repeat_value = _validated_repeat(repeat)
    if not isinstance(text, str):
        raise PerfParseError("perf JSON output must be text")

    metrics: list[dict[str, object]] = []
    event_names: set[str] = set()
    saw_record = False
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        saw_record = True
        try:
            record = json.loads(
                line,
                parse_constant=_reject_json_constant,
                parse_float=Decimal,
                object_pairs_hook=_unique_json_object,
            )
        except json.JSONDecodeError as exc:
            raise PerfParseError(
                f"malformed perf JSON on line {line_number}: {exc.msg}"
            ) from exc
        if not isinstance(record, Mapping):
            raise PerfParseError(
                f"perf JSON record on line {line_number} must be an object"
            )

        if "event" not in record:
            raise PerfParseError(
                f"perf JSON record on line {line_number} is missing event"
            )
        event_name = normalize_event_name(record["event"])
        if event_name in event_names:
            raise PerfParseError(
                f"duplicate perf event in JSON output: {event_name}"
            )
        if "counter-value" not in record:
            raise PerfParseError(
                f"perf JSON record on line {line_number} is missing counter-value"
            )
        counter_value = _counter_number(
            record["counter-value"], line_number=line_number
        )

        unit_value = record.get("unit")
        if unit_value is None or unit_value == "":
            unit = "count"
        elif isinstance(unit_value, str):
            unit = unit_value
        else:
            raise PerfParseError(
                f"perf record on line {line_number} has a non-string unit"
            )

        event_names.add(event_name)
        metrics.append(
            {
                "name": event_name,
                "value": counter_value,
                "unit": unit,
                "aggregation": "mean" if repeat_value > 1 else "single",
                "samples": repeat_value,
            }
        )

    if not saw_record:
        raise PerfParseError("perf JSON output contains no counter records")
    return metrics, event_names


def _add_ipc(
    metrics: Sequence[Mapping[str, object]], repeat: int | str
) -> list[dict[str, object]]:
    """Append instructions/cycles as a ratio of the two means when possible."""

    repeat_value = _validated_repeat(repeat)
    result = [dict(metric) for metric in metrics]
    values = {
        metric.get("name"): metric.get("value")
        for metric in metrics
        if metric.get("name") in {"instructions", "cycles"}
    }
    cycles = values.get("cycles")
    instructions = values.get("instructions")
    if isinstance(cycles, (int, float)) and isinstance(instructions, (int, float)):
        if (
            math.isfinite(float(cycles))
            and math.isfinite(float(instructions))
            and float(cycles) != 0.0
        ):
            result.append(
                {
                    "name": "ipc",
                    "value": float(instructions) / float(cycles),
                    "unit": "instructions/cycle",
                    "aggregation": "ratio-of-means",
                    "samples": repeat_value,
                }
            )
    return result


def parse_perf_stat(text: str, repeat: int | str) -> list[dict[str, object]]:
    """Parse perf JSON-lines output into schema-compatible counter metrics.

    The public parser intentionally has only ``text`` and ``repeat`` inputs:
    event selection is a collection concern, while this function validates all
    records present in the raw output.  It rejects malformed records, duplicate
    normalized event names, sentinels, and nonnumeric values.
    """

    metrics, _ = _parse_perf_records(text, repeat)
    return _add_ipc(metrics, repeat)


def _event_arguments(values: Iterable[str] | None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return event spellings for perf and normalized names for validation."""

    selected = tuple(DEFAULT_EVENTS if values is None else values)
    raw_names: list[str] = []
    normalized: list[str] = []
    seen: set[str] = set()
    for event in selected:
        if not isinstance(event, str) or not event.strip():
            raise PerfCollectionError("perf event name must not be empty")
        raw_name = event.strip()
        normalized_name = normalize_event_name(raw_name)
        if normalized_name in seen:
            raise PerfCollectionError(
                f"duplicate requested perf event: {normalized_name}"
            )
        seen.add(normalized_name)
        raw_names.append(raw_name)
        normalized.append(normalized_name)
    if not normalized:
        raise PerfCollectionError("at least one perf event is required")
    return tuple(raw_names), tuple(normalized)


def _requested_events(values: Iterable[str] | None) -> tuple[str, ...]:
    """Normalize selected event names and reject duplicate selections."""

    _, normalized = _event_arguments(values)
    return normalized


def _missing_requested_events(
    selected: Sequence[str], parsed: Iterable[str]
) -> None:
    selected_names = set(selected)
    parsed_names = set(parsed)
    missing = sorted(selected_names - parsed_names)
    if missing:
        joined = ", ".join(missing)
        raise PerfParseError(f"perf output is missing requested event(s): {joined}")
    unexpected = sorted(parsed_names - selected_names)
    if unexpected:
        joined = ", ".join(unexpected)
        raise PerfParseError(f"perf output contains unrequested event(s): {joined}")


def build_perf_command(
    perf: str | os.PathLike[str],
    raw_output: str | os.PathLike[str],
    repeat: int | str,
    events: Iterable[str],
    command: Sequence[str],
) -> list[str]:
    """Construct the exact perf invocation used by the collector."""

    repeat_value = _validated_repeat(repeat)
    if not isinstance(perf, (str, os.PathLike)) or not str(perf).strip():
        raise PerfCollectionError("perf executable must not be empty")
    workload = list(command)
    if not workload or not workload[0] or not str(workload[0]).strip():
        raise PerfCollectionError("a non-empty workload command is required after --")
    raw_events, _ = _event_arguments(events)
    return [
        str(perf),
        "stat",
        "--json-output",
        "--no-big-num",
        "--output",
        str(raw_output),
        "--repeat",
        str(repeat_value),
        "--event",
        ",".join(raw_events),
        "--",
        *workload,
    ]


def _read_raw_output(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PerfCollectionError(
            f"unable to read perf raw output {path}: {exc}"
        ) from exc
    except UnicodeError as exc:
        raise PerfCollectionError(
            f"perf raw output {path} is not valid UTF-8: {exc}"
        ) from exc


def _run_perf(argv: Sequence[str]) -> None:
    """Run perf while forwarding workload stdout/stderr to this process stderr."""

    environment = os.environ.copy()
    environment["LC_ALL"] = "C"
    try:
        completed = subprocess.run(
            list(argv),
            check=False,
            env=environment,
            stdout=sys.stderr,
            stderr=sys.stderr,
            text=True,
        )
    except FileNotFoundError as exc:
        raise PerfCollectionError(
            f"perf executable not found: {argv[0]}"
        ) from exc
    except PermissionError as exc:
        raise PerfCollectionError(
            f"permission denied while executing perf: {argv[0]}"
        ) from exc
    except OSError as exc:
        raise PerfCollectionError(f"unable to execute perf: {exc}") from exc

    if completed.returncode != 0:
        raise PerfCollectionError(
            f"perf stat failed with exit status {completed.returncode}; "
            "check permissions, event names, and workload command"
        )


def _perf_version(perf: str | os.PathLike[str]) -> str:
    """Collect the first nonempty perf version line before the workload run."""

    environment = os.environ.copy()
    environment["LC_ALL"] = "C"
    try:
        completed = subprocess.run(
            [str(perf), "--version"],
            check=False,
            env=environment,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise PerfCollectionError(
            f"perf executable not found: {perf}"
        ) from exc
    except PermissionError as exc:
        raise PerfCollectionError(
            f"permission denied while executing perf: {perf}"
        ) from exc
    except OSError as exc:
        raise PerfCollectionError(f"unable to execute perf --version: {exc}") from exc

    if completed.returncode != 0:
        raise PerfCollectionError(
            f"perf --version failed with exit status {completed.returncode}"
        )
    for output in (completed.stdout, completed.stderr):
        if not isinstance(output, str):
            continue
        for line in output.splitlines():
            if line.strip():
                return line.strip()
    raise PerfCollectionError("perf --version produced no version output")


def collect_perf_stat(args: argparse.Namespace) -> dict[str, object]:
    """Run perf, parse its complete output, and return a result bundle."""

    if platform.system() != "Linux":
        raise PerfCollectionError("perf stat collection is supported only on Linux")

    repeat = _validated_repeat(args.repeat)
    raw_events, selected = _event_arguments(getattr(args, "event", None))
    command = list(getattr(args, "command", ()))
    if command and command[0] == "--":
        command = command[1:]
    if not command or not command[0] or not str(command[0]).strip():
        raise PerfCollectionError("a non-empty workload command is required after --")

    raw_output = Path(args.raw_output)
    result_output = getattr(args, "output", None)
    try:
        validate_artifact_output_paths(
            raw_output,
            result_output,
            tool="perf",
        )
    except ValueError as exc:
        raise PerfCollectionError(str(exc)) from exc
    try:
        raw_output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PerfCollectionError(
            f"unable to create perf raw output directory {raw_output.parent}: {exc}"
        ) from exc
    try:
        raw_output.unlink(missing_ok=True)
    except OSError as exc:
        raise PerfCollectionError(
            f"unable to prepare perf raw output {raw_output}: {exc}"
        ) from exc
    invocation = build_perf_command(
        args.perf,
        raw_output,
        repeat,
        raw_events,
        command,
    )
    tool_version = _perf_version(args.perf)
    _run_perf(invocation)

    # Do not remove or rewrite this file: it is the reproducibility artifact,
    # including when a later parser or bundle step reports an error.
    raw_text = _read_raw_output(raw_output)
    raw_metrics, raw_event_names = _parse_perf_records(raw_text, repeat)
    _missing_requested_events(selected, raw_event_names)
    metrics = _add_ipc(raw_metrics, repeat)

    observation: dict[str, object] = {
        "kind": "counters",
        "metrics": metrics,
        "tool": str(args.perf),
    }
    parameters: dict[str, object] = {
        "perf_stat": {
            "scope": args.scope,
            "events": list(raw_events),
            "repeat": repeat,
            "command": command,
            "tool_version": tool_version,
            "invocation": invocation,
        }
    }
    artifacts = [
        artifact_record(
            raw_output,
            "perf-stat-json",
            "Exact raw perf stat JSON output",
        ),
    ]
    document = build_result_bundle(
        args,
        [observation],
        parameters=parameters,
        artifacts=artifacts,
    )
    write_result(document, args)
    return document


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect Linux perf stat counters into a result bundle"
    )
    add_result_arguments(parser)
    parser.add_argument("--perf", default="perf", help="perf executable")
    parser.add_argument(
        "--event",
        action="append",
        default=None,
        help="perf event (repeatable; defaults to cycles,instructions,branches,branch-misses)",
    )
    parser.add_argument(
        "--repeat",
        type=_positive_repeat,
        default=5,
        help="number of perf repetitions (positive integer; default: 5)",
    )
    parser.add_argument(
        "--scope",
        required=True,
        choices=("process-aggregate", "dedicated-workload"),
        help="scope represented by the workload command",
    )
    parser.add_argument(
        "--raw-output",
        required=True,
        type=Path,
        help="path where perf JSON output is written",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="workload command, introduced by --",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        collect_perf_stat(args)
    except (PerfCollectionError, OSError, ValueError) as exc:
        print(f"collect_perf_stat: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
