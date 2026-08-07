#!/usr/bin/env python3
"""Run llvm-mca on a deliberately bounded assembly region.

The adapter is intentionally small and conservative.  It does not attempt to
turn LLVM's model-specific JSON into portable counters: the complete JSON
report is kept as an artifact and the result bundle contains only a textual
analysis observation pointing at that artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from result_bundle import (
    add_result_arguments,
    artifact_record,
    build_result_bundle,
    paths_alias,
    validate_artifact_output_paths,
    write_result,
)


class McaError(RuntimeError):
    """An input, llvm-mca, or llvm-mca-report error."""


class AssemblyBoundsError(McaError):
    """The assembly did not contain an unambiguous analysis boundary."""


@dataclass(frozen=True)
class MarkerRegion:
    """A matched LLVM-MCA marker pair, using one-based source lines."""

    name: str | None
    start_line: int
    end_line: int


@dataclass(frozen=True)
class BoundedAssembly:
    """Assembly text and provenance for the boundary used by the run."""

    text: str
    mode: str
    markers: tuple[MarkerRegion, ...] = ()
    start_label: str | None = None
    end_label: str | None = None
    start_line: int | None = None
    end_line: int | None = None

    @property
    def selector(self) -> dict[str, object]:
        """Return a JSON-serializable description of the source boundary."""

        if self.mode == "labels":
            return {
                "mode": self.mode,
                "start_label": self.start_label,
                "end_label": self.end_label,
                "start_line": self.start_line,
                "end_line": self.end_line,
            }
        return {
            "mode": self.mode,
            "regions": [
                {
                    "name": marker.name,
                    "start_line": marker.start_line,
                    "end_line": marker.end_line,
                }
                for marker in self.markers
            ],
        }


@dataclass(frozen=True)
class McaReport:
    """The selected region and the two views required by this adapter."""

    document: dict[str, Any]
    region: dict[str, Any]
    summary_view: dict[str, Any]
    resource_pressure_view: dict[str, Any]

    @property
    def name(self) -> str | None:
        value = self.region.get("Name")
        return value if isinstance(value, str) else None


@dataclass(frozen=True)
class McaRun:
    """Raw subprocess results needed for provenance and parsing."""

    report_text: str
    report_bytes: bytes
    tool_version: str
    command: tuple[str, ...]


# Assembly labels can be local (``.Ltmp``), private (``Lfoo$bar``), or ordinary
# symbols.  Matching the beginning of a line also handles ``label: instruction``
# without mistaking a colon in an instruction comment for a label.
_LABEL_PREFIX_RE = re.compile(r"^\s*([.$A-Za-z_][\w.$@]*)\s*:(?:\s|$)")
_SYMBOL_ASSIGN_RE = re.compile(r"^\s*[.$A-Za-z_][\w.$@]*\s*=\s*")
_COMMENT_ONLY_RE = re.compile(r"^\s*(?:#|//|;|@)")
_DIRECTIVE_RE = re.compile(r"^\s*\.")

# Dialect directives change instruction parsing and must precede the selected
# body. Metadata-only directives are safe to omit; every other directive and
# symbol assignment is retained so extraction cannot change emitted instructions.
_DIALECT_DIRECTIVE_RE = re.compile(
    r"^\s*\.(?:"
    r"intel_syntax|att_syntax|syntax|arch(?:_extension)?|cpu|fpu|"
    r"thumb|arm|force_thumb|code16(?:gcc)?|code32|code64|"
    r"option|machine|abi|bundle_align_mode|bundle_lock|bundle_unlock|"
    r"variant_pcs"
    r")\b",
    re.IGNORECASE,
)
_DIALECT_SET_RE = re.compile(
    r"^\s*\.set\s+(?:"
    r"push|pop|at|noat|macro|nomacro|reorder|noreorder|"
    r"mips\d+|mips(?:16|32|64)|micromips|nomicromips"
    r")\b",
    re.IGNORECASE,
)
_METADATA_DIRECTIVE_RE = re.compile(
    r"^\s*\.(?:"
    r"section|pushsection|popsection|previous|subsection|text|data|bss|"
    r"globl|global|weak|weak_definition|private_extern|hidden|protected|"
    r"internal|local|extern|type|size|file|loc|cfi(?:_\w+)?|ident|"
    r"subsections_via_symbols|addrsig(?:_sym)?|note|build_version|"
    r"macosx_version_min"
    r")\b",
    re.IGNORECASE,
)
_MARKER_RE = re.compile(
    r"^\s*(?:#|//|;|@)\s*LLVM-MCA-(BEGIN|END)(?:\s+(.+?))?\s*$",
    re.IGNORECASE,
)


def _normalized_label(label: str) -> str:
    value = label.strip()
    if value.endswith(":"):
        value = value[:-1].rstrip()
    if not value:
        raise AssemblyBoundsError("assembly labels must not be empty")
    return value


def _label_positions(source: str, label: str) -> list[int]:
    wanted = _normalized_label(label)
    positions: list[int] = []
    for index, line in enumerate(source.splitlines(keepends=True)):
        match = _LABEL_PREFIX_RE.match(line)
        if match and match.group(1) == wanted:
            positions.append(index)
    return positions


def find_label_bounds(source: str, start_label: str, end_label: str) -> tuple[int, int]:
    """Find exactly one start and end label and return zero-based line bounds.

    The returned interval is inclusive of the start-label line and exclusive of
    the end-label line.  Missing, duplicated, and reversed labels are all
    errors rather than guesses.
    """

    if not start_label or not end_label:
        raise AssemblyBoundsError("--start-label and --end-label must be supplied together")

    start_positions = _label_positions(source, start_label)
    end_positions = _label_positions(source, end_label)
    if not start_positions:
        raise AssemblyBoundsError(f"start label {start_label!r} was not found")
    if len(start_positions) != 1:
        raise AssemblyBoundsError(f"start label {start_label!r} is ambiguous")
    if not end_positions:
        raise AssemblyBoundsError(f"end label {end_label!r} was not found")
    if len(end_positions) != 1:
        raise AssemblyBoundsError(f"end label {end_label!r} is ambiguous")

    start, end = start_positions[0], end_positions[0]
    if start >= end:
        raise AssemblyBoundsError(
            f"assembly labels are reversed: {start_label!r} must precede {end_label!r}"
        )
    return start, end


def _is_dialect_directive(line: str) -> bool:
    return bool(_DIALECT_DIRECTIVE_RE.match(line) or _DIALECT_SET_RE.match(line))


def _retain_region_line(line: str) -> bool:
    """Whether a line is safe/useful in an llvm-mca instruction stream."""

    if not line.strip():
        return True
    if _is_dialect_directive(line):
        return True
    if _COMMENT_ONLY_RE.match(line):
        return False
    if _SYMBOL_ASSIGN_RE.match(line):
        return True
    label = _LABEL_PREFIX_RE.match(line)
    if label is not None:
        return True
    if _DIRECTIVE_RE.match(line):
        return not bool(_METADATA_DIRECTIVE_RE.match(line))
    return True

def extract_label_region(source: str, start_label: str, end_label: str) -> str:
    """Extract a label-bounded stream with dialect setup and a faithful body.

    Dialect directives before the start label are retained because they affect
    parsing of the selected body. The selected interval is inclusive of the
    start label and exclusive of the end label. Metadata-only directives and
    comments are removed; semantic directives and symbol aliases are retained.
    """

    start, end = find_label_bounds(source, start_label, end_label)
    lines = source.splitlines(keepends=True)

    prefix = [line for line in lines[:start] if _is_dialect_directive(line)]
    body = [line for line in lines[start:end] if _retain_region_line(line)]
    return "".join(prefix + body)


# A spelling alias is useful to callers that use the adjective form.
extract_labeled_region = extract_label_region


def find_marker_regions(source: str) -> tuple[MarkerRegion, ...]:
    """Validate and return all non-nested LLVM-MCA marker pairs."""

    regions: list[MarkerRegion] = []
    open_marker: tuple[str | None, int] | None = None
    for line_number, line in enumerate(source.splitlines(keepends=True), start=1):
        matches = list(_MARKER_RE.finditer(line))
        if not matches:
            continue
        if len(matches) != 1:
            raise AssemblyBoundsError(f"multiple LLVM-MCA markers on line {line_number}")
        marker = matches[0]
        kind = marker.group(1).upper()
        name = marker.group(2)
        if kind == "BEGIN":
            if open_marker is not None:
                raise AssemblyBoundsError(
                    f"nested LLVM-MCA-BEGIN marker on line {line_number}"
                )
            open_marker = (name, line_number)
            continue

        if open_marker is None:
            raise AssemblyBoundsError(f"LLVM-MCA-END without BEGIN on line {line_number}")
        begin_name, begin_line = open_marker
        if name is not None and begin_name is not None and name != begin_name:
            raise AssemblyBoundsError(
                f"LLVM-MCA marker names do not match on lines {begin_line} and {line_number}"
            )
        if name is not None and begin_name is None:
            raise AssemblyBoundsError(
                f"named LLVM-MCA-END has no matching named BEGIN on line {line_number}"
            )
        regions.append(MarkerRegion(begin_name or name, begin_line, line_number))
        open_marker = None

    if open_marker is not None:
        _, begin_line = open_marker
        raise AssemblyBoundsError(f"LLVM-MCA-BEGIN on line {begin_line} has no END")
    return tuple(regions)


def bound_assembly(
    source: str,
    *,
    start_label: str | None = None,
    end_label: str | None = None,
) -> BoundedAssembly:
    """Apply explicit label extraction or require matched MCA markers."""

    if (start_label is None) != (end_label is None):
        raise AssemblyBoundsError("--start-label and --end-label must be supplied together")
    if start_label is not None and end_label is not None:
        start, end = find_label_bounds(source, start_label, end_label)
        lines = source.splitlines(keepends=True)
        prefix = [line for line in lines[:start] if _is_dialect_directive(line)]
        body = [line for line in lines[start:end] if _retain_region_line(line)]
        return BoundedAssembly(
            text="".join(prefix + body),
            mode="labels",
            start_label=_normalized_label(start_label),
            end_label=_normalized_label(end_label),
            start_line=start + 1,
            end_line=end + 1,
        )

    markers = find_marker_regions(source)
    if not markers:
        raise AssemblyBoundsError(
            "assembly must contain matched LLVM-MCA-BEGIN/END markers "
            "or use --start-label/--end-label"
        )
    return BoundedAssembly(text=source, mode="markers", markers=markers)


# Explicit name for callers that want to emphasize validation.
prepare_assembly = bound_assembly


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _as_bytes(value: object) -> bytes:
    if isinstance(value, bytes):
        return value
    if value is None:
        return b""
    if isinstance(value, str):
        return value.encode("utf-8")
    return str(value).encode("utf-8")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant {value!r}")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _is_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return not isinstance(value, float) or math.isfinite(value)


def _validate_view(region_name: str, view_name: str, value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        raise McaError(f"CodeRegion {region_name!r} has invalid or missing {view_name}")
    return value


def select_code_region(
    document: Mapping[str, Any], requested_region: str | None = None
) -> dict[str, Any]:
    """Select one CodeRegion, rejecting absent/ambiguous selections."""

    code_regions = document.get("CodeRegions")
    if not isinstance(code_regions, list) or not code_regions:
        raise McaError("llvm-mca JSON is missing a non-empty CodeRegions array")
    if any(not isinstance(region, dict) for region in code_regions):
        raise McaError("llvm-mca JSON CodeRegions entries must be objects")

    if requested_region is not None:
        matches = [
            region
            for region in code_regions
            if isinstance(region.get("Name"), str)
            and region.get("Name") == requested_region
        ]
        if not matches:
            raise McaError(f"llvm-mca region {requested_region!r} was not found")
        if len(matches) != 1:
            raise McaError(f"llvm-mca region {requested_region!r} is ambiguous")
        return matches[0]

    if len(code_regions) != 1:
        raise McaError(
            f"llvm-mca produced {len(code_regions)} CodeRegions; specify --region"
        )
    return code_regions[0]


def parse_mca_json(
    raw_report: str | bytes | Mapping[str, Any], requested_region: str | None = None
) -> McaReport:
    """Parse JSON and require the selected region's model views."""

    if isinstance(raw_report, Mapping):
        document = dict(raw_report)
    else:
        try:
            loaded = json.loads(
                _as_text(raw_report),
                parse_constant=_reject_json_constant,
                object_pairs_hook=_unique_json_object,
            )
        except (TypeError, json.JSONDecodeError, ValueError) as exc:
            raise McaError(f"llvm-mca produced invalid JSON: {exc}") from exc
        if not isinstance(loaded, dict):
            raise McaError("llvm-mca JSON root must be an object")
        document = loaded

    region = select_code_region(document, requested_region)
    region_name = region.get("Name") if isinstance(region.get("Name"), str) else "<unnamed>"
    summary_view = _validate_view(region_name, "SummaryView", region.get("SummaryView"))
    resource_view = _validate_view(
        region_name, "ResourcePressureView", region.get("ResourcePressureView")
    )
    if not any(
        key in summary_view
        for key in (
            "BlockRThroughput",
            "TotalCycles",
            "TotalInstructions",
            "TotalNumIterations",
            "Iterations",
            "Instructions",
            "TotaluOps",
            "uOpsPerCycle",
            "IPC",
            "DispatchWidth",
        )
    ):
        raise McaError(f"CodeRegion {region_name!r} has no recognized SummaryView fields")
    if not any(
        key in resource_view
        for key in ("ResourcePressure", "ResourcePressureInfo", "Resources")
    ):
        raise McaError(
            f"CodeRegion {region_name!r} has no recognized ResourcePressureView fields"
        )

    # Reject malformed top-level view members while staying tolerant of LLVM
    # version-specific field names.  The raw JSON remains authoritative.
    for key in (
        "TotalCycles",
        "TotalInstructions",
        "TotalNumIterations",
        "Iterations",
        "Instructions",
        "TotaluOps",
        "uOpsPerCycle",
        "IPC",
        "DispatchWidth",
        "BlockRThroughput",
    ):
        if key in summary_view and not _is_number(summary_view[key]):
            raise McaError(f"CodeRegion {region_name!r} has non-numeric SummaryView.{key}")
    for key in ("ResourcePressure", "ResourcePressureInfo", "Resources"):
        if key in resource_view and not isinstance(resource_view[key], (list, dict)):
            raise McaError(
                f"CodeRegion {region_name!r} has invalid ResourcePressureView.{key}"
            )

    return McaReport(document, region, summary_view, resource_view)


# Short aliases make the pure parser convenient in fixture tests and callers.
parse_report = parse_mca_json


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _pressure_entry_count(resource_view: Mapping[str, Any]) -> int | None:
    for key in ("ResourcePressure", "ResourcePressureInfo", "Resources"):
        value = resource_view.get(key)
        if isinstance(value, (list, dict)):
            return len(value)
    return None


def format_analysis_summary(report: McaReport) -> str:
    """Render a stable human-readable summary without normalizing model data."""

    name = report.name or "<unnamed>"
    summary = report.summary_view
    fields: list[str] = []
    for key, label in (
        ("BlockRThroughput", "block throughput"),
        ("TotalCycles", "total cycles"),
        ("TotalInstructions", "total instructions"),
        ("Instructions", "instructions"),
        ("TotalNumIterations", "iterations"),
        ("Iterations", "iterations"),
        ("TotaluOps", "total uops"),
        ("uOpsPerCycle", "uops per cycle"),
        ("IPC", "ipc"),
        ("DispatchWidth", "dispatch width"),
    ):
        if key in summary:
            fields.append(f"{label}={_format_value(summary[key])}")
    pressure_count = _pressure_entry_count(report.resource_pressure_view)
    if pressure_count is not None:
        fields.append(f"resource pressure entries={pressure_count}")
    details = "; ".join(fields) if fields else "model views present"
    return f"llvm-mca region {name!r}: {details}"


# Keep a concise alias for code that calls the operation a formatter.
format_summary = format_analysis_summary


def _run_subprocess(
    command: Sequence[str], *, input_text: str | None = None
) -> subprocess.CompletedProcess[bytes]:
    try:
        completed = subprocess.run(
            list(command),
            input=None if input_text is None else input_text.encode("utf-8"),
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise McaError(f"could not execute {command[0]!r}: {exc}") from exc
    return completed


def _raise_for_process_failure(
    completed: subprocess.CompletedProcess[bytes], command: Sequence[str]
) -> None:
    stderr = _as_text(getattr(completed, "stderr", None)).strip()
    if getattr(completed, "returncode", 1) != 0:
        detail = stderr or _as_text(getattr(completed, "stdout", None)).strip()
        rendered = " ".join(command)
        suffix = f": {detail}" if detail else ""
        raise McaError(f"llvm-mca command failed ({rendered}){suffix}")
    if re.search(
        r"\b(?:error|unsupported|unknown|invalid|failed|cannot)\b",
        stderr,
        re.IGNORECASE,
    ):
        raise McaError(f"llvm-mca reported an error: {stderr}")


def _tool_version(executable: str) -> str:
    completed = _run_subprocess([executable, "--version"])
    _raise_for_process_failure(completed, [executable, "--version"])
    output = _as_text(getattr(completed, "stdout", None)) or _as_text(
        getattr(completed, "stderr", None)
    )
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        raise McaError("llvm-mca --version returned no version")
    return lines[0]


def build_mca_command(
    executable: str,
    *,
    iterations: int,
    target: str,
    cpu: str,
    mattrs: Sequence[str] = (),
) -> list[str]:
    """Build the fixed, explicit llvm-mca invocation used by the adapter."""

    if iterations <= 0:
        raise McaError("llvm-mca iterations must be positive")
    command = [
        executable,
        "-json",
        "-iterations",
        str(iterations),
        "-mtriple",
        target,
        "-mcpu",
        cpu,
    ]
    if mattrs:
        command.extend(("-mattr", ",".join(mattrs)))
    command.append("-")
    return command


def run_llvm_mca(
    bounded_text: str,
    *,
    executable: str = "llvm-mca",
    iterations: int = 100,
    target: str,
    cpu: str,
    mattrs: Sequence[str] = (),
) -> McaRun:
    """Run llvm-mca on stdin and return exact report text plus provenance."""

    command = build_mca_command(
        executable,
        iterations=iterations,
        target=target,
        cpu=cpu,
        mattrs=mattrs,
    )
    version = _tool_version(executable)
    completed = _run_subprocess(command, input_text=bounded_text)
    _raise_for_process_failure(completed, command)
    report_bytes = _as_bytes(getattr(completed, "stdout", None))
    if not report_bytes.strip():
        raise McaError("llvm-mca produced an empty JSON report")
    try:
        report_text = report_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise McaError("llvm-mca JSON report is not valid UTF-8") from exc
    return McaRun(
        report_text=report_text,
        report_bytes=report_bytes,
        tool_version=version,
        command=tuple(command),
    )


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_result_arguments(parser)
    parser.add_argument("assembly", type=Path, help="assembly source to analyze")
    parser.add_argument("--llvm-mca", default="llvm-mca", help="llvm-mca executable")
    parser.add_argument(
        "--iterations", type=_positive_int, default=100, help="fixed model iterations"
    )
    parser.add_argument(
        "--mattr",
        action="append",
        default=[],
        metavar="FEATURES",
        help="target feature set (repeatable; values are joined with commas)",
    )
    bounds = parser.add_argument_group("assembly bounds")
    bounds.add_argument("--start-label", help="inclusive assembly start label")
    bounds.add_argument("--end-label", help="exclusive assembly end label")
    parser.add_argument("--region", help="named llvm-mca CodeRegion to select")
    parser.add_argument(
        "--raw-output",
        required=True,
        type=Path,
        help="path for the exact raw llvm-mca JSON report",
    )
    return parser


def _parameters(
    args: argparse.Namespace,
    *,
    source_path: Path,
    source_hash: str,
    run: McaRun,
    bounded: BoundedAssembly,
) -> dict[str, object]:
    return {
        "llvm_mca": {
            "tool_version": run.tool_version,
            "command": list(run.command),
            "input_path": str(source_path),
            "input_sha256": source_hash,
            "region": args.region,
            "start_label": args.start_label,
            "end_label": args.end_label,
            "iterations": args.iterations,
            "target": args.target,
            "cpu": args.cpu,
            "mattrs": list(args.mattr),
            "bounds": bounded.selector,
        }
    }


def run_analysis(args: argparse.Namespace) -> dict[str, object]:
    """Perform one analysis and return the result-bundle document."""

    source_path = Path(args.assembly)
    try:
        source_bytes = source_path.read_bytes()
        source_text = source_bytes.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise McaError(f"could not read assembly {source_path}: {exc}") from exc

    raw_path = Path(args.raw_output)
    result_output = getattr(args, "output", None)
    try:
        validate_artifact_output_paths(
            raw_path,
            result_output,
            tool="llvm-mca",
        )
    except ValueError as exc:
        raise McaError(str(exc)) from exc
    if paths_alias(raw_path, source_path):
        raise McaError("llvm-mca raw output must not overwrite the input assembly")
    if result_output is not None and paths_alias(result_output, source_path):
        raise McaError("llvm-mca result output must not overwrite the input assembly")

    bounded = bound_assembly(
        source_text,
        start_label=args.start_label,
        end_label=args.end_label,
    )
    try:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.unlink(missing_ok=True)
    except OSError as exc:
        raise McaError(
            f"could not prepare raw llvm-mca output {raw_path}: {exc}"
        ) from exc
    run = run_llvm_mca(
        bounded.text,
        executable=args.llvm_mca,
        iterations=args.iterations,
        target=args.target,
        cpu=args.cpu,
        mattrs=args.mattr,
    )
    try:
        raw_path.write_bytes(run.report_bytes)
    except OSError as exc:
        raise McaError(f"could not write raw llvm-mca report {raw_path}: {exc}") from exc
    parsed = parse_mca_json(run.report_text, args.region)

    artifact = artifact_record(
        raw_path,
        "llvm-mca-json",
        "Exact raw llvm-mca JSON report",
    )
    observation = {
        "kind": "analysis",
        "summary": format_analysis_summary(parsed),
        "severity": "info",
        "evidence": [artifact["path"]],
    }
    return build_result_bundle(
        args,
        [observation],
        parameters=_parameters(
            args,
            source_path=source_path,
            source_hash=hashlib.sha256(source_bytes).hexdigest(),
            bounded=bounded,
            run=run,
        ),
        artifacts=[artifact],
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        document = run_analysis(args)
        write_result(document, args)
    except (McaError, OSError, ValueError) as exc:
        print(f"analyze_mca: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
