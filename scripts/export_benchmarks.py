#!/usr/bin/env python3
"""Export ``simd-lab-benchmark-v2`` documents as CSV or Markdown tables."""

from __future__ import annotations

import argparse
import csv
import html
import io
import json
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


class ExportError(ValueError):
    """Raised when a benchmark document or normalized row is not exportable."""


CSV_COLUMNS = (
    "source",
    "timestamp_unix",
    "target_tier",
    "host_system",
    "host_release",
    "host_machine",
    "host_processor",
    "language",
    "benchmark",
    "implementation",
    "n",
    "working_set_bytes",
    "effective_bytes_per_iteration",
    "iterations_per_sample",
    "sample_count",
    "min_ns_per_element",
    "median_ns_per_element",
    "p95_ns_per_element",
    "mad_ns_per_element",
    "median_gib_per_second",
    "run_metadata",
    "target_configuration",
    "toolchains",
)

_STATISTIC_FIELDS = (
    "min_ns_per_element",
    "median_ns_per_element",
    "p95_ns_per_element",
    "mad_ns_per_element",
    "median_gib_per_second",
)
_REQUIRED_DOCUMENT_FIELDS = (
    "schema",
    "timestamp_unix",
    "protocol",
    "host",
    "target_configuration",
    "toolchains",
    "runs",
)
_REQUIRED_RESULT_FIELDS = (
    "name",
    "n",
    "working_set_bytes",
    "effective_bytes_per_iteration",
    "iterations_per_sample",
    "sample_count",
    "samples",
    "statistics",
)
_JSON_COLUMNS = {"run_metadata", "target_configuration", "toolchains"}


def _source_label(source: Path | str) -> str:
    return str(source)


def _error(source: str, detail: str) -> ExportError:
    return ExportError(f"{source}: {detail}")


def _mapping(value: Any, source: str, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(source, f"{label} must be an object")
    return value


def _required(mapping: Mapping[str, Any], key: str, source: str, label: str) -> Any:
    if key not in mapping:
        raise _error(source, f"{label} is missing required field {key!r}")
    return mapping[key]


def _nonempty_string(value: Any, source: str, label: str) -> str:
    if not isinstance(value, str):
        raise _error(source, f"{label} must be a string")
    if not value.strip():
        raise _error(source, f"{label} must not be empty")
    return value


def _integer(value: Any, source: str, label: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _error(source, f"{label} must be an integer")
    if value < minimum:
        raise _error(source, f"{label} must be at least {minimum}")
    return value


def _number(value: Any, source: str, label: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error(source, f"{label} must be a number")
    if isinstance(value, float) and not math.isfinite(value):
        raise _error(source, f"{label} must be finite")
    if isinstance(value, int):
        # Python integers are finite, including integers too large for a float.
        finite = True
    else:
        finite = math.isfinite(value)
    if not finite:
        raise _error(source, f"{label} must be finite")
    if value < 0:
        raise _error(source, f"{label} must be nonnegative")
    return value


def _validate_close(
    actual: int | float,
    expected: int | float,
    source: str,
    label: str,
    *,
    rel_tol: float,
    abs_tol: float,
) -> None:
    if math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol):
        return
    raise _error(
        source,
        f"{label} value {actual!r} is not close to recomputed value {expected!r} "
        f"(rel_tol={rel_tol}, abs_tol={abs_tol})",
    )


_STATISTIC_REL_TOL = 1e-8
_STATISTIC_ABS_TOL = 2e-9
_GIB_REL_TOL = 1e-6
_GIB_ABS_TOL = 1e-6




def _compact_json(value: Any, source: str, label: str) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise _error(source, f"{label} is not valid JSON: {exc}") from exc


def _parse_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _host_value(host: Mapping[str, Any], key: str, source: str) -> str:
    value = host.get(key)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise _error(source, f"host field {key!r} must be a string")
    return value


def _host_label(host: Mapping[str, Any], source: str) -> str:
    values = [
        _host_value(host, "system", source),
        _host_value(host, "release", source),
        _host_value(host, "machine", source),
        _host_value(host, "processor", source),
    ]
    label = " ".join(value for value in values if value)
    return label or "unknown"


def _document_parts(document: Mapping[str, Any], source: str) -> tuple[
    int,
    str,
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    list[Any],
]:
    if "schema" not in document:
        raise _error(source, "document is missing required field 'schema'")
    if document["schema"] != "simd-lab-benchmark-v2":
        raise _error(source, f"unsupported schema {document.get('schema')!r}")
    for key in _REQUIRED_DOCUMENT_FIELDS[1:]:
        if key not in document:
            raise _error(source, f"document is missing required field {key!r}")

    timestamp = _integer(document["timestamp_unix"], source, "timestamp_unix", minimum=0)
    protocol = _mapping(document["protocol"], source, "protocol")
    target_tier = _nonempty_string(
        _required(protocol, "target_tier", source, "protocol"),
        source,
        "protocol.target_tier",
    )
    for field, expected in (
        ("aggregation", "median"),
        ("spread", "median_absolute_deviation"),
    ):
        value = _required(protocol, field, source, "protocol")
        if value != expected:
            raise _error(source, f"protocol.{field} must equal {expected!r}")

    tail_percentile = _required(protocol, "tail_percentile", source, "protocol")
    if (
        isinstance(tail_percentile, bool)
        or not isinstance(tail_percentile, int)
        or tail_percentile != 95
    ):
        raise _error(source, "protocol.tail_percentile must equal integer 95")

    raw_samples_preserved = _required(
        protocol,
        "raw_samples_preserved",
        source,
        "protocol",
    )
    if raw_samples_preserved is not True:
        raise _error(source, "protocol.raw_samples_preserved must equal True")

    host = _mapping(document["host"], source, "host")
    target_configuration = _mapping(
        document["target_configuration"], source, "target_configuration"
    )
    toolchains = _mapping(document["toolchains"], source, "toolchains")
    runs = document["runs"]
    if not isinstance(runs, list):
        raise _error(source, "runs must be an array")
    if not runs:
        raise _error(source, "runs must not be empty")

    # Validate the values which are emitted as compact JSON while the source
    # context is still available for a useful error message.
    _compact_json(target_configuration, source, "target_configuration")
    _compact_json(toolchains, source, "toolchains")
    return timestamp, target_tier, host, target_configuration, toolchains, runs


def _validate_result(
    result: Mapping[str, Any],
    source: str,
    run_index: int,
    result_index: int,
    language: str,
) -> tuple[str, str, int, int, int, int, int, dict[str, Any]]:
    run_label = f"runs[{run_index}] (language {language!r})"
    result_name = result.get("name", "<missing>")
    result_label = f"{run_label}.results[{result_index}] ({result_name!r})"
    for key in _REQUIRED_RESULT_FIELDS:
        if key not in result:
            raise _error(source, f"{result_label} is missing required field {key!r}")

    full_name = _nonempty_string(result["name"], source, f"{result_label} field 'name'")
    if full_name.count("/") != 1:
        raise _error(
            source,
            f"{result_label} field 'name' must contain exactly one '/'",
        )
    benchmark, implementation = full_name.split("/", 1)
    if not benchmark or not implementation:
        raise _error(
            source,
            f"{result_label} field 'name' must contain nonempty benchmark/implementation parts",
        )

    n = _integer(result["n"], source, f"{result_label} field 'n'", minimum=0)
    working_set_bytes = _integer(
        result["working_set_bytes"],
        source,
        f"{result_label} field 'working_set_bytes'",
        minimum=0,
    )
    effective_bytes = _integer(
        result["effective_bytes_per_iteration"],
        source,
        f"{result_label} field 'effective_bytes_per_iteration'",
        minimum=0,
    )
    iterations = _integer(
        result["iterations_per_sample"],
        source,
        f"{result_label} field 'iterations_per_sample'",
        minimum=1,
    )
    sample_count = _integer(
        result["sample_count"],
        source,
        f"{result_label} field 'sample_count'",
        minimum=1,
    )

    samples = _mapping(result["samples"], source, f"{result_label} field 'samples'")
    if samples.get("unit") != "ns/element":
        raise _error(
            source,
            f"{result_label} field 'samples.unit' must equal 'ns/element'",
        )
    sample_values = samples.get("values")
    if not isinstance(sample_values, list):
        raise _error(source, f"{result_label} field 'samples.values' must be an array")
    if len(sample_values) != sample_count:
        raise _error(
            source,
            f"{result_label} field 'samples.values' has {len(sample_values)} values; "
            f"sample_count is {sample_count}",
        )
    normalized_samples: list[int | float] = []
    for sample_index, value in enumerate(sample_values):
        normalized_samples.append(
            _number(
                value,
                source,
                f"{result_label} field 'samples.values[{sample_index}]'",
            )
        )

    statistics = _mapping(result["statistics"], source, f"{result_label} field 'statistics'")
    normalized_statistics: dict[str, int | float] = {}
    for field in _STATISTIC_FIELDS:
        value = _required(statistics, field, source, f"{result_label} field 'statistics'")
        normalized_statistics[field] = _number(
            value,
            source,
            f"{result_label} field 'statistics.{field}'",
        )
    sorted_samples = sorted(normalized_samples)
    median_ns = sorted_samples[len(sorted_samples) // 2]
    p95_index = min(
        len(sorted_samples) - 1,
        math.ceil(95 * len(sorted_samples) / 100) - 1,
    )
    deviations = sorted(abs(value - median_ns) for value in normalized_samples)
    expected_statistics = {
        "min_ns_per_element": sorted_samples[0],
        "median_ns_per_element": median_ns,
        "p95_ns_per_element": sorted_samples[p95_index],
        "mad_ns_per_element": deviations[len(deviations) // 2],
    }
    for field, expected in expected_statistics.items():
        _validate_close(
            normalized_statistics[field],
            expected,
            source,
            f"{result_label} field 'statistics.{field}'",
            rel_tol=_STATISTIC_REL_TOL,
            abs_tol=_STATISTIC_ABS_TOL,
        )

    if n > 0:
        if median_ns == 0:
            raise _error(
                source,
                f"{result_label} field 'statistics.median_gib_per_second' "
                "cannot be validated because median_ns_per_element is zero",
            )
        expected_gib_per_second = (
            (effective_bytes / n)
            / (median_ns * 1e-9)
            / (1024**3)
        )
        _validate_close(
            normalized_statistics["median_gib_per_second"],
            expected_gib_per_second,
            source,
            f"{result_label} field 'statistics.median_gib_per_second'",
            rel_tol=_GIB_REL_TOL,
            abs_tol=_GIB_ABS_TOL,
        )


    return (
        full_name,
        benchmark,
        n,
        working_set_bytes,
        effective_bytes,
        iterations,
        sample_count,
        {
            "implementation": implementation,
            "samples": normalized_samples,
            "statistics": normalized_statistics,
        },
    )


def _document_identity(document: Mapping[str, Any], source: str) -> str:
    # The source path is human-facing; the private suffix keeps two documents
    # with the same path but different environments from being merged in a
    # Markdown export.
    try:
        fingerprint = json.dumps(
            document,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError):
        fingerprint = source
    return f"{source}\x00{fingerprint}"


def normalize_benchmark_rows(document: Mapping[str, Any], source: Path) -> list[dict[str, Any]]:
    """Validate and flatten one benchmark-v2 document into export rows."""

    source_text = _source_label(source)
    document_mapping = _mapping(document, source_text, "document")
    (
        timestamp,
        target_tier,
        host,
        target_configuration,
        toolchains,
        runs,
    ) = _document_parts(document_mapping, source_text)

    host_system = _host_value(host, "system", source_text)
    host_release = _host_value(host, "release", source_text)
    host_machine = _host_value(host, "machine", source_text)
    host_processor = _host_value(host, "processor", source_text)
    host_label = _host_label(host, source_text)
    target_configuration_data = dict(target_configuration)
    toolchains_data = dict(toolchains)
    _compact_json(target_configuration_data, source_text, "target_configuration")
    _compact_json(toolchains_data, source_text, "toolchains")
    document_id = _document_identity(document_mapping, source_text)

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for run_index, run_value in enumerate(runs):
        run_label = f"runs[{run_index}]"
        run = _mapping(run_value, source_text, run_label)
        language = _nonempty_string(
            _required(run, "language", source_text, run_label),
            source_text,
            f"{run_label} field 'language'",
        )
        metadata_value = run.get("metadata", {})
        if metadata_value is None:
            metadata_value = {}
        metadata = dict(
            _mapping(metadata_value, source_text, f"{run_label} field 'metadata'")
        )
        results_value = _required(run, "results", source_text, run_label)
        if not isinstance(results_value, list):
            raise _error(source_text, f"{run_label}.results must be an array")
        if not results_value:
            raise _error(source_text, f"{run_label}.results must not be empty")
        for result_index, result_value in enumerate(results_value):
            result = _mapping(
                result_value,
                source_text,
                f"{run_label}.results[{result_index}]",
            )
            (
                full_name,
                benchmark,
                n,
                working_set_bytes,
                effective_bytes,
                iterations,
                sample_count,
                normalized,
            ) = _validate_result(result, source_text, run_index, result_index, language)
            duplicate_key = (language, full_name, n)
            if duplicate_key in seen:
                raise _error(
                    source_text,
                    f"{run_label}.results[{result_index}] duplicates "
                    f"(language, name, n)=({language!r}, {full_name!r}, {n})",
                )
            seen.add(duplicate_key)

            row: dict[str, Any] = {
                "source": source_text,
                "timestamp_unix": timestamp,
                "target_tier": target_tier,
                "host_system": host_system,
                "host_release": host_release,
                "host_machine": host_machine,
                "host_processor": host_processor,
                "language": language,
                "benchmark": benchmark,
                "implementation": normalized["implementation"],
                "n": n,
                "working_set_bytes": working_set_bytes,
                "effective_bytes_per_iteration": effective_bytes,
                "iterations_per_sample": iterations,
                "sample_count": sample_count,
                **normalized["statistics"],
                "run_metadata": metadata,
                "target_configuration": target_configuration_data,
                "toolchains": toolchains_data,
                "_document_id": document_id,
                "_document_source": source_text,
                "_document_target_tier": target_tier,
                "_document_host": host_label,
            }
            rows.append(row)

    rows.sort(key=_row_sort_key)
    return rows


def load_benchmark_document(path: Path) -> dict[str, Any]:
    """Load one UTF-8 benchmark-v2 JSON document, failing with ExportError."""

    path = Path(path)
    source = _source_label(path)
    try:
        text = path.read_text(encoding="utf-8")
        document = json.loads(text, parse_constant=_parse_constant)
    except (OSError, UnicodeError) as exc:
        raise _error(source, f"could not read input: {exc}") from exc
    except (json.JSONDecodeError, ValueError) as exc:
        raise _error(source, f"invalid JSON: {exc}") from exc

    if not isinstance(document, Mapping):
        raise _error(source, "top-level JSON value must be an object")
    # Validate the envelope here so this public loader never returns an
    # unsupported document. Row-level validation is repeated by normalization,
    # which also supplies the flattened representation.
    _document_parts(document, source)
    return dict(document)


def _row_sort_key(row: Mapping[str, Any]) -> tuple[str, str, int, str, str]:
    values: list[str] = []
    for field in ("source", "benchmark", "n", "language", "implementation"):
        if field not in row:
            raise ExportError(f"row is missing required field {field!r}")
        values.append(str(row[field]))
    return (values[0], values[1], int(values[2]), values[3], values[4])


def load_benchmark_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Load and deterministically sort flattened rows from all input paths."""

    rows: list[dict[str, Any]] = []
    for document_index, path_value in enumerate(paths):
        path = Path(path_value)
        document = load_benchmark_document(path)
        document_rows = normalize_benchmark_rows(document, path)
        # A repeated path is still two input documents. Keep that distinction
        # private so Markdown cannot accidentally merge their sections.
        for row in document_rows:
            row["_document_id"] = f"{row['_document_id']}\x00{document_index}"
        rows.extend(document_rows)
    rows.sort(key=_row_sort_key)
    return rows


def _rows_for_render(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    materialized = list(rows)
    for index, row in enumerate(materialized):
        if not isinstance(row, Mapping):
            raise ExportError(f"row {index} must be an object")
        for field in CSV_COLUMNS:
            if field not in row:
                raise ExportError(f"row {index} is missing required field {field!r}")
    materialized.sort(key=_row_sort_key)
    return materialized


def _json_cell(row: Mapping[str, Any], field: str, row_index: int) -> str:
    value = row[field]
    if isinstance(value, str):
        return value
    return _compact_json(value, f"row {row_index}", field)


def render_csv(rows: Iterable[Mapping[str, Any]]) -> str:
    """Render normalized rows with the contracted stable CSV columns."""

    materialized = _rows_for_render(rows)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=list(CSV_COLUMNS),
        lineterminator="\n",
        extrasaction="ignore",
    )
    writer.writeheader()
    for index, row in enumerate(materialized):
        output_row = dict(row)
        for field in _JSON_COLUMNS:
            output_row[field] = _json_cell(row, field, index)
        writer.writerow({field: output_row.get(field, "") for field in CSV_COLUMNS})
    rendered = stream.getvalue()
    return rendered.rstrip("\n") + "\n"


def _markdown_escape(value: Any) -> str:
    text = str(value)
    text = text.replace("\\", "\\\\")
    text = text.replace("|", "\\|")
    text = html.escape(text, quote=False)
    return text.replace("\r\n", "<br>").replace("\r", "<br>").replace("\n", "<br>")


def _markdown_metric(row: Mapping[str, Any], field: str, row_index: int) -> str:
    value = row[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExportError(f"row {row_index} field {field!r} must be a number")
    if isinstance(value, float) and not math.isfinite(value):
        raise ExportError(f"row {row_index} field {field!r} must be finite")
    if value < 0:
        raise ExportError(f"row {row_index} field {field!r} must be nonnegative")
    return format(value, ".9g")


def _fallback_document_id(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("source", "")),
        str(row.get("target_tier", "")),
        " ".join(
            str(row.get(field, ""))
            for field in ("host_system", "host_release", "host_machine", "host_processor")
            if row.get(field, "") not in (None, "")
        )
        or "unknown",
    )


def render_markdown(rows: Iterable[Mapping[str, Any]]) -> str:
    """Render rows as escaped, grouped Markdown comparison tables."""

    materialized = _rows_for_render(rows)
    if not materialized:
        return "\n"

    documents: dict[Any, list[Mapping[str, Any]]] = {}
    document_order: list[Any] = []
    for row in materialized:
        key = row.get("_document_id")
        if key is None:
            key = _fallback_document_id(row)
        if key not in documents:
            documents[key] = []
            document_order.append(key)
        documents[key].append(row)

    sections: list[str] = []
    for key in document_order:
        document_rows = documents[key]
        first = document_rows[0]
        source = first.get("_document_source", first.get("source", ""))
        target_tier = first.get("_document_target_tier", first.get("target_tier", ""))
        host = first.get("_document_host")
        if host is None:
            host = _fallback_document_id(first)[2]

        lines = [
            f"## {_markdown_escape(source)}",
            f"Target tier: {_markdown_escape(target_tier)}; Host: {_markdown_escape(host)}",
        ]
        groups: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
        for row in document_rows:
            benchmark = str(row["benchmark"])
            n_value = row["n"]
            if isinstance(n_value, bool) or not isinstance(n_value, int):
                raise ExportError(f"Markdown row field 'n' for {benchmark!r} must be an integer")
            groups.setdefault((benchmark, n_value), []).append(row)

        for group_key, group_rows in sorted(groups.items()):
            benchmark, n = group_key
            lines.append("")
            lines.append(f"### {_markdown_escape(benchmark)} (n={n})")
            lines.append(
                "| Language | Implementation | Median ns/element | p95 ns/element | "
                "MAD ns/element | Median GiB/s |"
            )
            lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
            group_rows = sorted(
                group_rows,
                key=lambda row: (str(row["language"]), str(row["implementation"])),
            )
            for row_index, row in enumerate(group_rows):
                lines.append(
                    "| "
                    + " | ".join(
                        (
                            _markdown_escape(row["language"]),
                            _markdown_escape(row["implementation"]),
                            _markdown_metric(row, "median_ns_per_element", row_index),
                            _markdown_metric(row, "p95_ns_per_element", row_index),
                            _markdown_metric(row, "mad_ns_per_element", row_index),
                            _markdown_metric(row, "median_gib_per_second", row_index),
                        )
                    )
                    + " |"
                )
        sections.append("\n".join(lines))

    return "\n\n".join(sections).rstrip("\n") + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line exporter and return a process status."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=("csv", "markdown"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("input", nargs="+", type=Path)
    args = parser.parse_args(argv)

    try:
        rows = load_benchmark_rows(args.input)
        if args.format == "csv":
            rendered = render_csv(rows)
        else:
            rendered = render_markdown(rows)
        if args.output is None:
            sys.stdout.write(rendered)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
    except (ExportError, OSError) as exc:
        print(f"benchmark export failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
