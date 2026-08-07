from __future__ import annotations

import copy
import csv
import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

from scripts.export_benchmarks import (
    ExportError,
    load_benchmark_document,
    load_benchmark_rows,
    main,
    normalize_benchmark_rows,
    render_csv,
    render_markdown,
)


CSV_COLUMNS = [
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
]


INTEGER_RESULT_FIELDS = (
    "n",
    "working_set_bytes",
    "effective_bytes_per_iteration",
    "iterations_per_sample",
    "sample_count",
)

STATISTIC_FIELDS = (
    "min_ns_per_element",
    "median_ns_per_element",
    "p95_ns_per_element",
    "mad_ns_per_element",
    "median_gib_per_second",
)


def producer_statistics(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    median = ordered[len(ordered) // 2]
    deviations = sorted(abs(value - median) for value in ordered)
    p95_index = min(len(ordered) - 1, math.ceil(len(ordered) * 95 / 100) - 1)
    return {
        "min_ns_per_element": ordered[0],
        "median_ns_per_element": median,
        "p95_ns_per_element": ordered[p95_index],
        "mad_ns_per_element": deviations[len(deviations) // 2],
    }


def producer_gib_per_second(
    effective_bytes_per_iteration: int,
    n: int,
    median_ns_per_element: float,
) -> float:
    bytes_per_element = effective_bytes_per_iteration / n
    return bytes_per_element / (median_ns_per_element * 1e-9) / (2**30)


def benchmark_result(
    name: str = "axpy/scalar",
    *,
    n: int = 1024,
    sample_values: list[float] | None = None,
    statistics: dict[str, float] | None = None,
) -> dict[str, Any]:
    values = (
        [1.0, 1.125, 1.125, 1.25, 1.25, 1.75]
        if sample_values is None
        else sample_values
    )
    stats = producer_statistics(values)
    stats["median_gib_per_second"] = producer_gib_per_second(
        4096,
        n,
        stats["median_ns_per_element"],
    )
    if statistics is not None:
        stats.update(statistics)
    return {
        "name": name,
        "n": n,
        "working_set_bytes": 8192,
        "effective_bytes_per_iteration": 4096,
        "iterations_per_sample": 8,
        "sample_count": len(values),
        "statistics": stats,
        "samples": {"unit": "ns/element", "values": values},
    }



def benchmark_run(
    language: str = "rust",
    *,
    results: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "language": language,
        "metadata":
            {"zeta": "last", "alpha": 1}
            if metadata is None
            else metadata,
        "results":
            [benchmark_result()]
            if results is None
            else results,
        "skipped": [],
    }



def benchmark_document(
    *,
    timestamp_unix: int = 1_700_000_000,
    target_tier: str = "baseline",
    host: dict[str, Any] | None = None,
    target_configuration: dict[str, Any] | None = None,
    toolchains: dict[str, Any] | None = None,
    runs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "schema": "simd-lab-benchmark-v2",
        "timestamp_unix": timestamp_unix,
        "protocol": {
            "target_tier": target_tier,
            "aggregation": "median",
            "spread": "median_absolute_deviation",
            "tail_percentile": 95,
            "raw_samples_preserved": True,
        },
        "target_configuration": {
            "rustflags": None,
            "cmake_simd_lab_cpu": target_tier,
            "zig_args": [],
        }
        if target_configuration is None
        else target_configuration,
        "host": {
            "system": "Linux",
            "release": "6.8.0",
            "machine": "x86_64",
            "processor": "Test CPU",
            "python": "3.12.0",
        }
        if host is None
        else host,
        "toolchains": {
            "rustc": "rustc 1.80.0",
            "cargo": "cargo 1.80.0",
            "zig": "0.13.0",
            "clang++": "clang version 18",
            "g++": "g++ 14",
            "cmake": "cmake 3.30",
        }
        if toolchains is None
        else toolchains,
        "runs":
            [benchmark_run()]
            if runs is None
            else runs,
    }



def markdown_escape(value: object) -> str:
    """The Markdown table escaping contract used by the exporter."""
    text = str(value)
    return (
        text.replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\r\n", "<br>")
        .replace("\r", "<br>")
        .replace("\n", "<br>")
    )


class ExportBenchmarkTests(unittest.TestCase):
    def write_document(
        self,
        directory: str,
        name: str,
        document: dict[str, Any],
    ) -> Path:
        path = Path(directory) / name
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def assert_normalization_error(
        self,
        document: dict[str, Any],
        *context: str,
        source: Path = Path("invalid.json"),
    ) -> None:
        with self.assertRaises(ExportError) as raised:
            normalize_benchmark_rows(document, source)
        message = str(raised.exception)
        for part in context:
            self.assertIn(part, message)

    def test_valid_document_factory_loads_and_normalizes_name_and_metadata(self) -> None:
        document = benchmark_document(
            runs=[
                benchmark_run(
                    "rust",
                    results=[benchmark_result("workload/implementation")],
                )
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_document(directory, "benchmark.json", document)
            self.assertEqual(load_benchmark_document(path), document)
            rows = load_benchmark_rows([path])

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["source"], str(path))
        self.assertEqual(row["language"], "rust")
        self.assertEqual(row["benchmark"], "workload")
        self.assertEqual(row["implementation"], "implementation")
        self.assertEqual(row["timestamp_unix"], 1_700_000_000)
        self.assertEqual(row["target_tier"], "baseline")
        self.assertEqual(row["n"], 1024)
        self.assertEqual(row["run_metadata"], {"zeta": "last", "alpha": 1})

    def test_csv_has_stable_header_order_and_sorts_rows(self) -> None:
        document = benchmark_document(
            runs=[
                benchmark_run(
                    "zig",
                    results=[
                        benchmark_result("zeta/vector"),
                        benchmark_result("alpha/scalar"),
                    ],
                ),
                benchmark_run(
                    "rust",
                    results=[benchmark_result("alpha/autovec")],
                ),
                benchmark_run(
                    "cpp23",
                    results=[benchmark_result("alpha/dispatch")],
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_document(directory, "sorted.json", document)
            rows = load_benchmark_rows([path])

        rendered = render_csv(rows)
        records = list(csv.reader(io.StringIO(rendered)))
        self.assertEqual(records[0], CSV_COLUMNS)
        self.assertEqual(
            [(record[7], record[8], record[9]) for record in records[1:]],
            [
                ("cpp23", "alpha", "dispatch"),
                ("rust", "alpha", "autovec"),
                ("zig", "alpha", "scalar"),
                ("zig", "zeta", "vector"),
            ],
        )
        self.assertEqual(len(records), 5)
        self.assertTrue(rendered.endswith("\n"))

    def test_csv_uses_standard_escaping_and_compact_deterministic_json(self) -> None:
        document = benchmark_document(
            host={
                "system": "Linux",
                "release": 'release, "quoted"',
                "machine": "x86_64",
                "processor": 'CPU, "fast" \\path',
                "python": "3.12",
            },
            runs=[
                benchmark_run(
                    metadata={"zeta": "last", "alpha": "first"},
                    results=[benchmark_result("bench/impl")],
                )
            ],
            target_configuration={
                "zig_args": ["-Doptimize=ReleaseFast", "-Dcpu=native"],
                "cmake_simd_lab_cpu": "baseline",
                "rustflags": None,
            },
            toolchains={
                "g++": "g++ 14",
                "rustc": "rustc 1.80",
                "cargo": "cargo 1.80",
                "zig": "0.13",
                "clang++": "clang 18",
                "cmake": "cmake 3.30",
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_document(directory, 'input, "quoted".json', document)
            rows = load_benchmark_rows([path])

        rendered = render_csv(rows)
        record = list(csv.reader(io.StringIO(rendered)))[1]
        self.assertEqual(record[0], str(path))
        self.assertEqual(record[3], "Linux")
        self.assertEqual(record[4], 'release, "quoted"')
        self.assertEqual(record[5], "x86_64")
        self.assertEqual(record[6], 'CPU, "fast" \\path')
        self.assertIn('""quoted""', rendered)

        expected_json = {
            "run_metadata": json.dumps(
                document["runs"][0]["metadata"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "target_configuration": json.dumps(
                document["target_configuration"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "toolchains": json.dumps(
                document["toolchains"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
        self.assertEqual(record[20], expected_json["run_metadata"])
        self.assertEqual(record[21], expected_json["target_configuration"])
        self.assertEqual(record[22], expected_json["toolchains"])
        self.assertEqual(render_csv(rows), render_csv(copy.deepcopy(rows)))

    def test_markdown_separates_inputs_and_groups_sorted_benchmark_sizes(self) -> None:
        first = benchmark_document(
            target_tier="baseline",
            host={
                "system": "Linux",
                "release": "first-release",
                "machine": "first-machine",
                "processor": "first-cpu",
                "python": "3.12",
            },
            runs=[
                benchmark_run(
                    "rust",
                    results=[
                        benchmark_result("zeta/slow", n=2048),
                        benchmark_result("alpha/auto", n=1024),
                    ],
                ),
                benchmark_run(
                    "cpp23",
                    results=[
                        benchmark_result("zeta/fast", n=2048),
                        benchmark_result("alpha/dispatch", n=1024),
                    ],
                ),
            ],
        )
        second = benchmark_document(
            timestamp_unix=1_700_000_001,
            target_tier="native",
            host={
                "system": "Darwin",
                "release": "second-release",
                "machine": "second-machine",
                "processor": "second-cpu",
                "python": "3.12",
            },
            runs=[benchmark_run("zig", results=[benchmark_result("alpha/native")])],
        )
        with tempfile.TemporaryDirectory() as directory:
            first_path = self.write_document(directory, "a.json", first)
            second_path = self.write_document(directory, "b.json", second)
            rows = load_benchmark_rows([second_path, first_path])

        rendered = render_markdown(rows)
        first_heading = f"## {first_path}"
        second_heading = f"## {second_path}"
        self.assertIn(first_heading, rendered)
        self.assertIn(second_heading, rendered)
        self.assertLess(rendered.index(first_heading), rendered.index(second_heading))
        first_section = rendered[rendered.index(first_heading) : rendered.index(second_heading)]
        self.assertIn("Target tier: baseline; Host:", first_section)
        self.assertNotIn("Target tier: native; Host:", first_section)
        self.assertIn("Target tier: native; Host:", rendered[rendered.index(second_heading) :])

        self.assertEqual(
            rendered.count("| Language | Implementation | Median ns/element | p95 ns/element | MAD ns/element | Median GiB/s |"),
            3,
        )
        self.assertLess(
            first_section.index("### alpha (n=1024)"),
            first_section.index("### zeta (n=2048)"),
        )
        self.assertLess(
            first_section.index("| cpp23 | dispatch |"),
            first_section.index("| rust | auto |"),
        )
        self.assertIn("### alpha (n=1024)", rendered[rendered.index(second_heading) :])
        self.assertTrue(rendered.endswith("\n"))
        self.assertFalse(rendered.endswith("\n\n"))

    def test_markdown_formats_metrics_and_escapes_backslash_pipe_and_newline(self) -> None:
        benchmark = "bench|mark\\part\nline & < >"
        implementation = "impl|one\\two\nthree & < >"
        language = "rust & < >"
        source = Path("source|file\\name\npart & < >.json")
        host_system = "Linux|host\\x\nnext & < >"
        median_ns_per_element = 1.23456789123
        p95_ns_per_element = 9.876543211
        sample_values = [
            1.0,
            median_ns_per_element - 0.125,
            median_ns_per_element,
            median_ns_per_element + 0.125,
            p95_ns_per_element,
        ]
        median_gib_per_second = producer_gib_per_second(
            4096,
            1024,
            median_ns_per_element,
        )
        document = benchmark_document(
            host={
                "system": host_system,
                "release": "release",
                "machine": "machine",
                "processor": "processor",
                "python": "3.12",
            },
            runs=[
                benchmark_run(
                    language,
                    results=[
                        benchmark_result(
                            f"{benchmark}/{implementation}",
                            sample_values=sample_values,
                            statistics={
                                "median_ns_per_element": median_ns_per_element,
                                "p95_ns_per_element": p95_ns_per_element,
                                "median_gib_per_second": median_gib_per_second,
                            },
                        )
                    ],
                )
            ],
        )
        rows = normalize_benchmark_rows(document, source)
        rendered = render_markdown(rows)

        escaped_source = markdown_escape(source)
        escaped_benchmark = markdown_escape(benchmark)
        escaped_implementation = markdown_escape(implementation)
        escaped_language = markdown_escape(language)
        escaped_host = markdown_escape(host_system)
        self.assertIn(f"## {escaped_source}", rendered)
        self.assertIn(f"Target tier: baseline; Host: {escaped_host}", rendered)
        self.assertIn(f"### {escaped_benchmark} (n=1024)", rendered)
        expected_row = (
            f"| {escaped_language} | {escaped_implementation} | "
            f"{format(median_ns_per_element, '.9g')} | "
            f"{format(p95_ns_per_element, '.9g')} | "
            f"{format(0.125, '.9g')} | "
            f"{format(median_gib_per_second, '.9g')} |"
        )
        self.assertIn(expected_row, rendered)
        self.assertIn("&amp;", rendered)
        self.assertIn("&lt;", rendered)
        self.assertIn("&gt;", rendered)
        self.assertNotIn("bench|mark", rendered)
        self.assertNotIn("\nline", rendered)
        self.assertNotIn("\nnext", rendered)
        self.assertIn("\\\\", rendered)
        self.assertIn("\\|", rendered)
        self.assertIn("<br>", rendered)

    def test_missing_optional_host_strings_become_empty_cells_without_zero_metrics(self) -> None:
        document = benchmark_document(host={})
        rows = normalize_benchmark_rows(document, Path("missing-host.json"))
        row = rows[0]
        for field in ("host_system", "host_release", "host_machine", "host_processor"):
            self.assertEqual(row[field], "")
        self.assertEqual(row["n"], 1024)
        self.assertEqual(row["working_set_bytes"], 8192)
        self.assertEqual(row["median_ns_per_element"], 1.25)
        self.assertNotEqual(row["n"], 0)
        self.assertNotEqual(row["median_ns_per_element"], 0)

    def test_load_rejects_malformed_json_and_wrong_schema_with_path_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            malformed = Path(directory) / "malformed.json"
            malformed.write_text('{ "schema": ', encoding="utf-8")
            with self.assertRaises(ExportError) as malformed_error:
                load_benchmark_document(malformed)
            self.assertIn(str(malformed), str(malformed_error.exception))
            self.assertRegex(str(malformed_error.exception).lower(), "json|parse|decode")

            wrong = self.write_document(
                directory,
                "wrong-schema.json",
                {"schema": "simd-lab-result-v1"},
            )
            with self.assertRaises(ExportError) as schema_error:
                load_benchmark_rows([wrong])
            self.assertIn(str(wrong), str(schema_error.exception))
            self.assertIn("schema", str(schema_error.exception).lower())

    def test_cli_writes_csv_to_stdout_and_markdown_to_output(self) -> None:
        document = benchmark_document()
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_document(directory, "input.json", document)
            expected_csv = render_csv(load_benchmark_rows([path]))
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                status = main(["--format", "csv", str(path)])
            self.assertEqual(status, 0)
            self.assertEqual(stdout.getvalue(), expected_csv)
            self.assertEqual(stderr.getvalue(), "")
            self.assertTrue(stdout.getvalue().endswith("\n"))
            self.assertFalse(stdout.getvalue().endswith("\n\n"))

            output = Path(directory) / "summary.md"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                status = main(
                    ["--format", "markdown", "--output", str(output), str(path)]
                )
            self.assertEqual(status, 0)
            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(stderr.getvalue(), "")
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                render_markdown(load_benchmark_rows([path])),
            )
            self.assertTrue(output.read_text(encoding="utf-8").endswith("\n"))
            self.assertFalse(output.read_text(encoding="utf-8").endswith("\n\n"))

    def test_cli_reports_bad_input_with_nonzero_status_path_and_reason(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            malformed = Path(directory) / "bad.json"
            malformed.write_text("not json", encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                status = main(["--format", "csv", str(malformed)])
            self.assertNotEqual(status, 0)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn(str(malformed), stderr.getvalue())
            self.assertRegex(stderr.getvalue().lower(), "json|parse|decode|error")

            wrong = self.write_document(
                directory,
                "wrong.json",
                {"schema": "not-the-benchmark-schema"},
            )
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                status = main(["--format", "markdown", str(wrong)])
            self.assertNotEqual(status, 0)
            self.assertIn(str(wrong), stderr.getvalue())
            self.assertIn("schema", stderr.getvalue().lower())

    def test_missing_or_empty_runs_and_results_are_rejected(self) -> None:
        missing_runs = benchmark_document()
        del missing_runs["runs"]
        self.assert_normalization_error(missing_runs, "runs")

        empty_runs = benchmark_document(runs=[])
        self.assert_normalization_error(empty_runs, "runs")

        missing_results = benchmark_document(runs=[benchmark_run()])
        del missing_results["runs"][0]["results"]
        self.assert_normalization_error(missing_results, "results")

        empty_results = benchmark_document(runs=[benchmark_run(results=[])])
        self.assert_normalization_error(empty_results, "results")

    def test_required_top_level_objects_and_arrays_have_shape_validation(self) -> None:
        for field in ("protocol", "target_configuration", "host", "toolchains"):
            with self.subTest(field=field):
                candidate = benchmark_document()
                candidate[field] = []
                self.assert_normalization_error(candidate, field)

        candidate = benchmark_document()
        candidate["runs"] = {}
        self.assert_normalization_error(candidate, "runs")

        candidate = benchmark_document()
        candidate["protocol"] = None
        self.assert_normalization_error(candidate, "protocol")

    def test_invalid_run_and_result_row_shapes_include_field_context(self) -> None:
        cases: list[tuple[str, dict[str, Any], tuple[str, ...]]] = []

        candidate = benchmark_document(runs=[None])  # type: ignore[list-item]
        cases.append(("run object", candidate, ("runs",)))

        candidate = benchmark_document(runs=["not an object"])  # type: ignore[list-item]
        cases.append(("run type", candidate, ("runs",)))

        candidate = benchmark_document(runs=[benchmark_run()])
        del candidate["runs"][0]["language"]
        cases.append(("missing language", candidate, ("language",)))

        candidate = benchmark_document(runs=[benchmark_run("")])
        cases.append(("empty language", candidate, ("language",)))

        candidate = benchmark_document(runs=[benchmark_run()])
        candidate["runs"][0]["results"] = {}
        cases.append(("results type", candidate, ("results",)))

        candidate = benchmark_document(runs=[benchmark_run(results=[None])])  # type: ignore[list-item]
        cases.append(("result object", candidate, ("results",)))

        for label, invalid, context in cases:
            with self.subTest(case=label):
                self.assert_normalization_error(invalid, *context)

    def test_protocol_fields_are_required_and_exact(self) -> None:
        altered_values: dict[str, tuple[object, ...]] = {
            "target_tier": ("",),
            "aggregation": ("mean",),
            "spread": ("standard_deviation",),
            "tail_percentile": (90, 95.0, True),
            "raw_samples_preserved": (False,),
        }
        for field, values in altered_values.items():
            missing = benchmark_document()
            del missing["protocol"][field]
            with self.subTest(field=field, case="missing"):
                self.assert_normalization_error(missing, "protocol", field)
            for altered in values:
                candidate = benchmark_document()
                candidate["protocol"][field] = altered
                with self.subTest(field=field, case=("altered", repr(altered))):
                    self.assert_normalization_error(candidate, "protocol", field)


    def test_missing_required_metrics_are_errors_not_fabricated_zeroes(self) -> None:
        for field in INTEGER_RESULT_FIELDS:
            with self.subTest(field=field):
                candidate = benchmark_document()
                del candidate["runs"][0]["results"][0][field]
                self.assert_normalization_error(candidate, field)

        for field in STATISTIC_FIELDS:
            with self.subTest(field=field):
                candidate = benchmark_document()
                del candidate["runs"][0]["results"][0]["statistics"][field]
                self.assert_normalization_error(candidate, field)

    def test_statistics_must_match_raw_samples(self) -> None:
        expected_fields = (
            "min_ns_per_element",
            "median_ns_per_element",
            "p95_ns_per_element",
            "mad_ns_per_element",
        )
        document = benchmark_document()
        result = document["runs"][0]["results"][0]
        expected = producer_statistics(result["samples"]["values"])
        for field in expected_fields:
            self.assertEqual(result["statistics"][field], expected[field])

            candidate = copy.deepcopy(document)
            candidate["runs"][0]["results"][0]["statistics"][field] = (
                expected[field] + 1.0
            )
            with self.subTest(field=field):
                self.assert_normalization_error(candidate, field)

    def test_median_gib_per_second_must_match_raw_sample_median(self) -> None:
        document = benchmark_document()
        result = document["runs"][0]["results"][0]
        expected = producer_gib_per_second(
            result["effective_bytes_per_iteration"],
            result["n"],
            result["statistics"]["median_ns_per_element"],
        )
        self.assertEqual(result["statistics"]["median_gib_per_second"], expected)

        candidate = copy.deepcopy(document)
        candidate["runs"][0]["results"][0]["statistics"]["median_gib_per_second"] = (
            expected + 1.0
        )
        self.assert_normalization_error(candidate, "median_gib_per_second")

    def test_bool_integer_fields_are_rejected(self) -> None:
        for field in INTEGER_RESULT_FIELDS:
            with self.subTest(field=field):
                candidate = benchmark_document()
                candidate["runs"][0]["results"][0][field] = True
                self.assert_normalization_error(candidate, field)

    def test_negative_and_zero_integer_ranges_are_rejected(self) -> None:
        for field in ("n", "working_set_bytes", "effective_bytes_per_iteration"):
            with self.subTest(field=field):
                candidate = benchmark_document()
                candidate["runs"][0]["results"][0][field] = -1
                self.assert_normalization_error(candidate, field)

        for field in ("iterations_per_sample", "sample_count"):
            with self.subTest(field=field):
                candidate = benchmark_document()
                candidate["runs"][0]["results"][0][field] = 0
                self.assert_normalization_error(candidate, field)

    def test_statistics_must_be_finite_nonnegative_nonbool_numbers(self) -> None:
        invalid_values: tuple[object, ...] = (True, -1.0, float("nan"), float("inf"))
        for field in STATISTIC_FIELDS:
            for value in invalid_values:
                with self.subTest(field=field, value=repr(value)):
                    candidate = benchmark_document()
                    candidate["runs"][0]["results"][0]["statistics"][field] = value
                    self.assert_normalization_error(candidate, field)

    def test_samples_must_be_finite_nonnegative_and_have_declared_unit(self) -> None:
        for value in (True, -1.0, float("nan"), float("inf")):
            with self.subTest(value=repr(value)):
                candidate = benchmark_document()
                candidate["runs"][0]["results"][0]["samples"]["values"] = [
                    value,
                    1.0,
                ]
                self.assert_normalization_error(candidate, "samples")

        mismatch = benchmark_document()
        mismatch["runs"][0]["results"][0]["sample_count"] = 3
        self.assert_normalization_error(mismatch, "sample_count", "samples")

        wrong_unit = benchmark_document()
        wrong_unit["runs"][0]["results"][0]["samples"]["unit"] = "seconds"
        self.assert_normalization_error(wrong_unit, "unit")

        missing_values = benchmark_document()
        del missing_values["runs"][0]["results"][0]["samples"]["values"]
        self.assert_normalization_error(missing_values, "values")

    def test_malformed_names_are_rejected_with_name_context(self) -> None:
        for name in ("", "noslash", "/implementation", "benchmark/", "a/b/c"):
            with self.subTest(name=repr(name)):
                candidate = benchmark_document()
                candidate["runs"][0]["results"][0]["name"] = name
                self.assert_normalization_error(candidate, "name")

    def test_duplicate_language_and_name_identity_is_rejected_per_document(self) -> None:
        duplicate = benchmark_document(
            runs=[
                benchmark_run(
                    "rust",
                    results=[benchmark_result("same/implementation")],
                ),
                benchmark_run(
                    "rust",
                    results=[benchmark_result("same/implementation")],
                ),
            ]
        )
        self.assert_normalization_error(duplicate, "rust", "same/implementation")

        same_name_different_languages = benchmark_document(
            runs=[
                benchmark_run("cpp23", results=[benchmark_result("same/implementation")]),
                benchmark_run("rust", results=[benchmark_result("same/implementation")]),
            ]
        )
        rows = normalize_benchmark_rows(
            same_name_different_languages,
            Path("different-identities.json"),
        )
        self.assertEqual(len(rows), 2)

    def test_multiple_inputs_remain_distinct_even_when_identity_matches(self) -> None:
        first = benchmark_document(
            target_tier="baseline",
            runs=[benchmark_run("rust", results=[benchmark_result("same/impl")])],
        )
        second = benchmark_document(
            target_tier="native",
            timestamp_unix=1_700_000_123,
            runs=[benchmark_run("rust", results=[benchmark_result("same/impl")])],
        )
        with tempfile.TemporaryDirectory() as directory:
            first_path = self.write_document(directory, "first.json", first)
            second_path = self.write_document(directory, "second.json", second)
            rows = load_benchmark_rows([second_path, first_path])

        self.assertEqual(len(rows), 2)
        self.assertEqual({row["source"] for row in rows}, {str(first_path), str(second_path)})
        self.assertEqual({row["target_tier"] for row in rows}, {"baseline", "native"})
        self.assertEqual({row["timestamp_unix"] for row in rows}, {1_700_000_000, 1_700_000_123})


if __name__ == "__main__":
    unittest.main()
