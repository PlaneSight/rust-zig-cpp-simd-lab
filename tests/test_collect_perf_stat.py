from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import collect_perf_stat as collector


def perf_lines(*records: dict[str, object]) -> str:
    return "\n".join(json.dumps(record) for record in records) + "\n"


VALID_OUTPUT = perf_lines(
    {
        "counter-value": "1000",
        "unit": "cycles",
        "event": "cycles",
        "runtime": "1000000",
        "pcnt-running": "100.00",
        "variance": "0.01",
    },
    {
        "counter-value": "2500.5",
        "unit": "instructions",
        "event": "instructions",
        "runtime": "1000000",
        "running": "100.00",
        "variance": "0.02",
    },
    {"counter-value": "10", "event": "branches"},
    {"counter-value": "2", "unit": "count", "event": "branch-misses"},
)


class PerfParserTests(unittest.TestCase):
    def test_valid_ndjson_numeric_strings_and_optional_fields(self) -> None:
        metrics = collector.parse_perf_stat(VALID_OUTPUT, 1)

        self.assertEqual(
            [metric["name"] for metric in metrics],
            ["cycles", "instructions", "branches", "branch_misses", "ipc"],
        )
        self.assertEqual(metrics[0]["value"], 1000.0)
        self.assertEqual(metrics[0]["unit"], "cycles")
        self.assertEqual(metrics[2]["unit"], "count")
        self.assertEqual(metrics[3]["aggregation"], "single")
        self.assertEqual(metrics[3]["samples"], 1)
        self.assertNotIn("variance", metrics[0])
        self.assertNotIn("runtime", metrics[0])
        self.assertNotIn("running", metrics[0])

    def test_ipc_is_ratio_of_means(self) -> None:
        metrics = collector.parse_perf_stat(VALID_OUTPUT, 3)
        ipc = metrics[-1]

        self.assertEqual(ipc["name"], "ipc")
        self.assertAlmostEqual(ipc["value"], 2.5005)
        self.assertEqual(ipc["unit"], "instructions/cycle")
        self.assertEqual(ipc["aggregation"], "ratio-of-means")
        self.assertEqual(ipc["samples"], 3)
        self.assertEqual(metrics[0]["aggregation"], "mean")
        self.assertEqual(metrics[0]["samples"], 3)


    def test_large_integral_counter_preserves_precision(self) -> None:
        expected = (1 << 53) + 1
        metrics = collector.parse_perf_stat(
            perf_lines({"counter-value": str(expected), "event": "instructions"}),
            1,
        )

        self.assertEqual(metrics[0]["value"], expected)
        self.assertIsInstance(metrics[0]["value"], int)

    def test_event_names_are_normalized_to_snake_case(self) -> None:
        metrics = collector.parse_perf_stat(
            perf_lines({"counter-value": "4", "event": "L1-dcache-load-misses"}),
            1,
        )
        self.assertEqual(metrics[0]["name"], "l1_dcache_load_misses")

    def test_unsupported_and_not_counted_are_rejected(self) -> None:
        for value in ("<not supported>", "<not counted>"):
            with self.subTest(value=value):
                with self.assertRaises(collector.PerfParseError):
                    collector.parse_perf_stat(
                        perf_lines({"counter-value": value, "event": "cycles"}),
                        1,
                    )

    def test_malformed_and_nonnumeric_records_are_rejected(self) -> None:
        with self.assertRaises(collector.PerfParseError):
            collector.parse_perf_stat('{"event":"cycles"}\n', 1)
        with self.assertRaises(collector.PerfParseError):
            collector.parse_perf_stat("not-json\n", 1)
        with self.assertRaises(collector.PerfParseError):
            collector.parse_perf_stat(
                perf_lines({"counter-value": "unknown", "event": "cycles"}),
                1,
            )

        invalid_json = (
            '{"event":"cycles","counter-value":1,"variance":NaN}\n',
            '{"event":"cycles","event":"instructions","counter-value":1}\n',
        )
        for text in invalid_json:
            with self.subTest(text=text), self.assertRaises(collector.PerfParseError):
                collector.parse_perf_stat(text, 1)

        for value in (-1, int("9" * 1000)):
            with self.subTest(value_type=type(value).__name__), self.assertRaises(
                collector.PerfParseError
            ):
                collector.parse_perf_stat(
                    perf_lines({"counter-value": value, "event": "cycles"}),
                    1,
                )

    def test_duplicate_normalized_events_are_rejected(self) -> None:
        text = perf_lines(
            {"counter-value": "1", "event": "branch-misses"},
            {"counter-value": "2", "event": "branch_misses"},
        )
        with self.assertRaises(collector.PerfParseError):
            collector.parse_perf_stat(text, 1)

    def test_invalid_repeats_are_rejected(self) -> None:
        for repeat in (0, -1, "", "abc", 1.5, True, 101):
            with self.subTest(repeat=repeat):
                with self.assertRaises(collector.PerfParseError):
                    collector.parse_perf_stat(
                        perf_lines({"counter-value": "1", "event": "cycles"}),
                        repeat,
                    )

    def test_missing_requested_event_is_rejected_by_collection_validation(self) -> None:
        with self.assertRaises(collector.PerfParseError):
            collector._missing_requested_events(
                ("cycles", "instructions"),
                {"cycles"},
            )
        with self.assertRaises(collector.PerfParseError):
            collector._missing_requested_events(
                ("cycles",),
                {"cycles", "instructions"},
            )


class PerfCommandTests(unittest.TestCase):
    def test_command_construction(self) -> None:
        command = collector.build_perf_command(
            "perf",
            "/tmp/perf.json",
            5,
            ("cycles", "branch-misses"),
            ("./workload", "--size", "8"),
        )
        self.assertEqual(
            command,
            [
                "perf",
                "stat",
                "--json-output",
                "--no-big-num",
                "--output",
                "/tmp/perf.json",
                "--repeat",
                "5",
                "--event",
                "cycles,branch-misses",
                "--",
                "./workload",
                "--size",
                "8",
            ],
        )

    def test_empty_command_is_rejected(self) -> None:
        with self.assertRaises(collector.PerfCollectionError):
            collector.build_perf_command("perf", "/tmp/raw.json", 1, ("cycles",), ())

    def test_mocked_linux_subprocess_forwards_workload_output(self) -> None:
        args = argparse.Namespace(
            perf="perf",
            event=["cycles", "instructions"],
            repeat=2,
            scope="dedicated-workload",
            raw_output=None,
            command=["./workload", "--size", "4"],
        )
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "perf.json"
            raw_text = perf_lines(
                {"counter-value": "100", "event": "cycles"},
                {"counter-value": "200", "event": "instructions"},
            )
            args.raw_output = raw_path

            def run_perf(command, **kwargs):
                if command == ["perf", "--version"]:
                    return mock.Mock(
                        returncode=0,
                        stdout="perf version 6.8\n",
                        stderr="",
                    )
                raw_path.write_text(raw_text, encoding="utf-8")
                return mock.Mock(returncode=0)

            with (
                mock.patch.object(collector.platform, "system", return_value="Linux"),
                mock.patch.object(
                    collector.subprocess,
                    "run",
                    side_effect=run_perf,
                ) as run,
                mock.patch.object(
                    collector,
                    "artifact_record",
                    return_value={"kind": "raw", "path": str(raw_path)},
                ),
                mock.patch.object(
                    collector,
                    "build_result_bundle",
                    return_value={"ok": True},
                ),
                mock.patch.object(collector, "write_result") as write_result,
            ):
                document = collector.collect_perf_stat(args)

        self.assertEqual(document, {"ok": True})
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args_list[0].args[0], ["perf", "--version"])
        invocation = run.call_args_list[1].args[0]
        self.assertEqual(invocation[:4], ["perf", "stat", "--json-output", "--no-big-num"])
        self.assertEqual(invocation[-4:], ["--", "./workload", "--size", "4"])
        self.assertEqual(run.call_args_list[0].kwargs["env"]["LC_ALL"], "C")
        self.assertEqual(run.call_args_list[1].kwargs["env"]["LC_ALL"], "C")
        self.assertIs(run.call_args_list[1].kwargs["stdout"], sys.stderr)
        self.assertIs(run.call_args_list[1].kwargs["stderr"], sys.stderr)
        write_result.assert_called_once()

    def test_non_linux_does_not_execute_subprocess(self) -> None:
        args = argparse.Namespace(
            perf="perf",
            event=["cycles"],
            repeat=1,
            scope="process-aggregate",
            raw_output=Path("/tmp/perf.json"),
            command=["./workload"],
        )
        with (
            mock.patch.object(collector.platform, "system", return_value="Darwin"),
            mock.patch.object(collector.subprocess, "run") as run,
        ):
            with self.assertRaises(collector.PerfCollectionError):
                collector.collect_perf_stat(args)
        run.assert_not_called()

    def test_parse_failure_does_not_emit_bundle_or_change_raw_file(self) -> None:
        args = argparse.Namespace(
            perf="perf",
            event=["cycles"],
            repeat=1,
            scope="process-aggregate",
            raw_output=None,
            command=["./workload"],
        )
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "perf.json"
            raw_text = "not-json\n"
            raw_path.write_text("stale\n", encoding="utf-8")
            args.raw_output = raw_path

            def run_perf(command, **kwargs):
                if command == ["perf", "--version"]:
                    return mock.Mock(
                        returncode=0,
                        stdout="perf version 6.8\n",
                        stderr="",
                    )
                raw_path.write_text(raw_text, encoding="utf-8")
                return mock.Mock(returncode=0)

            with (
                mock.patch.object(collector.platform, "system", return_value="Linux"),
                mock.patch.object(
                    collector.subprocess,
                    "run",
                    side_effect=run_perf,
                ),
                mock.patch.object(collector, "build_result_bundle") as build,
            ):
                with self.assertRaises(collector.PerfParseError):
                    collector.collect_perf_stat(args)
            self.assertEqual(raw_path.read_text(encoding="utf-8"), raw_text)
            build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
