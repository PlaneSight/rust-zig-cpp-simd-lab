from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts.result_bundle import (
    REPOSITORY,
    ROOT,
    add_result_arguments,
    artifact_record,
    build_result_bundle,
    paths_alias,
    validate_artifact_output_paths,
    write_result,
)


class ResultBundleTests(unittest.TestCase):
    def parse_args(self, *extra: str) -> argparse.Namespace:
        parser = add_result_arguments(argparse.ArgumentParser())
        return parser.parse_args(
            [
                "--id",
                "run-1",
                "--family",
                "reductions",
                "--kernel",
                "sad_u8",
                "--implementation",
                "portable",
                "--target",
                "x86_64-unknown-linux-gnu",
                "--cpu",
                "test-cpu",
                *extra,
            ]
        )

    def build(self, args: argparse.Namespace, **kwargs: object) -> dict[str, object]:
        with patch("scripts.result_bundle._git_value", side_effect=["deadbeef", "main"]):
            return build_result_bundle(
                args,
                [{"kind": "analysis", "summary": "test", "severity": "info", "evidence": []}],
                **kwargs,
            )

    def test_required_provenance_environment_and_experiment_fields(self) -> None:
        args = self.parse_args(
            "--source",
            "languages/rust/src/lib.rs",
            "--cpu-feature",
            "avx2",
            "--virtualized",
            "yes",
            "--data-type",
            "u8",
            "--lanes",
            "32",
            "--tag",
            "smoke",
            "--notes",
            "created by test",
        )
        document = self.build(args)

        self.assertEqual(document["schema"], "simd-lab-result-v1")
        self.assertEqual(document["provenance"]["repository"], REPOSITORY)
        self.assertEqual(document["provenance"]["commit"], "deadbeef")
        self.assertEqual(document["provenance"]["branch"], "main")
        self.assertEqual(document["provenance"]["source_files"], ["languages/rust/src/lib.rs"])
        self.assertTrue(document["created_at"].endswith("Z"))

        environment = document["environment"]
        self.assertTrue(environment["os"])
        self.assertTrue(environment["arch"])
        self.assertEqual(environment["cpu"], "test-cpu")
        self.assertEqual(environment["cpu_features"], ["avx2"])
        self.assertEqual(environment["target"], "x86_64-unknown-linux-gnu")
        self.assertIs(environment["virtualized"], True)

        experiment = document["experiment"]
        self.assertEqual(experiment["family"], "reductions")
        self.assertEqual(experiment["kernel"], "sad_u8")
        self.assertEqual(experiment["implementation"], "portable")
        self.assertEqual(experiment["data_type"], "u8")
        self.assertEqual(experiment["lanes"], 32)
        self.assertEqual(experiment["parameters"], {})
        self.assertEqual(document["tags"], ["smoke"])
        self.assertEqual(document["notes"], "created by test")

    def test_parameter_values_are_json_and_duplicates_or_malformed_values_fail(self) -> None:
        args = self.parse_args(
            "--parameter",
            "elements=1048576",
            "--parameter",
            'label="u8"',
            "--parameter",
            "enabled=true",
            "--parameter",
            "shape=[1,2]",
        )
        document = self.build(args, parameters={"source": "perf"})
        self.assertEqual(
            document["experiment"]["parameters"],
            {
                "elements": 1048576,
                "label": "u8",
                "enabled": True,
                "shape": [1, 2],
                "source": "perf",
            },
        )

        invalid_parameters = (
            ("--parameter", "elements=1", "--parameter", "elements=2"),
            ("--parameter", "missing-json"),
            ("--parameter", "=1"),
        )
        for parameters in invalid_parameters:
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.parse_args(*parameters)

        with self.assertRaises(ValueError):
            self.build(self.parse_args("--parameter", "x=1"), parameters={"x": 2})

    def test_virtualized_no_and_unknown_map_to_false_and_null(self) -> None:
        self.assertIs(
            self.build(self.parse_args("--virtualized", "no"))["environment"][
                "virtualized"
            ],
            False,
        )
        self.assertIs(
            self.build(self.parse_args())["environment"]["virtualized"],
            None,
        )

    def test_empty_toolchain_is_omitted_and_nonempty_toolchain_is_included(self) -> None:
        without_toolchain = self.build(self.parse_args())
        self.assertNotIn("toolchain", without_toolchain)

        with_toolchain = self.build(
            self.parse_args(
                "--language",
                "rust",
                "--compiler",
                "rustc",
                "--compiler-version",
                "rustc 1.80",
                "--optimization",
                "release",
                "--compiler-flag=-C",
                "--compiler-flag",
                "target-cpu=native",
            )
        )
        self.assertEqual(
            with_toolchain["toolchain"],
            {
                "language": "rust",
                "compiler": "rustc",
                "version": "rustc 1.80",
                "optimization": "release",
                "flags": ["-C", "target-cpu=native"],
            },
        )

    def test_artifact_record_hashes_bytes_and_preserves_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.json"
            payload = b"{\"value\": 42}\n\x00"
            path.write_bytes(payload)
            record = artifact_record(path, "perf-raw", "raw perf JSON")
            record_without_description = artifact_record(path, "perf-raw", None)

        self.assertEqual(record["kind"], "perf-raw")
        self.assertEqual(record["path"], str(path))
        self.assertEqual(record["sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(record["description"], "raw perf JSON")
        self.assertNotIn("description", record_without_description)

    def test_artifact_and_result_paths_must_be_distinct_and_publishable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.json"
            alias = Path(directory) / "alias.json"
            raw.write_text("raw", encoding="utf-8")
            os.link(raw, alias)
            self.assertTrue(paths_alias(raw, alias))
            with self.assertRaisesRegex(ValueError, "different files"):
                validate_artifact_output_paths(raw, alias, tool="test")

        with self.assertRaisesRegex(ValueError, "must not be stored"):
            validate_artifact_output_paths(
                ROOT / "results" / "runs" / "raw.json",
                None,
                tool="test",
            )
        with self.assertRaisesRegex(ValueError, "require raw output"):
            validate_artifact_output_paths(
                Path("/tmp/raw.json"),
                ROOT / "results" / "runs" / "result.json",
                tool="test",
            )

        validate_artifact_output_paths(
            ROOT / "results" / "artifacts" / "run" / "raw.json",
            ROOT / "results" / "runs" / "result.json",
            tool="test",
        )

    def test_write_result_uses_newline_json_on_stdout_or_output_path(self) -> None:
        document = {"schema": "simd-lab-result-v1", "observations": []}
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            write_result(document, argparse.Namespace(output=None))
        self.assertTrue(stdout.getvalue().endswith("\n"))
        self.assertEqual(json.loads(stdout.getvalue()), document)
        self.assertEqual(stdout.getvalue().count("\n"), 1)

        pretty_stdout = io.StringIO()
        with redirect_stdout(pretty_stdout):
            write_result(document, argparse.Namespace(output=None, pretty=True))
        self.assertGreater(pretty_stdout.getvalue().count("\n"), 1)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "result.json"
            write_result(document, argparse.Namespace(output=output))
            rendered = output.read_text(encoding="utf-8")
        self.assertTrue(rendered.endswith("\n"))
        self.assertEqual(json.loads(rendered), document)

    def test_write_result_rejects_nonfinite_json(self) -> None:
        document = {"schema": "simd-lab-result-v1", "observations": [float("nan")]}

        with self.assertRaises(ValueError):
            write_result(document, argparse.Namespace(output=None))


if __name__ == "__main__":
    unittest.main()
