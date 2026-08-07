"""Tests for the reviewed-baseline codegen regression policy."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import check_codegen_regressions  # noqa: E402


class CheckerLogicTests(unittest.TestCase):
    @staticmethod
    def manifest(*, instruction_count: int = 10, vector_count: int = 4, tracked: dict | None = None) -> dict:
        return {
            "schema": "simd-lab-codegen-v1",
            "target_profile": "x86-64-v3",
            "metrics": {
                "/tmp/sad-x86-64-v3.s": {
                    "instruction_count": instruction_count,
                    "vector_instruction_count": vector_count,
                    "tracked": tracked or {"vpmovzx": 2},
                }
            },
        }

    @staticmethod
    def policy() -> dict:
        return {
            "schema": "simd-lab-codegen-policy-v1",
            "defaults": {
                "max_instruction_growth_ratio": 1.35,
                "max_vector_instruction_growth_ratio": 1.50,
            },
            "rules": [
                {
                    "match": "sad-x86-64-v3",
                    "tracked_max_growth": {"vpmovzx": 1},
                }
            ],
        }

    def test_manifest_within_limits_passes(self) -> None:
        result = check_codegen_regressions.compare_manifests(
            self.manifest(),
            self.manifest(instruction_count=13, vector_count=6, tracked={"vpmovzx": 3}),
            self.policy(),
        )
        self.assertEqual(result["failures"], [])
        self.assertEqual(result["observations"][0]["tracked_deltas"]["vpmovzx"], 1)

    def test_instruction_vector_and_tracked_growth_fail(self) -> None:
        result = check_codegen_regressions.compare_manifests(
            self.manifest(),
            self.manifest(instruction_count=14, vector_count=7, tracked={"vpmovzx": 4}),
            self.policy(),
        )
        self.assertEqual(
            {failure["kind"] for failure in result["failures"]},
            {"instruction_growth", "vector_instruction_growth", "tracked_mnemonic_growth"},
        )

    def test_missing_metric_fails_closed(self) -> None:
        candidate = self.manifest()
        candidate["metrics"] = {}
        with self.assertRaisesRegex(ValueError, "metrics must be a non-empty object"):
            check_codegen_regressions.compare_manifests(
                self.manifest(), candidate, self.policy()
            )

    def test_invalid_schema_fails_closed(self) -> None:
        candidate = self.manifest()
        candidate["schema"] = "wrong-schema"
        with self.assertRaisesRegex(ValueError, "schema must be 'simd-lab-codegen-v1'"):
            check_codegen_regressions.compare_manifests(
                self.manifest(), candidate, self.policy()
            )

    def test_removed_metric_fails_closed(self) -> None:
        candidate = self.manifest()
        candidate["metrics"]["/tmp/other.s"] = candidate["metrics"].pop(
            "/tmp/sad-x86-64-v3.s"
        )
        result = check_codegen_regressions.compare_manifests(
            self.manifest(), candidate, self.policy()
        )
        self.assertEqual(result["failures"], [{"file": "sad-x86-64-v3.s", "kind": "missing_metric"}])


class CheckerCliTests(unittest.TestCase):
    def test_cli_reports_invalid_input_and_returns_two(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            policy = root / "policy.json"
            baseline.write_text("{}", encoding="utf-8")
            candidate.write_text("{}", encoding="utf-8")
            policy.write_text(json.dumps(CheckerLogicTests.policy()), encoding="utf-8")

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = check_codegen_regressions.run(
                    [str(baseline), str(candidate), "--policy", str(policy)]
                )

            result = json.loads(output.getvalue())
            self.assertEqual(status, 2)
            self.assertEqual(result["failures"][0]["kind"], "invalid_input")

    def test_cli_reports_missing_input_and_returns_two(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = root / "policy.json"
            policy.write_text(json.dumps(CheckerLogicTests.policy()), encoding="utf-8")

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = check_codegen_regressions.run(
                    [
                        str(root / "missing-baseline.json"),
                        str(root / "missing-candidate.json"),
                        "--policy",
                        str(policy),
                    ]
                )

            result = json.loads(output.getvalue())
            self.assertEqual(status, 2)
            self.assertEqual(result["failures"][0]["kind"], "invalid_input")

    def test_cli_returns_zero_for_known_good_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            policy = root / "policy.json"
            payload = CheckerLogicTests.manifest()
            baseline.write_text(json.dumps(payload), encoding="utf-8")
            candidate.write_text(json.dumps(payload), encoding="utf-8")
            policy.write_text(json.dumps(CheckerLogicTests.policy()), encoding="utf-8")

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = check_codegen_regressions.run(
                    [str(baseline), str(candidate), "--policy", str(policy)]
                )

            result = json.loads(output.getvalue())
            self.assertEqual(status, 0)
            self.assertEqual(result["failures"], [])


if __name__ == "__main__":
    unittest.main()
