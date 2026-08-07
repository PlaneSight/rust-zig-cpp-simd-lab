from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_results_catalog import ROOT, build, summarize


class BuildResultsCatalogTests(unittest.TestCase):
    def make_bundle(self, *, observations=None, artifacts=None):
        counter = {
            "kind": "counters",
            "tool": "perf stat",
            "metrics": [
                {
                    "name": "instructions",
                    "value": 42.0,
                    "unit": "count",
                    "aggregation": "mean",
                    "lower": 40.0,
                    "upper": 44.0,
                    "samples": 5,
                }
            ],
        }
        runtime = {
            "kind": "runtime",
            "metrics": [
                {
                    "name": "ns_per_element",
                    "value": 1.25,
                    "unit": "ns",
                    "aggregation": "median",
                    "lower": 1.1,
                    "upper": 1.4,
                    "samples": 5,
                }
            ],
        }
        default_observations = [runtime, counter]
        bundle = {
            "schema": "simd-lab-result-v1",
            "id": "unit-test-result",
            "created_at": "2026-08-07T00:00:00Z",
            "provenance": {"commit": "0123456789abcdef"},
            "environment": {
                "target": "x86_64-unknown-linux-gnu",
                "arch": "x86_64",
                "cpu": "test-cpu",
            },
            "toolchain": {
                "language": "rust",
                "compiler": "rustc",
                "version": "1.90.0",
                "optimization": "release",
            },
            "experiment": {
                "family": "unit",
                "kernel": "axpy",
                "implementation": "scalar",
                "variant": "default",
                "data_type": "f32",
                "isa": "avx2",
                "semantics": "ieee",
                "dataset": "tiny",
            },
            "observations": default_observations if observations is None else observations,
            "artifacts": artifacts
            if artifacts is not None
            else [
                {
                    "kind": "perf-json",
                    "path": "results/artifacts/unit-test/perf.json",
                    "sha256": "deadbeef",
                    "description": "Raw perf counters",
                }
            ],
            "tags": ["test"],
            "notes": "catalog test",
        }
        return bundle

    def test_preserves_artifacts_and_full_counter_observations(self):
        bundle = self.make_bundle()
        row = summarize(ROOT / "results" / "runs" / "unit-test-result.json", bundle)

        self.assertEqual(row["artifacts"], bundle["artifacts"])
        self.assertEqual(row["counter_observations"], [bundle["observations"][1]])
        self.assertEqual(row["counter_observations"][0]["tool"], "perf stat")
        self.assertEqual(row["analysis"], [])

    def test_summarize_accepts_relative_repository_path(self):
        bundle = self.make_bundle()
        path = Path("results") / "runs" / "unit-test-result.json"

        row = summarize(path, bundle)

        self.assertEqual(row["source"], "results/runs/unit-test-result.json")

    def test_summarize_accepts_external_absolute_path(self):
        bundle = self.make_bundle()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unit-test-result.json"

            row = summarize(path, bundle)

        self.assertEqual(row["source"], str(path.resolve()).replace("\\", "/"))

    def test_retains_legacy_flattened_runtime_and_counter_fields(self):
        bundle = self.make_bundle()
        row = summarize(ROOT / "results" / "runs" / "unit-test-result.json", bundle)

        self.assertEqual(
            row["runtime"],
            {
                "ns_per_element": {
                    "value": 1.25,
                    "unit": "ns",
                    "aggregation": "median",
                    "lower": 1.1,
                    "upper": 1.4,
                    "samples": 5,
                }
            },
        )
        self.assertEqual(
            row["counters"],
            {
                "instructions": {
                    "value": 42.0,
                    "unit": "count",
                    "aggregation": "mean",
                    "lower": 40.0,
                    "upper": 44.0,
                    "samples": 5,
                }
            },
        )
        self.assertEqual(row["family"], "unit")
        self.assertEqual(row["kernel"], "axpy")
        self.assertEqual(row["implementation"], "scalar")
        self.assertEqual(row["semantics_contract"], "ieee")

    def test_analysis_evidence_remains_on_existing_analysis_path(self):
        analysis = {
            "kind": "analysis",
            "summary": "Counter evidence is available",
            "severity": "interesting",
            "evidence": ["results/artifacts/unit-test/perf.json"],
        }
        bundle = self.make_bundle(observations=[analysis])
        row = summarize(ROOT / "results" / "runs" / "unit-test-result.json", bundle)

        self.assertEqual(row["analysis"], [analysis])
        self.assertEqual(row["artifacts"][0]["path"], analysis["evidence"][0])
        self.assertEqual(row["counter_observations"], [])

    def test_build_counts_observation_kinds_and_reports_malformed_bundles(self):
        codegen = {
            "kind": "codegen",
            "instruction_count": 3,
            "vector_instruction_count": 1,
            "unique_mnemonics": 2,
        }
        semantics = {"kind": "semantics", "status": "match"}
        analysis = {
            "kind": "analysis",
            "summary": "derived evidence",
            "severity": "info",
            "evidence": ["results/artifacts/unit-test/perf.json"],
        }
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            runs_dir = Path(temp_dir)
            first = self.make_bundle()
            second = self.make_bundle(
                observations=[
                    codegen,
                    semantics,
                    analysis,
                    {"kind": "counters", "tool": "llvm", "metrics": []},
                ]
            )
            (runs_dir / "first.json").write_text(json.dumps(first), encoding="utf-8")
            (runs_dir / "second.json").write_text(json.dumps(second), encoding="utf-8")
            malformed_path = runs_dir / "malformed.json"
            malformed_path.write_text(
                json.dumps({"schema": "simd-lab-result-v1", "id": "malformed"}),
                encoding="utf-8",
            )

            catalog = build(runs_dir)

        self.assertEqual(catalog["count"], 2)
        self.assertEqual(
            catalog["observation_counts"],
            {
                "analysis": 1,
                "codegen": 1,
                "counters": 2,
                "runtime": 1,
                "semantics": 1,
            },
        )
        self.assertEqual(len(catalog["errors"]), 1)
        self.assertEqual(catalog["errors"][0]["path"], str(malformed_path))
        self.assertIn("missing required key", catalog["errors"][0]["error"])


if __name__ == "__main__":
    unittest.main()
