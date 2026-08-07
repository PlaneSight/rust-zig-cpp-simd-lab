"""Unit tests for the bounded llvm-mca adapter.

These tests only exercise pure parsing and mocked subprocess calls.  They never
invoke llvm-mca or any other external command.
"""

from __future__ import annotations

import argparse

import json
import sys
import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_mca  # noqa: E402


class AssemblyBoundsTests(unittest.TestCase):
    def test_markers_are_required_and_matched(self) -> None:
        source = (
            ".intel_syntax noprefix\n"
            "# LLVM-MCA-BEGIN first loop\n"
            "first:\n"
            "  add eax, ebx\n"
            "# LLVM-MCA-END first loop\n"
            "# LLVM-MCA-BEGIN second\n"
            "second:\n"
            "  ret\n"
            "# LLVM-MCA-END\n"
        )
        markers = analyze_mca.find_marker_regions(source)
        self.assertEqual(
            markers,
            (
                analyze_mca.MarkerRegion("first loop", 2, 5),
                analyze_mca.MarkerRegion("second", 6, 9),
            ),
        )
        bounded = analyze_mca.bound_assembly(source)
        self.assertEqual(bounded.mode, "markers")
        self.assertEqual(bounded.text, source)
        self.assertEqual([entry.name for entry in bounded.markers], ["first loop", "second"])

    def test_marker_text_must_be_a_comment_directive(self) -> None:
        source = '.ascii "LLVM-MCA-BEGIN false"\n  nop\n'

        self.assertEqual(analyze_mca.find_marker_regions(source), ())
        with self.assertRaises(analyze_mca.AssemblyBoundsError):
            analyze_mca.bound_assembly(source)

    def test_label_region_keeps_dialect_and_excludes_end(self) -> None:
        source = (
            ".intel_syntax noprefix\n"
            ".section .text\n"
            ".globl public_name\n"
            ".type public_name,@function\n"
            "public_name:\n"
            "  mov eax, edi\n"
            "  add eax, 1\n"
            ".size public_name, .-public_name\n"
            "end_label:\n"
            "  ret\n"
        )
        bounded = analyze_mca.bound_assembly(
            source, start_label="public_name", end_label="end_label"
        )
        self.assertEqual(bounded.start_line, 5)
        self.assertEqual(bounded.end_line, 9)
        self.assertIn(".intel_syntax noprefix", bounded.text)
        self.assertIn("public_name:\n", bounded.text)
        self.assertIn("  add eax, 1", bounded.text)
        self.assertNotIn(".section", bounded.text)
        self.assertNotIn(".globl", bounded.text)
        self.assertNotIn(".size", bounded.text)
        self.assertNotIn("end_label:", bounded.text)
        self.assertNotIn("  ret", bounded.text)

        local_source = (
            ".text\n"
            ".Lloop:\n"
            "  add eax, 1\n"
            "  jne .Lloop\n"
            ".Lend:\n"
        )
        local = analyze_mca.bound_assembly(
            local_source, start_label=".Lloop", end_label=".Lend"
        )
        self.assertIn(".Lloop:\n", local.text)

        semantic_source = (
            ".intel_syntax noprefix\n"
            "start:\n"
            "  count = 4\n"
            "  .rept 4\n"
            "  nop\n"
            "  .endr\n"
            "  .cfi_startproc\n"
            "end:\n"
        )
        semantic = analyze_mca.bound_assembly(
            semantic_source, start_label="start", end_label="end"
        )
        self.assertIn("  count = 4\n", semantic.text)
        self.assertIn("  .rept 4\n", semantic.text)
        self.assertIn("  .endr\n", semantic.text)
        self.assertNotIn(".cfi_startproc", semantic.text)
        self.assertIn("  jne .Lloop\n", local.text)

    def test_invalid_bounds_are_rejected(self) -> None:
        source = "start:\n  nop\nend:\n"
        with self.assertRaises(analyze_mca.AssemblyBoundsError):
            analyze_mca.find_label_bounds(source, "missing", "end")
        with self.assertRaises(analyze_mca.AssemblyBoundsError):
            analyze_mca.find_label_bounds("start:\nstart:\nend:\n", "start", "end")
        with self.assertRaises(analyze_mca.AssemblyBoundsError):
            analyze_mca.find_label_bounds("end:\nstart:\n", "start", "end")
        with self.assertRaises(analyze_mca.AssemblyBoundsError):
            analyze_mca.bound_assembly("nop\n")
        with self.assertRaises(analyze_mca.AssemblyBoundsError):
            analyze_mca.find_marker_regions("# LLVM-MCA-BEGIN x\n nop\n")


class ReportParserTests(unittest.TestCase):
    @staticmethod
    def region(name: str, *, summary: dict[str, object] | None = None) -> dict[str, object]:
        return {
            "Name": name,
            "SummaryView": summary or {"BlockRThroughput": 1.25, "TotalCycles": 8},
            "ResourcePressureView": {"ResourcePressure": [{"resource": "ports"}]},
        }

    def test_single_and_named_multiple_region_selection(self) -> None:
        single = {"CodeRegions": [self.region("only")]}
        parsed = analyze_mca.parse_mca_json(json.dumps(single))
        self.assertEqual(parsed.name, "only")
        with self.assertRaises(analyze_mca.McaError):
            analyze_mca.parse_mca_json(
                '{"CodeRegions":[],"unmodeled":NaN}'
            )
        with self.assertRaises(analyze_mca.McaError):
            analyze_mca.parse_mca_json(
                '{"CodeRegions":[],"CodeRegions":[]}'
            )

        multiple = {"CodeRegions": [self.region("a"), self.region("b")]}
        with self.assertRaises(analyze_mca.McaError):
            analyze_mca.parse_mca_json(multiple)
        selected = analyze_mca.parse_mca_json(multiple, requested_region="b")
        self.assertEqual(selected.name, "b")
        with self.assertRaises(analyze_mca.McaError):
            analyze_mca.parse_mca_json(multiple, requested_region="missing")

    def test_required_views_and_json_shape(self) -> None:
        for missing in ("SummaryView", "ResourcePressureView"):
            region = self.region("bad")
            del region[missing]
            with self.subTest(missing=missing), self.assertRaises(analyze_mca.McaError):
                analyze_mca.parse_mca_json({"CodeRegions": [region]})
        with self.assertRaises(analyze_mca.McaError):
            analyze_mca.parse_mca_json("not json")
        with self.assertRaises(analyze_mca.McaError):
            analyze_mca.parse_mca_json({"CodeRegions": []})

    def test_summary_formatting_is_stable(self) -> None:
        parsed = analyze_mca.parse_mca_json(
            {"CodeRegions": [self.region("loop", summary={
                "BlockRThroughput": 0.5,
                "TotalCycles": 12,
                "TotalInstructions": 24,
                "TotalNumIterations": 4,
            })]}
        )
        self.assertEqual(
            analyze_mca.format_analysis_summary(parsed),
            "llvm-mca region 'loop': block throughput=0.5; total cycles=12; "
            "total instructions=24; iterations=4; resource pressure entries=1",
        )


class AnalysisWorkflowTests(unittest.TestCase):
    def test_parse_failure_preserves_exact_raw_report_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "loop.s"
            raw = Path(directory) / "mca.json"
            source.write_text(
                "# LLVM-MCA-BEGIN loop\n  nop\n# LLVM-MCA-END loop\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                assembly=source,
                raw_output=raw,
                output=None,
                start_label=None,
                end_label=None,
                llvm_mca="llvm-mca",
                iterations=10,
                target="x86_64-unknown-linux-gnu",
                cpu="skylake",
                mattr=[],
                region="loop",
            )
            run = analyze_mca.McaRun(
                report_text="not json\r\n",
                report_bytes=b"not json\r\n",
                tool_version="LLVM version 18",
                command=("llvm-mca",),
            )

            with patch.object(analyze_mca, "run_llvm_mca", return_value=run):
                with self.assertRaises(analyze_mca.McaError):
                    analyze_mca.run_analysis(args)

            self.assertEqual(raw.read_bytes(), b"not json\r\n")

    def test_output_paths_must_not_overwrite_input_assembly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "loop.s"
            source.write_text(
                "# LLVM-MCA-BEGIN loop\n  nop\n# LLVM-MCA-END loop\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                assembly=source,
                raw_output=source,
                output=None,
                start_label=None,
                end_label=None,
                llvm_mca="llvm-mca",
                iterations=10,
                target="x86_64-unknown-linux-gnu",
                cpu="skylake",
                mattr=[],
                region="loop",
            )

            with patch.object(analyze_mca, "run_llvm_mca") as run:
                with self.assertRaisesRegex(analyze_mca.McaError, "input assembly"):
                    analyze_mca.run_analysis(args)
            run.assert_not_called()


class SubprocessTests(unittest.TestCase):
    def test_subprocess_failure_is_reported(self) -> None:
        failed = CompletedProcess(
            ["llvm-mca", "--version"], returncode=1, stdout="", stderr="unsupported CPU"
        )
        with patch.object(analyze_mca.subprocess, "run", return_value=failed):
            with self.assertRaises(analyze_mca.McaError):
                analyze_mca.run_llvm_mca(
                    "# LLVM-MCA-BEGIN\nnop\n# LLVM-MCA-END\n",
                    target="x86_64",
                    cpu="skylake",
                )

    def test_success_uses_fixed_json_command_and_stdin(self) -> None:
        report = json.dumps(
            {
                "CodeRegions": [
                    {
                        "Name": "only",
                        "SummaryView": {"BlockRThroughput": 1.25, "TotalCycles": 8},
                        "ResourcePressureView": {
                            "ResourcePressure": [{"resource": "ports"}]
                        },
                    }
                ]
            }
        )
        version = CompletedProcess(
            ["llvm-mca", "--version"], returncode=0, stdout="LLVM version 18.1\n", stderr=""
        )
        result = CompletedProcess(
            ["llvm-mca"], returncode=0, stdout=report, stderr=""
        )
        with patch.object(analyze_mca.subprocess, "run", side_effect=[version, result]) as run:
            completed = analyze_mca.run_llvm_mca(
                "# LLVM-MCA-BEGIN\nnop\n# LLVM-MCA-END\n",
                executable="llvm-mca",
                iterations=17,
                target="x86_64-unknown-linux-gnu",
                cpu="skylake",
                mattrs=["+avx2", "+fma"],
            )
        self.assertEqual(completed.tool_version, "LLVM version 18.1")
        self.assertEqual(
            list(completed.command),
            [
                "llvm-mca",
                "-json",
                "-iterations",
                "17",
                "-mtriple",
                "x86_64-unknown-linux-gnu",
                "-mcpu",
                "skylake",
                "-mattr",
                "+avx2,+fma",
                "-",
            ],
        )
        self.assertEqual(run.call_count, 2)
        self.assertEqual(
            run.call_args_list[1].kwargs["input"],
            b"# LLVM-MCA-BEGIN\nnop\n# LLVM-MCA-END\n",
        )


if __name__ == "__main__":
    unittest.main()
