#!/usr/bin/env python3
"""Extract simple, reproducible codegen metrics from compiler assembly output.

This is intentionally conservative: it counts instruction-looking lines and a
small set of SIMD/conversion mnemonics that are useful for the lab. It is not a
replacement for llvm-mca, uiCA, perf, or hardware counters.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

TRACKED_PREFIXES = (
    # x86 SIMD / FP16
    "vcvtph2ps",
    "vcvtps2ph",
    "vmin",
    "vmax",
    "vpmin",
    "vpmax",
    "vadd",
    "vsub",
    "vmul",
    "vfmadd",
    "vpadd",
    "vpsub",
    "vpmul",
    "vpack",
    "vpunpck",
    "vperm",
    "vshuffle",
    "vpshuf",
    "vpsadbw",
    "psadbw",
    "vpmovzx",
    "vpmovsx",
    # AArch64 AdvSIMD / FP16 / widening-reduction idioms
    "fmin",
    "fmax",
    "fcvt",
    "uabd",
    "uabal",
    "uadalp",
    "uaddlp",
    "uaddlv",
    "sadalp",
    "saddlp",
    "saddlv",
    "ushll",
    "sshll",
    "xtn",
    # WebAssembly SIMD opcode families (normalized before the first '.')
    "v128",
    "i8x16",
    "i16x8",
    "i32x4",
    "i64x2",
    "f32x4",
    "f64x2",
)

LABEL_RE = re.compile(r"^[.$A-Za-z_][\w.$@]*:\s*(?:[#;].*)?$")
DIRECTIVE_RE = re.compile(r"^\s*\.")
COMMENT_RE = re.compile(r"^\s*(?:[#;]|//)")
INSTRUCTION_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_.]*)\b")

WASM_VECTOR_PREFIXES = ("v128", "i8x16", "i16x8", "i32x4", "i64x2", "f32x4", "f64x2")
ARM_VECTOR_HINTS = {
    "fmin", "fmax", "fcvt", "uabd", "uabal", "uadalp", "uaddlp", "uaddlv",
    "sadalp", "saddlp", "saddlv", "ushll", "sshll", "xtn",
}


def normalize_mnemonic(mnemonic: str) -> str:
    return mnemonic.lower().split(".", 1)[0]


def analyze(text: str) -> dict[str, object]:
    instructions: Counter[str] = Counter()
    tracked: Counter[str] = Counter()

    for raw in text.splitlines():
        line = raw.strip()
        if not line or LABEL_RE.match(line) or DIRECTIVE_RE.match(line) or COMMENT_RE.match(line):
            continue

        match = INSTRUCTION_RE.match(raw)
        if not match:
            continue

        mnemonic = normalize_mnemonic(match.group(1))
        instructions[mnemonic] += 1
        for prefix in TRACKED_PREFIXES:
            if mnemonic.startswith(prefix):
                tracked[prefix] += 1
                break

    vectorish = 0
    for name, count in instructions.items():
        if (
            name.startswith("v")
            or name.startswith(WASM_VECTOR_PREFIXES)
            or name in ARM_VECTOR_HINTS
        ):
            vectorish += count

    return {
        "instruction_count": sum(instructions.values()),
        "unique_mnemonics": len(instructions),
        "vector_instruction_count": vectorish,
        "tracked": dict(sorted(tracked.items())),
        "top_mnemonics": dict(instructions.most_common(20)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("assembly", type=Path, nargs="+")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    results = {}
    for path in args.assembly:
        results[str(path)] = analyze(path.read_text(encoding="utf-8", errors="replace"))

    print(json.dumps(results, indent=2 if args.pretty else None, sort_keys=True))


if __name__ == "__main__":
    main()
