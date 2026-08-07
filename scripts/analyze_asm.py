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
    # x86 SIMD / FP16 / saturation
    "vcvtph2ps",
    "vcvtps2ph",
    "vcvtdq2ps",
    "vcvtps2dq",
    "vmin",
    "vmax",
    "vpmin",
    "vpmax",
    "vadd",
    "vsub",
    "vmul",
    "vfmadd",
    "vpaddusb",
    "vpadd",
    "vpsub",
    "vpmul",
    "vpack",
    "vpunpck",
    "vperm",
    "vshuffle",
    "vpshuf",
    "vpsadbw",
    "vpmaddwd",
    "vpmaddubsw",
    "vpdpbusd",
    "vpdpwssd",
    "psadbw",
    "vpmovzx",
    "vpmovsx",
    "vpmovwb",
    "vpmovuswb",
    "vpmovdw",
    "vpmovusdw",
    "vpmovdb",
    "vpmovusdb",
    # AArch64 AdvSIMD / FP16 / widening-reduction / saturation
    "fmin",
    "fmax",
    "fcvt",
    "scvtf",
    "ucvtf",
    "fcvtz",
    "fcvtn",
    "fcvta",
    "uabd",
    "uabal",
    "uadalp",
    "uaddl",
    "saddl",
    "uaddlp",
    "uaddlv",
    "sadalp",
    "saddlp",
    "saddlv",
    "ushll",
    "sshll",
    "uqadd",
    "sqadd",
    "xtn",
    "uqxtn",
    "sqxtn",
    "shrn",
    "rshrn",
    "smull",
    "umull",
    "smlal",
    "umlal",
    "sdot",
    "udot",
    "usdot",
    "zip",
    "uzp",
    "fmla",
    "fmul",
    # WebAssembly SIMD exact/high-signal operations then broad families
    "i8x16.add_sat_u",
    "i8x16.add_sat_s",
    "i16x8.extmul_low_i8x16_s",
    "i16x8.extmul_high_i8x16_s",
    "i16x8.extmul_low_i8x16_u",
    "i16x8.extmul_high_i8x16_u",
    "i32x4.extmul_low_i16x8_s",
    "i32x4.extmul_high_i16x8_s",
    "i32x4.extmul_low_i16x8_u",
    "i32x4.extmul_high_i16x8_u",
    "i32x4.dot_i16x8_s",
    "v128",
    "i8x16",
    "i16x8",
    "i32x4",
    "i64x2",
    "f32x4",
    "f64x2",
)
TRACKED_PREFIXES_LONGEST_FIRST = tuple(
    sorted(TRACKED_PREFIXES, key=len, reverse=True)
)

LABEL_RE = re.compile(r"^[.$A-Za-z_][\w.$@]*:\s*(?:(?:[#;]|//).*)?$")
DIRECTIVE_RE = re.compile(r"^\s*\.")
COMMENT_RE = re.compile(r"^\s*(?:[#;]|//)")
INSTRUCTION_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_.]*)\b")
SYMBOL_ASSIGN_RE = re.compile(r"^[.$A-Za-z_][\w.$@]*\s*=\s*")

WASM_VECTOR_PREFIXES = ("v128", "i8x16", "i16x8", "i32x4", "i64x2", "f32x4", "f64x2")
ARM_VECTOR_HINTS = {
    "fmin", "fmax", "fcvt", "scvtf", "ucvtf", "fcvtz", "fcvtn", "fcvta",
    "uabd", "uabal", "uadalp", "uaddl", "uaddlp", "uaddlv", "sadalp", "saddlp",
    "saddl",
    "saddlv", "ushll", "sshll", "uqadd", "sqadd", "xtn", "uqxtn", "sqxtn",
    "shrn", "rshrn", "smull", "umull", "smlal", "umlal", "sdot", "udot",
    "usdot", "zip", "uzp", "fmla", "fmul",
}


def normalize_mnemonic(mnemonic: str) -> str:
    return mnemonic.lower().split(".", 1)[0]


def analyze(text: str) -> dict[str, object]:
    instructions: Counter[str] = Counter()
    tracked: Counter[str] = Counter()

    for raw in text.splitlines():
        line = raw.strip()
        if (
            not line
            or LABEL_RE.match(line)
            or DIRECTIVE_RE.match(line)
            or COMMENT_RE.match(line)
            or SYMBOL_ASSIGN_RE.match(line)
        ):
            continue

        match = INSTRUCTION_RE.match(raw)
        if not match:
            continue

        raw_mnemonic = match.group(1).lower()
        mnemonic = normalize_mnemonic(raw_mnemonic)
        instructions[mnemonic] += 1
        for prefix in TRACKED_PREFIXES_LONGEST_FIRST:
            if raw_mnemonic.startswith(prefix) or mnemonic.startswith(prefix):
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
