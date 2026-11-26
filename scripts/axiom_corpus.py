#!/usr/bin/env python3
"""Rewrite a training corpus to the AXIOM connector algebra.

AXIOM replaces grammatical connectors with one-byte mathematical operators and
canonicalises number words, typography and whitespace. The prepared file is
shorter, and the byte-level model reads the same content in fewer tokens.

Usage:
    python scripts/axiom_corpus.py --data corpus.txt --out trained-model/corpus.axiom.bin
    python scripts/axiom_corpus.py --data corpus.txt --report
    python scripts/axiom_corpus.py --lexicon

The prepared corpus is what ``python -m mariana train --axiom --data ...``
produces internally; use this script to inspect or ship the rewrite separately.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mariana import axiom


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="axiom_corpus",
        description="Rewrite a text corpus to one-byte AXIOM connector operators.",
    )
    p.add_argument("--data", type=str, help="source corpus (UTF-8 text)")
    p.add_argument("--out", type=str, default=None,
                   help="prepared corpus path; omit to measure without writing")
    p.add_argument("--profile", type=str, default="auto",
                   choices=list(axiom.PROFILES),
                   help="auto (default) detects text/language/sequence; compact and "
                        "reversible force English connectors; lossless keeps text exact")
    p.add_argument("--max-passes", type=int, default=4,
                   help="fixed-point iteration limit (default: 4)")
    p.add_argument("--report", action="store_true", help="print the full JSON report")
    p.add_argument("--display", type=int, default=0, metavar="CHARS",
                   help="print the first CHARS characters of the codec surface form")
    p.add_argument("--lexicon", action="store_true", help="print the operator alphabet")
    return p


def print_lexicon() -> None:
    print(f"{'code':<6} {'glyph':<6} {'operator':<12} canonical / variants")
    for entry in axiom.operator_table():
        variants = ", ".join(entry["variants"]) or "-"
        print(
            f"{entry['code']:<6} {entry['glyph']:<6} {entry['name']:<12} "
            f"{entry['canonical']} / {variants}"
        )


def main() -> None:
    args = build_parser().parse_args()
    if args.lexicon:
        print_lexicon()
        return
    if not args.data:
        raise SystemExit("--data is required unless --lexicon is used")
    if not os.path.isfile(args.data):
        raise SystemExit(f"corpus not found: {args.data}")
    if args.max_passes < 1:
        raise SystemExit("--max-passes must be >= 1")

    stats = axiom.rewrite_file(
        args.data, args.out, profile=args.profile, max_passes=args.max_passes
    )
    print(axiom.summary(stats))
    print(
        f"Codec: {stats.codec.mode}"
        + (f"/{stats.codec.language}" if stats.codec.mode == "text" else "")
        + (f" · {len(stats.codec.dictionary)} dictionary entries"
           if stats.codec.dictionary else "")
        + (f" · {stats.codec.bits}-bit packed"
           if stats.codec.mode in ("dna", "protein") else "")
    )
    if args.out:
        print(f"Wrote {args.out}")
    if stats.operators:
        top = ", ".join(f"{name} {count}" for name, count in stats.operators[:8])
        print(f"Operators: {top}")
    if stats.residual:
        top = ", ".join(f"{word} {count}" for word, count in stats.residual[:8])
        print(f"Left to the model: {top}")
    if args.report:
        print(json.dumps(stats.to_dict(), indent=2, sort_keys=True))
    if args.display and args.out:
        with open(args.out, "rb") as handle:
            head = handle.read(max(4096, args.display * 4))
        print("\n--- AXIOM surface form ---")
        print(axiom.display_bytes(head, stats.codec)[:args.display])


if __name__ == "__main__":
    main()
