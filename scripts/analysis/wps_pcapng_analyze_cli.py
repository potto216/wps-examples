#!/usr/bin/env python3
"""Command-line wrapper for Bluetooth LE pcapng analysis.

Uses `analysis.pcapng_analysis.PcapngAnalyzer` to produce a JSON-serializable
analysis payload.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

# Allow running this script directly from any working directory.
# (The repository root contains the top-level `analysis/` package.)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from analysis.pcapng_analysis import PcapngAnalyzer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze a Bluetooth LE pcapng file and emit a JSON report."
    )
    parser.add_argument("pcapng", help="Path to the .pcapng file to analyze.")
    parser.add_argument(
        "-o",
        "--output",
        help="Write JSON report to this path (defaults to stdout if omitted).",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indent level (default: 2). Use 0 for compact.",
    )
    return parser


def _write_json(payload: Dict[str, Any], *, output_path: Optional[str], indent: int) -> None:
    json_kwargs: Dict[str, Any] = {"sort_keys": True}
    if indent and indent > 0:
        json_kwargs["indent"] = indent

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, **json_kwargs)
            handle.write("\n")
        return

    json.dump(payload, sys.stdout, **json_kwargs)
    sys.stdout.write("\n")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not os.path.exists(args.pcapng):
        raise FileNotFoundError(f"pcapng file not found: {args.pcapng}")

    analyzer = PcapngAnalyzer(args.pcapng)
    payload = analyzer.analyze()

    indent = int(args.indent)
    _write_json(payload, output_path=args.output, indent=indent)


if __name__ == "__main__":
    main()
