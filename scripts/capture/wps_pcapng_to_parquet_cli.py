#!/usr/bin/env python3
"""Convert .pcapng files into one-row-per-packet parquet datasets."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lib.io import convert_pcapng_path_to_parquet, iter_pcapng_files, parquet_path_for

LOG_LEVELS = {"debug", "info", "warning", "error", "critical"}


def configure_logging(log_level: str, log_file: Optional[str]) -> logging.Logger:
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )
    return logging.getLogger("wps_pcapng_to_parquet_cli")


def validate_args(args: argparse.Namespace) -> None:
    if args.log_level not in LOG_LEVELS:
        raise ValueError(f"Invalid log level '{args.log_level}'. Expected one of {sorted(LOG_LEVELS)}.")
    if not args.input_path:
        raise ValueError("input_path is required.")
    if not os.path.exists(args.input_path):
        raise ValueError(f"Path does not exist: {args.input_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert .pcapng files into parquet DataFrames where each row represents one packet. "
            "Output .parquet files are saved next to the source .pcapng files."
        )
    )
    parser.add_argument("input_path", help="Path to a .pcapng file or directory containing .pcapng files.")
    parser.add_argument("--recursive", action="store_true", help="Recursively search for .pcapng files.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip conversion if the .parquet output already exists.",
    )
    parser.add_argument("--log-level", default="info", help="Log level (debug, info, warning, error, critical).")
    parser.add_argument("--log-file", help="Optional log file path.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.log_level = args.log_level.lower()

    validate_args(args)
    logger = configure_logging(args.log_level, args.log_file)

    pcapng_files = sorted(iter_pcapng_files(args.input_path, args.recursive))
    if not pcapng_files:
        logger.warning("No .pcapng files found under %s", args.input_path)
        return

    converted = 0
    skipped = 0

    for pcapng_path in pcapng_files:
        parquet_path = parquet_path_for(pcapng_path)
        if args.skip_existing and os.path.exists(parquet_path):
            logger.info("Skipping %s (.parquet already exists)", pcapng_path)
            skipped += 1
            continue

        logger.info("Converting %s -> %s", pcapng_path, parquet_path)
        convert_pcapng_path_to_parquet(pcapng_path, parquet_path)
        converted += 1

    logger.info("Done. Converted=%d Skipped=%d Total=%d", converted, skipped, len(pcapng_files))


if __name__ == "__main__":
    main()
