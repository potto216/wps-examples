from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

from wireless_packet_system.analysis import detect_anomalies, summarize_anomalies
from wireless_packet_system.db import DatabaseConfig, bulk_insert_packets, connect, insert_session
from wireless_packet_system.gps import load_gps
from wireless_packet_system.ingestion import normalize_packets


def _load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _db_config(config: Dict[str, Any]) -> DatabaseConfig:
    db = config["database"]
    return DatabaseConfig(
        host=db["host"],
        port=int(db["port"]),
        name=db["name"],
        user=db["user"],
        password=db["password"],
    )


def ingest_command(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    database = _db_config(config)
    protocol = config.get("protocol", "bluetooth")
    capture_device = config.get("capture_device")

    gps_series = load_gps(args.gps) if args.gps else None

    packets = normalize_packets(
        file_path=args.packets,
        session_id=args.session_id,
        protocol=protocol,
        gps_series=gps_series,
    )

    with connect(database) as connection:
        insert_session(
            connection,
            session_id=args.session_id,
            protocol=protocol,
            capture_device=capture_device,
            capture_start=args.capture_start,
            capture_end=args.capture_end,
        )
        inserted = bulk_insert_packets(connection, packets, batch_size=args.batch_size)

    print(f"Inserted {inserted} packets into TimescaleDB.")


def analyze_command(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    database = _db_config(config)
    analysis_cfg = config.get("analysis", {})
    bucket_seconds = int(analysis_cfg.get("interval_bucket_seconds", 60))
    zscore_threshold = float(analysis_cfg.get("anomaly_threshold_zscore", 3.0))

    anomalies = detect_anomalies(
        database=database,
        start=args.start,
        end=args.end,
        bucket_seconds=bucket_seconds,
        zscore_threshold=zscore_threshold,
    )

    print(summarize_anomalies(anomalies))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wireless packet analysis system")
    parser.add_argument("--config", required=True, help="Path to YAML config")

    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Ingest packets and GPS data")
    ingest.add_argument("--packets", required=True, help="Path to Wireshark JSON capture")
    ingest.add_argument("--gps", required=False, help="Path to GPX/KML file")
    ingest.add_argument("--session-id", required=True, help="Session identifier for this capture")
    ingest.add_argument("--capture-start", required=False, help="Capture start time")
    ingest.add_argument("--capture-end", required=False, help="Capture end time")
    ingest.add_argument("--batch-size", type=int, default=1000, help="Batch size for inserts")
    ingest.set_defaults(func=ingest_command)

    analyze = subparsers.add_parser("analyze", help="Run anomaly analysis")
    analyze.add_argument("--start", required=True, help="Start timestamp (ISO 8601)")
    analyze.add_argument("--end", required=True, help="End timestamp (ISO 8601)")
    analyze.set_defaults(func=analyze_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
