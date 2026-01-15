from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from wireless_packet_system.db import DatabaseConfig, connect, fetch_packet_counts


@dataclass(frozen=True)
class AnomalyResult:
    bucket: str
    packet_count: int
    zscore: float


def detect_anomalies(database: DatabaseConfig, start: str, end: str, bucket_seconds: int, zscore_threshold: float) -> List[AnomalyResult]:
    with connect(database) as connection:
        rows = fetch_packet_counts(connection, start_time=start, end_time=end, bucket_seconds=bucket_seconds)

    if not rows:
        return []

    frame = pd.DataFrame(rows)
    frame["packet_count"] = frame["packet_count"].astype(float)
    mean = frame["packet_count"].mean()
    std = frame["packet_count"].std(ddof=0) or 1.0
    frame["zscore"] = (frame["packet_count"] - mean) / std

    anomalies = frame[frame["zscore"].abs() >= zscore_threshold]
    return [
        AnomalyResult(bucket=str(row["bucket"]), packet_count=int(row["packet_count"]), zscore=float(row["zscore"]))
        for _, row in anomalies.iterrows()
    ]


def summarize_anomalies(anomalies: List[AnomalyResult]) -> str:
    if not anomalies:
        return "No anomalies detected for the selected interval."

    summary_lines = ["Detected anomalies in packet counts:"]
    for anomaly in anomalies:
        summary_lines.append(
            f"- Bucket {anomaly.bucket}: count={anomaly.packet_count} (z={anomaly.zscore:.2f})"
        )
    return "\n".join(summary_lines)
