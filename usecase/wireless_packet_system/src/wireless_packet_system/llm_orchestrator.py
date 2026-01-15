from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    api_key_env: str


@dataclass(frozen=True)
class AnalysisPlan:
    description: str
    sql_queries: List[str]
    python_steps: List[str]


def build_plan(goal: str) -> AnalysisPlan:
    """
    Create a conservative plan that can be executed by a human or automation.
    This does not run any model; it provides a starting point for analysis.
    """
    description = f"Plan to satisfy: {goal}"
    sql_queries = [
        "SELECT device_address, COUNT(*) AS packet_count "
        "FROM bluetooth_packets GROUP BY device_address ORDER BY packet_count DESC LIMIT 20;",
        "SELECT time_bucket('1 minute', time) AS bucket, COUNT(*) AS packet_count "
        "FROM bluetooth_packets GROUP BY bucket ORDER BY bucket;",
    ]
    python_steps = [
        "Load packet counts into pandas, compute z-scores, and flag anomalies.",
        "Compute per-device mean RSSI and compare against the fleet median.",
    ]
    return AnalysisPlan(description=description, sql_queries=sql_queries, python_steps=python_steps)


def summarize_results(goal: str, findings: Dict[str, str]) -> str:
    lines = [f"Summary for goal: {goal}"]
    for key, value in findings.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)
