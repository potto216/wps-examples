#!/usr/bin/env python3
"""Generate LLM-oriented summary artifacts from a single parquet file."""

from __future__ import annotations

import argparse
import json
import math
from datetime import timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

TIME_COLUMN_CANDIDATES = ("ts", "timestamp", "time", "datetime")
TOP_N = 10
SAMPLE_ROWS = 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build CaptureSummary and TableCatalog artifacts from one parquet file for "
            "LLM-guided statistical analysis workflows."
        )
    )
    parser.add_argument("parquet", help="Path to a single parquet file.")
    parser.add_argument(
        "-o",
        "--output-dir",
        help=(
            "Directory to write summary files. If omitted, no files are written "
            "(stdout summary still prints unless --no-human-readable is set)."
        ),
    )
    parser.add_argument(
        "--no-human-readable",
        action="store_true",
        help="Disable the human-readable summary printed to stdout.",
    )
    parser.add_argument(
        "--table-name",
        help="Optional table name override (defaults to parquet file stem).",
    )
    return parser


def _safe_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return float(value)
    try:
        val = float(value)
    except Exception:
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return val


def _jsonable_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    if isinstance(value, (int, float, str, bool)):
        return value
    return str(value)


def _find_time_column(frame: pd.DataFrame) -> Optional[str]:
    lower_lookup = {column.lower(): column for column in frame.columns}
    for candidate in TIME_COLUMN_CANDIDATES:
        if candidate in frame.columns:
            return candidate
        if candidate in lower_lookup:
            return lower_lookup[candidate]
    return None


def _format_time(value: Any) -> Optional[str]:
    if value is None:
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    formatted_time = ts.to_pydatetime().astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return formatted_time
    

def _numeric_summary(series: pd.Series) -> Dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p05": None,
            "p95": None,
        }
    return {
        "count": int(numeric.shape[0]),
        "min": _safe_number(numeric.min()),
        "max": _safe_number(numeric.max()),
        "mean": _safe_number(numeric.mean()),
        "median": _safe_number(numeric.median()),
        "p05": _safe_number(numeric.quantile(0.05)),
        "p95": _safe_number(numeric.quantile(0.95)),
    }


def _top_values(series: pd.Series, top_n: int = TOP_N) -> List[Dict[str, Any]]:
    value_counts = series.dropna().astype(str).value_counts().head(top_n)
    return [
        {
            "value": value,
            "count": int(count),
        }
        for value, count in value_counts.items()
    ]


def _column_catalog(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    row_count = max(int(frame.shape[0]), 1)
    catalog: List[Dict[str, Any]] = []
    for name in frame.columns:
        series = frame[name]
        non_null = int(series.notna().sum())
        unique = int(series.nunique(dropna=True))
        entry: Dict[str, Any] = {
            "name": name,
            "dtype": str(series.dtype),
            "non_null_count": non_null,
            "null_count": int(frame.shape[0]) - non_null,
            "unique_count": unique,
            "unique_ratio": round(unique / row_count, 6),
        }

        if pd.api.types.is_numeric_dtype(series):
            entry["numeric_summary"] = _numeric_summary(series)
        elif pd.api.types.is_datetime64_any_dtype(series):
            non_na = series.dropna()
            entry["datetime_summary"] = {
                "min": _format_time(non_na.min() if not non_na.empty else None),
                "max": _format_time(non_na.max() if not non_na.empty else None),
            }
        else:
            entry["top_values"] = _top_values(series)
        catalog.append(entry)
    return catalog


def _infer_key_columns(frame: pd.DataFrame) -> List[str]:
    candidates: List[str] = []
    total_rows = int(frame.shape[0])
    for name in frame.columns:
        lname = name.lower()
        if any(token in lname for token in ("addr", "mac", "id", "uuid", "bssid", "ssid")):
            unique = int(frame[name].nunique(dropna=True))
            if total_rows == 0 or 0 < unique <= total_rows:
                candidates.append(name)
    return candidates[:8]


def build_summary_payloads(frame: pd.DataFrame, table_name: str, source_path: str) -> Dict[str, Any]:
    time_col = _find_time_column(frame)
    time_start: Optional[str] = None
    time_end: Optional[str] = None
    if time_col:
        parsed = pd.to_datetime(frame[time_col], utc=True, errors="coerce").dropna()
        if not parsed.empty:
            time_start = _format_time(parsed.min())
            time_end = _format_time(parsed.max())

    capture_summary = {
        "source_file": source_path,
        "table": table_name,
        "row_count": int(frame.shape[0]),
        "column_count": int(frame.shape[1]),
        "time_column": time_col,
        "time_range": {"start": time_start, "end": time_end},
        "numeric_columns": [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])],
        "top_entities": {
            col: _top_values(frame[col], top_n=TOP_N)
            for col in _infer_key_columns(frame)
        },
    }

    table_catalog = {
        "tables": [
            {
                "table": table_name,
                "source_file": source_path,
                "row_count": int(frame.shape[0]),
                "time_range": {"start": time_start, "end": time_end},
                "key_columns": _infer_key_columns(frame),
                "columns": _column_catalog(frame),
                "sample_rows": [
                    {col: _jsonable_value(val) for col, val in row.items()}
                    for row in frame.head(SAMPLE_ROWS).to_dict(orient="records")
                ],
            }
        ]
    }
    return {
        "capture_summary": capture_summary,
        "table_catalog": table_catalog,
    }


def _human_readable(summary: Dict[str, Any], catalog: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("LLM Summary Report")
    lines.append("==================")
    lines.append(f"Source file: {summary['source_file']}")
    lines.append(f"Table: {summary['table']}")
    lines.append(f"Rows: {summary['row_count']}")
    lines.append(f"Columns: {summary['column_count']}")

    tr = summary["time_range"]
    if tr["start"] and tr["end"]:
        lines.append(f"Time range: {tr['start']} -> {tr['end']}")

    lines.append("")
    lines.append("Top entity values:")
    if not summary["top_entities"]:
        lines.append("  (none inferred)")
    else:
        for column, values in summary["top_entities"].items():
            lines.append(f"  - {column}:")
            for item in values[:5]:
                lines.append(f"      {item['value']}: {item['count']}")

    lines.append("")
    lines.append("Column overview:")
    for column in catalog["tables"][0]["columns"]:
        lines.append(
            f"  - {column['name']} ({column['dtype']}), non-null={column['non_null_count']}, unique={column['unique_count']}"
        )
    return "\n".join(lines)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    args = build_parser().parse_args()

    parquet_path = Path(args.parquet)
    if not parquet_path.exists() or not parquet_path.is_file():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    frame = pd.read_parquet(parquet_path)
    table_name = args.table_name or parquet_path.stem

    payloads = build_summary_payloads(frame, table_name=table_name, source_path=str(parquet_path))

    if args.output_dir:
        output_dir = Path(args.output_dir)
        _write_json(output_dir / "capture_summary.json", payloads["capture_summary"])
        _write_json(output_dir / "table_catalog.json", payloads["table_catalog"])

    if not args.no_human_readable:
        print(_human_readable(payloads["capture_summary"], payloads["table_catalog"]))


if __name__ == "__main__":
    main()
