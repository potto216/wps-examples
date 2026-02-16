from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency for local demo
    OpenAI = None

LOG_LEVELS = {"debug", "info", "warning", "error", "critical"}
LOG_TO_CHOICES = {"stdout", "file"}
logger = logging.getLogger("llm_guided_stat_analyst_demo")


def configure_logging(log_level: str, *, log_to: str, log_file: Optional[str]) -> None:
    level = str(log_level).strip().lower()
    if level not in LOG_LEVELS:
        raise ValueError(f"Invalid --log-level '{log_level}'. Expected one of {sorted(LOG_LEVELS)}")
    dest = str(log_to).strip().lower()
    if dest not in LOG_TO_CHOICES:
        raise ValueError(f"Invalid --log-to '{log_to}'. Expected one of {sorted(LOG_TO_CHOICES)}")
    handlers: list[logging.Handler] = []
    if dest == "stdout":
        handlers.append(logging.StreamHandler(sys.stdout))
    else:
        file_name = (log_file or "llm_guided_stat_analyst_demo.log").strip()
        log_path = Path(file_name)
        if not log_path.is_absolute():
            log_path = Path.cwd() / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=handlers,
    )
    logger.info("Logging configured level=%s destination=%s", level, dest)
    if dest == "file":
        logger.info("Logging to file=%s", str(log_path))


def _openai_response_to_jsonable(response: Any) -> Any:
    if response is None:
        return None
    # openai-python v1 objects are pydantic-like and typically expose model_dump()
    for attr in ("model_dump", "to_dict", "dict", "to_dict_recursive"):
        fn = getattr(response, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    # Best-effort fallback: try __dict__ then string
    try:
        return dict(getattr(response, "__dict__", {}))
    except Exception:
        return str(response)


def _json_dumps_safe(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, sort_keys=True, default=str)
    except Exception:
        return str(obj)


def _redact_secrets(obj: Any) -> Any:
    """Best-effort redaction for logs (avoid leaking API keys)."""

    sensitive = {"openai_api_key", "api_key", "openai_key", "openaiapikey", "openai_api_token"}

    if isinstance(obj, dict):
        redacted: Dict[Any, Any] = {}
        for key, value in obj.items():
            key_str = str(key)
            if key_str in {"OPENAI_API_KEY"} or key_str.lower().replace("-", "_") in sensitive:
                redacted[key] = "***REDACTED***"
            else:
                redacted[key] = _redact_secrets(value)
        return redacted
    if isinstance(obj, list):
        return [_redact_secrets(v) for v in obj]
    return obj


@dataclass
class AnalysisRequest:
    question: str
    planner: str
    execution_mode: str
    model: str
    dataset: Optional[str] = None


class DemoAnalyst:
    """Simple end-to-end analyst using plan execution or safe Python snippets."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        metadata_path = data_dir / "metadata.json"
        capture_summary_path = data_dir / "capture_summary.json"
        table_catalog_path = data_dir / "table_catalog.json"

        logger.info("Initializing analyst data_dir=%s", str(data_dir))
        parquet_files = sorted(data_dir.glob("*.parquet"))
        if not parquet_files:
            raise ValueError(
                f"No parquet data files found in data directory '{data_dir}'. "
                "Provide at least one valid parquet capture dataset."
            )

        logger.info("Loading %d parquet file(s) from data_dir", len(parquet_files))
        self.tables = {p.stem: pd.read_parquet(p) for p in parquet_files}

        for table in self.tables.values():
            if "timestamp" in table.columns:
                table["timestamp"] = pd.to_datetime(table["timestamp"], utc=True, errors="coerce")

        try:
            table_counts = {name: int(len(frame)) for name, frame in self.tables.items()}
            logger.info("Tables loaded row counts=%s", _json_dumps_safe(table_counts))
        except Exception:
            logger.debug("Could not log table sizes", exc_info=True)

        if metadata_path.exists():
            logger.info("Loading metadata=%s", str(metadata_path))
            with open(metadata_path, "r", encoding="utf-8") as handle:
                self.metadata = json.load(handle)
        else:
            logger.info("Metadata missing; inferring metadata from loaded tables")

            known_devices: List[str] = []
            # Best-effort: infer "known devices" from common identifier columns.
            for table in self.tables.values():
                for col in ("advertiser_addr", "transmitter_mac", "receiver_mac", "src", "dst"):
                    if col in table.columns:
                        try:
                            vals = (
                                table[col]
                                .dropna()
                                .astype(str)
                                .loc[lambda s: s.str.len() > 0]
                                .unique()
                                .tolist()
                            )
                            known_devices.extend(vals)
                        except Exception:
                            logger.debug("Could not infer devices from column=%s", col, exc_info=True)
            known_devices = sorted(set(known_devices))[:50]

            self.metadata = {
                "environment": "inferred_from_loaded_tables",
                "known_devices_present": known_devices,
            }

        self.external_context = self._load_external_context(capture_summary_path, table_catalog_path)
        # Always provide a minimal schema view of whatever tables we actually loaded so the LLM
        # doesn't guess columns like "highest_layer" against the wrong dataset.
        self.external_context.setdefault("loaded_tables", self._build_loaded_tables_context())

    def _build_loaded_tables_context(self) -> Dict[str, Any]:
        context: Dict[str, Any] = {}
        for name, frame in self.tables.items():
            columns = [
                {"name": str(col), "dtype": str(frame[col].dtype)}
                for col in list(frame.columns)[:50]
            ]
            time_range: Dict[str, Any] = {}
            if "timestamp" in frame.columns:
                parsed = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dropna()
                if not parsed.empty:
                    time_range = {
                        "start": parsed.min().isoformat().replace("+00:00", "Z"),
                        "end": parsed.max().isoformat().replace("+00:00", "Z"),
                    }
            context[name] = {
                "row_count": int(len(frame)),
                "column_count": int(frame.shape[1]),
                "columns": columns,
                "time_range": time_range,
            }
        return context

    def validate_plan(self, plan: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        dataset = plan.get("dataset")
        if not isinstance(dataset, str) or dataset not in self.tables:
            errors.append(
                f"Unknown dataset '{dataset}'. Available datasets: {sorted(self.tables.keys())}"
            )
            return errors

        frame = self.tables[dataset]
        columns = set(str(c) for c in frame.columns)

        for idx, filt in enumerate(plan.get("filters", []) or []):
            if not isinstance(filt, dict):
                errors.append(f"Filter #{idx} is not an object")
                continue
            col = filt.get("col")
            op = filt.get("op")
            if not isinstance(col, str) or col not in columns:
                errors.append(f"Filter #{idx} references unknown column '{col}'")
            if op not in {"==", "in", "contains", "!=", "between"}:
                errors.append(f"Filter #{idx} has unsupported op '{op}'")
                continue

            if op == "between":
                value = filt.get("value")
                if not isinstance(value, list) or len(value) != 2:
                    errors.append(
                        f"Filter #{idx} op 'between' requires value as a 2-item array like [lower, upper]"
                    )
                else:
                    if any(v is None or (isinstance(v, str) and not v.strip()) for v in value):
                        errors.append(
                            f"Filter #{idx} op 'between' has empty bounds; provide both lower and upper"
                        )

        for idx, col in enumerate(plan.get("groupby", []) or []):
            if not isinstance(col, str) or col not in columns:
                errors.append(f"Groupby column #{idx} unknown '{col}'")

        plot_cfg = plan.get("plot") or {}
        if isinstance(plot_cfg, dict):
            plot_type = plot_cfg.get("type")
            if plot_type == "timeseries" and "timestamp" not in columns:
                errors.append("Timeseries plot requested but dataset has no 'timestamp' column")

        # Enforce metrics rules when grouping (prevents KeyError: 'count' later).
        groupby_cols = plan.get("groupby") or []
        metrics = plan.get("metrics") or []

        if groupby_cols:
            if not isinstance(metrics, list) or len(metrics) == 0:
                errors.append(
                    "Plan uses 'groupby' but 'metrics' is empty. "
                    "Include at least: metrics: [{\"name\": \"count\"}]"
                )
            else:
                metric_names: List[str] = []
                for m in metrics:
                    if not isinstance(m, dict):
                        errors.append("Each entry in 'metrics' must be an object like {\"name\": \"count\"}.")
                        continue
                    name = str(m.get("name", "")).strip().lower()
                    if not name:
                        errors.append("Metric is missing required field 'name'.")
                        continue
                    metric_names.append(name)

                # This demo executor currently only supports 'count' for grouped results.
                supported_metrics = {"count"}
                unsupported = sorted({n for n in metric_names if n and n not in supported_metrics})
                if unsupported:
                    errors.append(
                        f"Unsupported metric(s) for grouped analysis: {unsupported}. "
                        f"Supported metrics: {sorted(supported_metrics)}"
                    )

                if "count" not in metric_names:
                    errors.append(
                        "Plan uses 'groupby' but does not include metric 'count'. "
                        "Add: metrics: [{\"name\": \"count\"}]"
                    )

        return errors

    @staticmethod
    def _load_external_context(capture_summary_path: Path, table_catalog_path: Path) -> Dict[str, Any]:
        context: Dict[str, Any] = {}

        if capture_summary_path.exists():
            with open(capture_summary_path, "r", encoding="utf-8") as handle:
                capture_summary = json.load(handle)
            context["capture_summary"] = {
                "source_file": capture_summary.get("source_file"),
                "table": capture_summary.get("table"),
                "row_count": capture_summary.get("row_count"),
                "column_count": capture_summary.get("column_count"),
                "numeric_columns": capture_summary.get("numeric_columns", []),
                "time_column": capture_summary.get("time_column"),
                "time_range": capture_summary.get("time_range", {}),
            }

        if table_catalog_path.exists():
            with open(table_catalog_path, "r", encoding="utf-8") as handle:
                table_catalog = json.load(handle)
            table_entries = table_catalog.get("tables", [])
            if table_entries:
                first_table = table_entries[0]
                top_columns: List[Dict[str, Any]] = []
                for col in first_table.get("columns", [])[:8]:
                    top_columns.append(
                        {
                            "name": col.get("name"),
                            "dtype": col.get("dtype"),
                            "non_null_count": col.get("non_null_count"),
                            "null_count": col.get("null_count"),
                            "numeric_summary": col.get("numeric_summary"),
                            "datetime_summary": col.get("datetime_summary"),
                            "top_values": col.get("top_values", [])[:5],
                        }
                    )
                context["table_catalog"] = {
                    "table": first_table.get("table"),
                    "source_file": first_table.get("source_file"),
                    "row_count": first_table.get("row_count"),
                    "time_range": first_table.get("time_range", {}),
                    "columns": top_columns,
                    "sample_rows": first_table.get("sample_rows", [])[:3],
                }

        return context

    def generate_plan(self, request: AnalysisRequest) -> Dict[str, Any]:
        logger.info("Generating plan planner=%s model=%s", request.planner, request.model)

        if request.dataset:
            if request.dataset not in self.tables:
                logger.warning(
                    "Requested dataset override not found dataset=%s available=%s",
                    request.dataset,
                    sorted(self.tables.keys()),
                )
            else:
                logger.info("Using dataset override=%s", request.dataset)

        if request.planner == "llm":
            plan = self._llm_plan(request, validation_errors=None)
            if plan:
                errors = self.validate_plan(plan)
                if not errors:
                    logger.info("LLM planner returned a valid plan")
                    logger.debug("LLM plan JSON=%s", _json_dumps_safe(plan))
                    return plan

                logger.warning("LLM plan invalid; retrying once with structured errors")
                logger.info("Plan validation errors: %s", "; ".join(errors))
                repaired = self._llm_plan(request, validation_errors=errors)
                if repaired:
                    repaired_errors = self.validate_plan(repaired)
                    if not repaired_errors:
                        logger.info("LLM planner repaired plan successfully")
                        logger.debug("LLM repaired plan JSON=%s", _json_dumps_safe(repaired))
                        return repaired
                    logger.warning("LLM repaired plan still invalid: %s", "; ".join(repaired_errors))

            logger.warning("LLM planner unavailable/failed/invalid; falling back to heuristic planner")
        return self._heuristic_plan(request.question, request.dataset)

    def _llm_plan(self, request: AnalysisRequest, *, validation_errors: Optional[List[str]]) -> Optional[Dict[str, Any]]:
        api_key_set = bool(os.getenv("OPENAI_API_KEY"))
        if OpenAI is None or not api_key_set:
            logger.info(
                "LLM planner not available openai_imported=%s OPENAI_API_KEY_set=%s",
                OpenAI is not None,
                api_key_set,
            )
            return None

        available_datasets = sorted(self.tables.keys())
        allowed_datasets = available_datasets
        if request.dataset and request.dataset in self.tables:
            allowed_datasets = [request.dataset]

        prompt = (
            "Return JSON only for an analysis plan with keys: dataset, filters, groupby, metrics, plot, anomaly. "
            "You MUST use only datasets and column names that exist in the provided table schema. "
            "Filter operators must be one of: ==, !=, in, contains, between. "
            "For op 'between', set value to a 2-item array like [lower, upper] (numbers or timestamps). "
            f"Datasets available: {available_datasets}. "
            f"Question: {request.question}"
        )

        if request.dataset:
            prompt += f"\nYou MUST use dataset: {request.dataset}"

        if self.external_context:
            prompt += (
                "\n\nAdditional capture context from preprocessing summary files "
                "(use when useful for filters/metrics/time windows):\n"
                f"{json.dumps(self.external_context, indent=2, default=str)}"
            )

        if validation_errors:
            prompt += (
                "\n\nYour previous plan failed validation with these errors. "
                "Return a corrected plan that fixes them:\n"
                + "\n".join(f"- {e}" for e in validation_errors)
            )

        # Prefer `developer` message for instructions (replaces `system` for newer models).
        messages = [
            {"role": "developer", "content": "You are a wireless packet analytics planner."},
            {"role": "user", "content": prompt},
        ]

        # Strict schema for the plan you expect.
        # Note: enums are set from the actual loaded tables.
        plan_schema = {
            "name": "analysis_plan",
            "description": "Planner output for wireless packet analytics.",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["dataset", "filters", "groupby", "metrics", "plot", "anomaly"],
                "properties": {
                    "dataset": {"type": "string", "enum": allowed_datasets},

                    "filters": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["col", "op", "value"],
                            "properties": {
                                "col": {"type": "string"},
                                "op": {
                                    "type": "string",
                                    "enum": ["==", "!=", "in", "contains", "between"],
                                },
                                "value": {
                                    "anyOf": [
                                        {"type": "string"},
                                        {"type": "number"},
                                        {"type": "boolean"},
                                        {
                                            "type": "array",
                                            "items": {"anyOf": [{"type": "string"}, {"type": "number"}]},
                                        },
                                    ]
                                },
                            },
                        },
                    },

                    "groupby": {"type": "array", "items": {"type": "string"}},

                    "metrics": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,  # <-- MUST be false
                            "required": ["name"],
                            "properties": {
                                "name": {"type": "string"},
                            },
                        },
                    },

                    "plot": {
                        "type": "object",
                        "additionalProperties": False,  # <-- MUST be false
                        "required": ["type", "x", "y"],
                        "properties": {
                            "type": {"type": "string"},
                            "x": {"type": "string"},
                            "y": {"type": "string"},
                        },
                    },

                    "anomaly": {
                        "type": "object",
                        "additionalProperties": False,  # <-- MUST be false
                        "required": ["method", "on", "threshold"],
                        "properties": {
                            "method": {"type": "string"},
                            "on": {"type": "string"},
                            "threshold": {"type": "number"},
                        },
                    },
                },
            },
        }


        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Build kwargs without temperature (gpt-5-nano rejects non-default temperature).
        kwargs: Dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "response_format": {"type": "json_schema", "json_schema": plan_schema},
            # Optional: reduce reasoning cost/speed; models before gpt-5.1 default to medium. :contentReference[oaicite:3]{index=3}
            "reasoning_effort": "minimal",
        }

        logger.info("OpenAI chat.completions.create model=%s", request.model)
        logger.debug("OpenAI request payload=%s", _json_dumps_safe(kwargs))

        try:
            response = client.chat.completions.create(**kwargs)

        except Exception as e:
            # If you later decide to add temperature for some models, keep this retry logic:
            # retry when the API says temperature is unsupported for this model.
            try:
                body = getattr(e, "body", None) or {}
                err = (body.get("error") or {}) if isinstance(body, dict) else {}
                if err.get("param") == "temperature" and err.get("code") == "unsupported_value":
                    logger.info("Model rejected temperature; retrying with default temperature.")
                    kwargs.pop("temperature", None)
                    response = client.chat.completions.create(**kwargs)
                else:
                    raise
            except Exception:
                logger.exception("OpenAI request failed")
                return None

        logger.debug("OpenAI raw response=%s", _json_dumps_safe(_openai_response_to_jsonable(response)))

        try:
            content = response.choices[0].message.content
        except Exception:
            logger.exception("Unexpected OpenAI response shape")
            return None

        logger.debug("OpenAI parsed content=%s", content)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning("LLM plan JSON parse failed; content=%s", content)
            return None

    def _find_dataset_with_columns(self, columns: List[str], preferred_dataset: Optional[str] = None) -> Optional[str]:
        if preferred_dataset and preferred_dataset in self.tables:
            if set(columns) <= set(self.tables[preferred_dataset].columns):
                return preferred_dataset

        for name, frame in self.tables.items():
            if set(columns) <= set(frame.columns):
                return name
        return None

    def _heuristic_plan(self, question: str, preferred_dataset: Optional[str] = None) -> Dict[str, Any]:
        logger.info("Using heuristic planner")
        q = question.lower()
        fallback_dataset = preferred_dataset if preferred_dataset in self.tables else next(iter(self.tables.keys()))

        if "bluetooth" in q and ("advertis" in q or "advertising" in q) and ("most" in q and "common" in q):
            packet_dataset = self._find_dataset_with_columns(
                ["highest_layer", "layers", "timestamp"],
                preferred_dataset,
            )
            if packet_dataset:
                return {
                    "dataset": packet_dataset,
                    "filters": [{"col": "layers", "op": "contains", "value": "BTLE_ADV"}],
                    "groupby": ["highest_layer"],
                    "metrics": [{"name": "count"}],
                    "plot": {"type": "bar", "x": "highest_layer", "y": "count"},
                    "anomaly": {"method": "robust_z", "on": "count", "threshold": 3.5},
                }

        if "advertise" in q:
            adv_dataset = self._find_dataset_with_columns(["advertiser_addr"], preferred_dataset)
            if adv_dataset:
                device = ""
                for token in question.replace("?", "").split():
                    if token.count(":") == 5:
                        device = token.lower()
                filters = [{"col": "advertiser_addr", "op": "==", "value": device}] if device else []
                return {
                    "dataset": adv_dataset,
                    "filters": filters,
                    "groupby": ["advertiser_addr"],
                    "metrics": [{"name": "count"}],
                    "plot": {"type": "bar", "x": "advertiser_addr", "y": "count"},
                    "anomaly": {"method": "robust_z", "on": "count", "threshold": 3.5},
                }

        if "probe" in q and ("ap" in q or "bssid" in q):
            probe_dataset = self._find_dataset_with_columns(["frame_subtype", "receiver_mac"], preferred_dataset)
            if probe_dataset:
                return {
                    "dataset": probe_dataset,
                    "filters": [{"col": "frame_subtype", "op": "==", "value": "probe_req"}],
                    "groupby": ["receiver_mac"],
                    "metrics": [{"name": "count"}],
                    "plot": {"type": "bar", "x": "receiver_mac", "y": "count"},
                    "anomaly": {"method": "robust_z", "on": "count", "threshold": 3.5},
                }

        subtype_dataset = self._find_dataset_with_columns(["frame_subtype"], preferred_dataset)
        if subtype_dataset:
            return {
                "dataset": subtype_dataset,
                "filters": [
                    {"col": "frame_subtype", "op": "in", "value": ["probe_req", "assoc_req"]}
                ],
                "groupby": ["frame_subtype"],
                "metrics": [{"name": "count"}],
                "plot": {"type": "bar", "x": "frame_subtype", "y": "count"},
                "anomaly": {"method": "rolling_shift", "on": "count", "threshold": 2.5},
            }

        frame = self.tables[fallback_dataset]
        group_col = "timestamp" if "timestamp" in frame.columns else str(frame.columns[0])
        return {
            "dataset": fallback_dataset,
            "filters": [],
            "groupby": [group_col],
            "metrics": [{"name": "count"}],
            "plot": {"type": "bar", "x": group_col, "y": "count"},
            "anomaly": {"method": "robust_z", "on": "count", "threshold": 3.5},
        }

    def execute_plan(self, plan: Dict[str, Any], question: str, plot_out: Optional[Path]) -> Dict[str, Any]:
        dataset = plan["dataset"]
        logger.info("Executing plan dataset=%s plot_out=%s", dataset, str(plot_out) if plot_out else None)
        logger.debug("Plan details=%s", _json_dumps_safe(plan))
        frame = self.tables[dataset].copy()

        filters = plan.get("filters", [])
        logger.info("Applying %d filter(s)", len(filters))
        for f in filters:
            col = f.get("col")
            op = f.get("op")
            value = f.get("value")
            if col not in frame.columns:
                raise ValueError(
                    f"Plan references unknown column '{col}' for dataset '{dataset}'. "
                    f"Available columns: {sorted(frame.columns.astype(str).tolist())}"
                )

            if op == "==":
                frame = frame[frame[col] == value]
            elif op == "in":
                values = value
                if isinstance(values, str) or not isinstance(values, list):
                    values = [values]
                frame = frame[frame[col].isin(values)]
            elif op == "!=":
                frame = frame[frame[col] != value]
            elif op == "contains":
                needle = "" if value is None else str(value)
                frame = frame[frame[col].astype(str).str.contains(needle, na=False)]
            elif op == "between":
                if not isinstance(value, list) or len(value) != 2:
                    raise ValueError(
                        f"Filter op 'between' requires a 2-item array value for col '{col}', got {value!r}"
                    )
                lower, upper = value

                series = frame[col]
                is_datetime = pd.api.types.is_datetime64_any_dtype(series)

                if is_datetime:
                    lo_dt = pd.to_datetime(lower, utc=True, errors="coerce")
                    hi_dt = pd.to_datetime(upper, utc=True, errors="coerce")
                    if pd.isna(lo_dt) or pd.isna(hi_dt):
                        raise ValueError(
                            f"Could not parse datetime bounds for 'between' filter on '{col}': {value!r}"
                        )
                    frame = frame[(frame[col] >= lo_dt) & (frame[col] <= hi_dt)]
                else:
                    try:
                        lo_num = float(lower)
                        hi_num = float(upper)
                    except Exception as e:
                        raise ValueError(
                            f"Could not parse numeric bounds for 'between' filter on '{col}': {value!r}"
                        ) from e
                    frame = frame[(frame[col].astype(float) >= lo_num) & (frame[col].astype(float) <= hi_num)]

        facts: Dict[str, Any] = {
            "dataset": dataset,
            "rows_after_filter": int(len(frame)),
            "question": question,
        }
        logger.info("Rows after filter=%d", int(len(frame)))

        result_table = frame
        groupby_cols = plan.get("groupby") or []
        if groupby_cols:
            logger.info("Computing grouped metrics groupby=%s metrics=%s", groupby_cols, plan.get("metrics", []))
            grouped = frame.groupby(groupby_cols, dropna=False)
            agg = pd.DataFrame(index=grouped.size().index)
            for metric in plan.get("metrics", []):
                if metric["name"] == "count":
                    agg["count"] = grouped.size()
            result_table = agg.reset_index().sort_values("count", ascending=False)
            facts["top_group"] = result_table.iloc[0].to_dict() if not result_table.empty else {}
            if facts.get("top_group"):
                logger.info("Top group=%s", _json_dumps_safe(facts["top_group"]))

        # Additional computed feature for timing-style questions.
        if ("timing" in question.lower() or "distribution" in question.lower()) and "timestamp" in frame.columns:
            logger.info("Computing timing features inter_arrival_sec")
            timed = frame.sort_values("timestamp").copy()
            timed["inter_arrival_sec"] = timed["timestamp"].diff().dt.total_seconds()
            result_table = timed.dropna(subset=["inter_arrival_sec"])
            facts["inter_arrival_mean"] = float(result_table["inter_arrival_sec"].mean()) if not result_table.empty else None
            facts["inter_arrival_p95"] = float(result_table["inter_arrival_sec"].quantile(0.95)) if not result_table.empty else None

        anomaly_report = self._anomaly_detect(plan.get("anomaly", {}), result_table)
        logger.info("Anomalies detected=%d", len(anomaly_report))
        logger.debug("Anomaly details=%s", _json_dumps_safe(anomaly_report))

        if plot_out:
            logger.info("Writing plot=%s", str(plot_out))
            self._plot(plan.get("plot", {}), result_table, plot_out)
            facts["plot_path"] = str(plot_out)

        return {
            "table": result_table,
            "facts": facts,
            "anomalies": anomaly_report,
        }

    def execute_python(self, question: str) -> Dict[str, Any]:
        """Safe, tiny code execution mode with prebuilt snippets (extendable)."""
        logger.info("Executing python snippet mode")
        q = question.lower()

        if "advertise" in q:
            dataset = self._find_dataset_with_columns(["advertiser_addr"]) or next(iter(self.tables.keys()))
            frame = self.tables[dataset]
            if "advertiser_addr" in frame.columns:
                result = (
                    frame.groupby("advertiser_addr")
                    .size()
                    .reset_index(name="count")
                    .sort_values("count", ascending=False)
                )
            else:
                result = pd.DataFrame({"count": [int(len(frame))]})
        else:
            dataset = (
                self._find_dataset_with_columns(["frame_subtype", "receiver_mac"])
                or self._find_dataset_with_columns(["frame_subtype"])
                or next(iter(self.tables.keys()))
            )
            frame = self.tables[dataset]
            if {"frame_subtype", "receiver_mac"} <= set(frame.columns):
                probe = frame[frame["frame_subtype"] == "probe_req"].copy()
                result = (
                    probe.groupby("receiver_mac")
                    .size()
                    .reset_index(name="count")
                    .sort_values("count", ascending=False)
                )
            elif "frame_subtype" in frame.columns:
                result = (
                    frame.groupby("frame_subtype")
                    .size()
                    .reset_index(name="count")
                    .sort_values("count", ascending=False)
                )
            else:
                result = pd.DataFrame({"count": [int(len(frame))]})

        return {
            "table": result,
            "facts": {
                "mode": "python",
                "dataset": dataset,
                "rows": int(len(result)),
                "top_row": result.iloc[0].to_dict() if not result.empty else {},
            },
            "anomalies": [],
        }

    @staticmethod
    def _anomaly_detect(anomaly_cfg: Dict[str, Any], table: pd.DataFrame) -> List[Dict[str, Any]]:
        if table.empty or "count" not in table.columns:
            return []

        values = table["count"].astype(float)
        method = anomaly_cfg.get("method", "robust_z")
        if method == "robust_z":
            median = values.median()
            mad = np.median(np.abs(values - median)) or 1.0
            score = 0.6745 * (values - median) / mad
            threshold = float(anomaly_cfg.get("threshold", 3.5))
            flags = score.abs() > threshold
            flagged = table.loc[flags].copy()
            flagged["score"] = score[flags]
            return flagged.to_dict("records")

        # rolling shift heuristic
        window = int(anomaly_cfg.get("window", 5))
        threshold = float(anomaly_cfg.get("threshold", 2.5))
        rolling = values.rolling(window, min_periods=max(2, window // 2)).median()
        shift = (values - rolling).abs() / (rolling.replace(0, np.nan))
        flags = shift > threshold
        flagged = table.loc[flags].copy()
        flagged["score"] = shift[flags].fillna(0.0)
        return flagged.to_dict("records")

    @staticmethod
    def _plot(plot_cfg: Dict[str, Any], table: pd.DataFrame, out_path: Path) -> None:
        logger.info("Plotting type=%s out_path=%s", plot_cfg.get("type"), str(out_path))
        plt.figure(figsize=(8, 4))
        plot_type = plot_cfg.get("type")
        if plot_type == "bar" and {plot_cfg.get("x"), plot_cfg.get("y")} <= set(table.columns):
            plt.bar(table[plot_cfg["x"]].astype(str), table[plot_cfg["y"]])
            plt.xticks(rotation=45, ha="right")
        elif plot_type == "hist" and plot_cfg.get("x") in table.columns:
            plt.hist(table[plot_cfg["x"]].dropna(), bins=int(plot_cfg.get("bins", 20)))
        elif plot_type == "timeseries" and "timestamp" in table.columns:
            counts = table.set_index("timestamp").resample("30S").size()
            plt.plot(counts.index, counts.values)
            plt.xticks(rotation=30)
        else:
            table.head(20).plot(kind="bar")
        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path)
        plt.close()


def build_narrative(facts: Dict[str, Any], anomalies: List[Dict[str, Any]]) -> str:
    base = [
        f"Dataset `{facts.get('dataset', 'n/a')}` returned {facts.get('rows_after_filter', facts.get('rows', 0))} rows after filtering.",
    ]
    top = facts.get("top_group") or facts.get("top_row")
    if top:
        base.append(f"Top entity observed: {top}.")
    if facts.get("inter_arrival_mean") is not None:
        base.append(
            f"Inter-arrival mean={facts['inter_arrival_mean']:.3f}s p95={facts['inter_arrival_p95']:.3f}s."
        )
    if anomalies:
        base.append(f"Detected {len(anomalies)} anomaly candidate(s).")
    else:
        base.append("No anomalies detected with current threshold.")
    return " ".join(base)


def build_incident_report(question: str, facts: Dict[str, Any], anomalies: List[Dict[str, Any]]) -> str:
    if not anomalies:
        return "No incident report generated (no anomalies)."
    return (
        "## Incident Report\n"
        f"- Query: {question}\n"
        f"- What: {len(anomalies)} anomalous aggregate rows detected\n"
        f"- Who/Where: {anomalies[0]}\n"
        "- Confidence: medium (simple robust thresholding)\n"
        f"- Supporting context: top group {facts.get('top_group', facts.get('top_row', {}))}"
    )


def build_llm_report(
    *,
    question: str,
    plan: Dict[str, Any],
    facts: Dict[str, Any],
    anomalies: List[Dict[str, Any]],
    model: str,
) -> Optional[str]:
    api_key_set = bool(os.getenv("OPENAI_API_KEY"))
    if OpenAI is None or not api_key_set:
        logger.info(
            "Skipping LLM report generation openai_imported=%s OPENAI_API_KEY_set=%s",
            OpenAI is not None,
            api_key_set,
        )
        return None

    logger.info("Sending execution results to LLM for report model=%s", model)
    payload = {
        "question": question,
        "plan": plan,
        "facts": facts,
        "anomalies": anomalies,
    }
    messages = [
        {
            "role": "developer",
            "content": (
                "You are a wireless analytics assistant. "
                "Write a concise report with sections: Summary, Key Findings, and Anomaly Assessment."
            ),
        },
        {
            "role": "user",
            "content": f"Create a human-readable report from this analysis output:\n{json.dumps(payload, indent=2, default=str)}",
        },
    ]

    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            reasoning_effort="minimal",
        )
        logger.debug("LLM report raw response=%s", _json_dumps_safe(_openai_response_to_jsonable(response)))
        content = response.choices[0].message.content
        logger.info("Received human-readable report from LLM")
        return str(content).strip()
    except Exception:
        logger.exception("LLM report generation failed; falling back to local summary")
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-guided statistical analyst demo for BLE/Wi-Fi tables")
    parser.add_argument(
        "--config-dir",
        default=".",
        help="Directory containing JSON config file (default: current directory)",
    )
    parser.add_argument(
        "--config-file",
        default="llm_guided_stat_analyst_demo.json",
        help="JSON configuration filename (default: llm_guided_stat_analyst_demo.json)",
    )
    parser.add_argument("--data-dir", default=argparse.SUPPRESS, help="Directory with Parquet tables + metadata")
    parser.add_argument("--question", default=argparse.SUPPRESS, help="Question in natural language")
    parser.add_argument(
        "--dataset",
        default=argparse.SUPPRESS,
        help="Force dataset name to analyze (overrides planner dataset selection)",
    )
    parser.add_argument("--planner", choices=["heuristic", "llm"], default=argparse.SUPPRESS)
    parser.add_argument("--execution-mode", choices=["plan", "python"], default=argparse.SUPPRESS)
    parser.add_argument("--model", default=argparse.SUPPRESS)
    parser.add_argument(
        "--skip-plot-generation",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Skip writing plot files even if the plan includes a plot config",
    )
    parser.add_argument(
        "--output-dir",
        default=argparse.SUPPRESS,
        help="Directory to write plots and report artifacts",
    )
    parser.add_argument("--log-level", default=argparse.SUPPRESS, help="Logging level: debug, info, warning, error, critical")
    parser.add_argument(
        "--log-to",
        choices=sorted(LOG_TO_CHOICES),
        default=argparse.SUPPRESS,
        help="Where to write logs: stdout or file (default: stdout)",
    )
    parser.add_argument(
        "--log-file",
        nargs="?",
        const="llm_guided_stat_analyst_demo.log",
        default=argparse.SUPPRESS,
        help="Log file path (used when --log-to file). If provided without a value, defaults to ./llm_guided_stat_analyst_demo.log",
    )

    args = parser.parse_args()
    config_path = Path(args.config_dir) / args.config_file
    defaults: Dict[str, Any] = {
        "data_dir": "usecase/llm_rag_ml/data",
        "planner": "heuristic",
        "execution_mode": "plan",
        "model": "gpt-5-nano",
        "dataset": None,
        "openai_api_key": None,
        "output_dir": "usecase/llm_rag_ml/artifacts",
        "skip_plot_generation": False,
        "log_level": "info",
        "log_to": "stdout",
        "log_file": None,
    }

    config_values: Dict[str, Any] = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            raise ValueError(f"Config file must contain a JSON object: {config_path}")
        config_values = loaded

    merged = defaults.copy()
    merged.update({k: v for k, v in config_values.items() if k in defaults or k == "question"})
    cli_overrides = vars(args).copy()
    cli_overrides.pop("config_dir", None)
    cli_overrides.pop("config_file", None)
    merged.update(cli_overrides)
    merged["config_dir"] = args.config_dir
    merged["config_file"] = args.config_file
    merged["config_path"] = str(config_path)

    if not merged.get("question"):
        parser.error("A question must be provided via --question or in the JSON config file under key 'question'.")

    return argparse.Namespace(**merged)


def write_reports(
    output_dir: Path,
    *,
    plan: Dict[str, Any],
    output: Dict[str, Any],
    question: str,
    llm_model: str,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Preparing report artifacts")
    summary = build_narrative(output["facts"], output["anomalies"])
    incident_report = build_incident_report(question, output["facts"], output["anomalies"])
    llm_summary = build_llm_report(
        question=question,
        plan=plan,
        facts=output["facts"],
        anomalies=output["anomalies"],
        model=llm_model,
    )
    if llm_summary:
        summary = llm_summary
    table_preview = output["table"].head(10).to_string(index=False)

    artifacts = {
        "plan": output_dir / "plan.json",
        "facts": output_dir / "facts.json",
        "anomalies": output_dir / "anomalies.json",
        "summary": output_dir / "summary.txt",
        "incident_report": output_dir / "incident_report.md",
        "result_preview": output_dir / "result_preview.txt",
    }
    artifacts["plan"].write_text(json.dumps(plan, indent=2, default=str), encoding="utf-8")
    artifacts["facts"].write_text(json.dumps(output["facts"], indent=2, default=str), encoding="utf-8")
    artifacts["anomalies"].write_text(json.dumps(output["anomalies"], indent=2, default=str), encoding="utf-8")
    artifacts["summary"].write_text(summary + "\n", encoding="utf-8")
    artifacts["incident_report"].write_text(incident_report + "\n", encoding="utf-8")
    artifacts["result_preview"].write_text(table_preview + "\n", encoding="utf-8")
    return {name: str(path) for name, path in artifacts.items()}


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level, log_to=args.log_to, log_file=args.log_file)

    # Optionally allow OPENAI_API_KEY to be provided via config.
    config_key = getattr(args, "openai_api_key", None)
    if config_key:
        if os.getenv("OPENAI_API_KEY"):
            logger.info("OPENAI_API_KEY already set in environment; ignoring config value")
        else:
            os.environ["OPENAI_API_KEY"] = str(config_key)
            logger.info("OPENAI_API_KEY set from JSON config")

    logger.info("CLI args=%s", _json_dumps_safe(_redact_secrets(vars(args))))
    analyst = DemoAnalyst(Path(args.data_dir))
    req = AnalysisRequest(
        question=args.question,
        planner=args.planner,
        execution_mode=args.execution_mode,
        model=args.model,
        dataset=getattr(args, "dataset", None),
    )
    logger.info("AnalysisRequest=%s", _json_dumps_safe(req.__dict__))

    if req.execution_mode == "python":
        output = analyst.execute_python(req.question)
        plan: Dict[str, Any] = {"execution_mode": "python"}
    else:
        plan = analyst.generate_plan(req)
        has_plot = bool(plan.get("plot"))
        if args.skip_plot_generation and has_plot:
            logger.info("Skipping plot generation by configuration despite plot in plan")
        elif not has_plot:
            logger.info("Plan has no plot config; skipping plot generation")
        plot_out = (Path(args.output_dir) / "plot.png") if (has_plot and not args.skip_plot_generation) else None
        output = analyst.execute_plan(plan=plan, question=req.question, plot_out=plot_out)
    logger.info("Run complete facts=%s", _json_dumps_safe(output.get("facts")))
    artifacts = write_reports(
        Path(args.output_dir),
        plan=plan,
        output=output,
        question=req.question,
        llm_model=req.model,
    )
    logger.info("Artifacts written=%s", _json_dumps_safe(artifacts))

    print("=== PLAN ===")
    print(json.dumps(plan, indent=2))
    print("\n=== FACTS ===")
    print(json.dumps(output["facts"], indent=2, default=str))
    print("\n=== SUMMARY ===")
    print(build_narrative(output["facts"], output["anomalies"]))
    print("\n=== INCIDENT REPORT ===")
    print(build_incident_report(req.question, output["facts"], output["anomalies"]))
    print("\n=== RESULT PREVIEW ===")
    print(output["table"].head(10).to_string(index=False))
    print("\n=== ARTIFACTS ===")
    print(json.dumps(artifacts, indent=2))


if __name__ == "__main__":
    main()
