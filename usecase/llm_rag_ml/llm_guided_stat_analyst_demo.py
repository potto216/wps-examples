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


@dataclass
class AnalysisRequest:
    question: str
    planner: str
    execution_mode: str
    model: str


class DemoAnalyst:
    """Simple end-to-end analyst using plan execution or safe Python snippets."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        ble_path = data_dir / "ble_adv_events.csv"
        wifi_path = data_dir / "wifi_mgmt_frames.csv"
        metadata_path = data_dir / "metadata.json"

        logger.info("Initializing analyst data_dir=%s", str(data_dir))
        if ble_path.exists() and wifi_path.exists():
            logger.info("Loading CSV tables ble=%s wifi=%s", str(ble_path), str(wifi_path))
            self.tables = {
                "ble_adv_events": pd.read_csv(ble_path, parse_dates=["timestamp"]),
                "wifi_mgmt_frames": pd.read_csv(wifi_path, parse_dates=["timestamp"]),
            }
        else:
            logger.info("CSV tables missing; using built-in synthetic demo tables")
            self.tables = self._demo_tables()

        try:
            logger.info(
                "Tables loaded rows ble_adv_events=%d wifi_mgmt_frames=%d",
                int(len(self.tables.get("ble_adv_events", []))),
                int(len(self.tables.get("wifi_mgmt_frames", []))),
            )
        except Exception:
            logger.debug("Could not log table sizes", exc_info=True)

        if metadata_path.exists():
            logger.info("Loading metadata=%s", str(metadata_path))
            with open(metadata_path, "r", encoding="utf-8") as handle:
                self.metadata = json.load(handle)
        else:
            logger.info("Metadata missing; using synthetic metadata")
            self.metadata = {
                "environment": "synthetic_demo",
                "known_devices_present": sorted(
                    set(self.tables["ble_adv_events"]["advertiser_addr"].tolist())
                ),
                "expected_anomaly_windows": [
                    {
                        "start": "2025-01-09T12:10:00Z",
                        "end": "2025-01-09T12:10:03Z",
                        "description": "High probe request period",
                    }
                ],
            }

    @staticmethod
    def _demo_tables() -> Dict[str, pd.DataFrame]:
        ble_rows = [
            ["2025-01-10T09:00:00Z", "aa:bb:cc:dd:ee:ff", -55, "ADV_IND", 37, 32, "004C", "180D|180F", 12],
            ["2025-01-10T09:00:01Z", "aa:bb:cc:dd:ee:ff", -54, "ADV_IND", 38, 30, "004C", "180D", 10],
            ["2025-01-10T09:00:02Z", "aa:bb:cc:dd:ee:ff", -56, "ADV_IND", 39, 31, "004C", "180D|180F", 11],
            ["2025-01-10T09:00:10Z", "11:22:33:44:55:66", -70, "ADV_NONCONN_IND", 37, 28, "00E0", "FEAA", 8],
            ["2025-01-10T09:00:20Z", "11:22:33:44:55:66", -71, "ADV_NONCONN_IND", 38, 29, "00E0", "FEAA", 8],
            ["2025-01-10T09:00:30Z", "11:22:33:44:55:66", -69, "ADV_NONCONN_IND", 39, 27, "00E0", "FEAA", 8],
            ["2025-01-10T09:01:00Z", "77:88:99:aa:bb:cc", -62, "ADV_SCAN_IND", 37, 33, "1234", "180A", 9],
            ["2025-01-10T09:01:01Z", "77:88:99:aa:bb:cc", -61, "ADV_SCAN_IND", 38, 35, "1234", "180A", 9],
            ["2025-01-10T09:01:02Z", "77:88:99:aa:bb:cc", -63, "ADV_SCAN_IND", 39, 34, "1234", "180A", 9],
            ["2025-01-10T09:01:20Z", "aa:bb:cc:dd:ee:ff", -58, "ADV_IND", 37, 30, "004C", "180D", 10],
            ["2025-01-10T09:01:21Z", "aa:bb:cc:dd:ee:ff", -58, "ADV_IND", 38, 31, "004C", "180D", 10],
            ["2025-01-10T09:01:22Z", "aa:bb:cc:dd:ee:ff", -59, "ADV_IND", 39, 31, "004C", "180D", 10],
        ]
        wifi_rows = [
            ["2025-01-09T12:00:00Z", "10:10:10:10:10:01", "aa:aa:aa:aa:aa:01", "beacon", "CampusWiFi", "6|12|24", "ESS|PRIVACY", 1, -48],
            ["2025-01-09T12:00:01Z", "10:10:10:10:10:02", "aa:aa:aa:aa:aa:02", "beacon", "CafeGuest", "6|12|24", "ESS|PRIVACY", 6, -56],
            ["2025-01-09T12:00:05Z", "20:20:20:20:20:01", "aa:aa:aa:aa:aa:01", "probe_req", None, None, None, 1, -60],
            ["2025-01-09T12:00:06Z", "20:20:20:20:20:02", "aa:aa:aa:aa:aa:01", "probe_req", None, None, None, 1, -62],
            ["2025-01-09T12:00:07Z", "20:20:20:20:20:03", "aa:aa:aa:aa:aa:02", "probe_req", None, None, None, 6, -64],
            ["2025-01-09T12:00:08Z", "20:20:20:20:20:04", "aa:aa:aa:aa:aa:01", "probe_req", None, None, None, 1, -63],
            ["2025-01-09T12:00:10Z", "30:30:30:30:30:01", "aa:aa:aa:aa:aa:01", "assoc_req", "CampusWiFi", "6|12|24", "ESS|PRIVACY", 1, -51],
            ["2025-01-09T12:00:11Z", "30:30:30:30:30:02", "aa:aa:aa:aa:aa:01", "assoc_req", "CampusWiFi", "6|12|24", "ESS|PRIVACY", 1, -53],
            ["2025-01-09T12:10:00Z", "20:20:20:20:20:10", "aa:aa:aa:aa:aa:03", "probe_req", None, None, None, 11, -66],
            ["2025-01-09T12:10:01Z", "20:20:20:20:20:11", "aa:aa:aa:aa:aa:03", "probe_req", None, None, None, 11, -65],
            ["2025-01-09T12:10:02Z", "20:20:20:20:20:12", "aa:aa:aa:aa:aa:03", "probe_req", None, None, None, 11, -64],
            ["2025-01-09T12:10:03Z", "20:20:20:20:20:13", "aa:aa:aa:aa:aa:03", "probe_req", None, None, None, 11, -63],
            ["2025-01-10T09:00:00Z", "20:20:20:20:20:21", "aa:aa:aa:aa:aa:02", "probe_req", None, None, None, 6, -58],
            ["2025-01-10T09:00:01Z", "20:20:20:20:20:22", "aa:aa:aa:aa:aa:02", "probe_req", None, None, None, 6, -57],
        ]
        ble = pd.DataFrame(
            ble_rows,
            columns=["timestamp", "advertiser_addr", "rssi", "adv_type", "channel", "length", "company_id", "service_uuids", "mfg_data_len"],
        )
        wifi = pd.DataFrame(
            wifi_rows,
            columns=["timestamp", "transmitter_mac", "receiver_mac", "frame_subtype", "ssid", "supported_rates", "capabilities_flags", "channel", "rssi"],
        )
        ble["timestamp"] = pd.to_datetime(ble["timestamp"], utc=True)
        wifi["timestamp"] = pd.to_datetime(wifi["timestamp"], utc=True)
        return {"ble_adv_events": ble, "wifi_mgmt_frames": wifi}

    def generate_plan(self, request: AnalysisRequest) -> Dict[str, Any]:
        logger.info("Generating plan planner=%s model=%s", request.planner, request.model)
        if request.planner == "llm":
            plan = self._llm_plan(request)
            if plan:
                logger.info("LLM planner returned a plan")
                logger.debug("LLM plan JSON=%s", _json_dumps_safe(plan))
                return plan
            logger.warning("LLM planner unavailable/failed; falling back to heuristic planner")
        return self._heuristic_plan(request.question)

    def _llm_plan(self, request: AnalysisRequest) -> Optional[Dict[str, Any]]:
        api_key_set = bool(os.getenv("OPENAI_API_KEY"))
        if OpenAI is None or not api_key_set:
            logger.info(
                "LLM planner not available openai_imported=%s OPENAI_API_KEY_set=%s",
                OpenAI is not None,
                api_key_set,
            )
            return None

        prompt = (
            "Return JSON only for analysis plan with keys: dataset, filters, groupby, metrics, "
            "plot, anomaly. Datasets available: ble_adv_events and wifi_mgmt_frames. "
            f"Question: {request.question}"
        )

        # Prefer `developer` message for instructions (replaces `system` for newer models).
        messages = [
            {"role": "developer", "content": "You are a wireless packet analytics planner."},
            {"role": "user", "content": prompt},
        ]

        # Strict schema for the plan you expect.
        plan_schema = {
            "name": "analysis_plan",
            "description": "Planner output for wireless packet analytics.",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["dataset", "filters", "groupby", "metrics", "plot", "anomaly"],
                "properties": {
                    "dataset": {"type": "string", "enum": ["ble_adv_events", "wifi_mgmt_frames"]},

                    "filters": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["col", "op", "value"],
                            "properties": {
                                "col": {"type": "string"},
                                "op": {"type": "string"},
                                "value": {"type": "string"},
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

    def _heuristic_plan(self, question: str) -> Dict[str, Any]:
        logger.info("Using heuristic planner")
        q = question.lower()
        if "advertise" in q:
            device = "aa:bb:cc:dd:ee:ff"
            for token in question.replace("?", "").split():
                if token.count(":") == 5:
                    device = token.lower()
            return {
                "dataset": "ble_adv_events",
                "filters": [{"col": "advertiser_addr", "op": "==", "value": device}],
                "groupby": ["advertiser_addr"],
                "metrics": [{"name": "count"}],
                "plot": {"type": "timeseries", "x": "timestamp", "y": "events"},
                "anomaly": {"method": "robust_z", "on": "count", "threshold": 3.5},
            }

        if "probe" in q and ("ap" in q or "bssid" in q):
            return {
                "dataset": "wifi_mgmt_frames",
                "filters": [{"col": "frame_subtype", "op": "==", "value": "probe_req"}],
                "groupby": ["receiver_mac"],
                "metrics": [{"name": "count"}],
                "plot": {"type": "bar", "x": "receiver_mac", "y": "count"},
                "anomaly": {"method": "robust_z", "on": "count", "threshold": 3.5},
            }

        return {
            "dataset": "wifi_mgmt_frames",
            "filters": [
                {"col": "frame_subtype", "op": "in", "value": ["probe_req", "assoc_req"]}
            ],
            "groupby": ["frame_subtype"],
            "metrics": [{"name": "count"}],
            "plot": {"type": "hist", "x": "inter_arrival_sec", "bins": 20},
            "anomaly": {"method": "rolling_shift", "on": "count", "window": 5, "threshold": 2.5},
        }

    def execute_plan(self, plan: Dict[str, Any], question: str, plot_out: Optional[Path]) -> Dict[str, Any]:
        dataset = plan["dataset"]
        logger.info("Executing plan dataset=%s plot_out=%s", dataset, str(plot_out) if plot_out else None)
        logger.debug("Plan details=%s", _json_dumps_safe(plan))
        frame = self.tables[dataset].copy()

        for f in plan.get("filters", []):
            if f["op"] == "==":
                frame = frame[frame[f["col"]] == f["value"]]
            elif f["op"] == "in":
                frame = frame[frame[f["col"]].isin(f["value"])]

        facts: Dict[str, Any] = {
            "dataset": dataset,
            "rows_after_filter": int(len(frame)),
            "question": question,
        }
        logger.info("Rows after filter=%d", int(len(frame)))

        result_table = frame
        if plan.get("groupby"):
            grouped = frame.groupby(plan["groupby"], dropna=False)
            agg = pd.DataFrame(index=grouped.size().index)
            for metric in plan.get("metrics", []):
                if metric["name"] == "count":
                    agg["count"] = grouped.size()
            result_table = agg.reset_index().sort_values("count", ascending=False)
            facts["top_group"] = result_table.iloc[0].to_dict() if not result_table.empty else {}
            if facts.get("top_group"):
                logger.info("Top group=%s", _json_dumps_safe(facts["top_group"]))

        # Additional computed feature for timing-style questions.
        if "timing" in question.lower() or "distribution" in question.lower():
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
            code = (
                "df = ble_adv_events.copy()\n"
                "result = df.groupby('advertiser_addr').size().reset_index(name='count').sort_values('count', ascending=False)"
            )
        else:
            code = (
                "df = wifi_mgmt_frames[wifi_mgmt_frames['frame_subtype']=='probe_req'].copy()\n"
                "result = df.groupby('receiver_mac').size().reset_index(name='count').sort_values('count', ascending=False)"
            )
        logger.debug("Python snippet code=%s", code)

        local_vars = {
            "ble_adv_events": self.tables["ble_adv_events"],
            "wifi_mgmt_frames": self.tables["wifi_mgmt_frames"],
            "pd": pd,
            "np": np,
        }
        exec(code, {"__builtins__": {}}, local_vars)
        result = local_vars["result"]
        return {
            "table": result,
            "facts": {
                "mode": "python",
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
    parser.add_argument("--data-dir", default=argparse.SUPPRESS, help="Directory with CSV tables + metadata")
    parser.add_argument("--question", default=argparse.SUPPRESS, help="Question in natural language")
    parser.add_argument("--planner", choices=["heuristic", "llm"], default=argparse.SUPPRESS)
    parser.add_argument("--execution-mode", choices=["plan", "python"], default=argparse.SUPPRESS)
    parser.add_argument("--model", default=argparse.SUPPRESS)
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
        "model": "gpt-4o-mini",
        "output_dir": "usecase/llm_rag_ml/artifacts",
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


def write_reports(output_dir: Path, *, plan: Dict[str, Any], output: Dict[str, Any], question: str) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_narrative(output["facts"], output["anomalies"])
    incident_report = build_incident_report(question, output["facts"], output["anomalies"])
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
    logger.info("CLI args=%s", _json_dumps_safe(vars(args)))
    analyst = DemoAnalyst(Path(args.data_dir))
    req = AnalysisRequest(
        question=args.question,
        planner=args.planner,
        execution_mode=args.execution_mode,
        model=args.model,
    )
    logger.info("AnalysisRequest=%s", _json_dumps_safe(req.__dict__))

    if req.execution_mode == "python":
        output = analyst.execute_python(req.question)
        plan: Dict[str, Any] = {"execution_mode": "python"}
    else:
        plan = analyst.generate_plan(req)
        plot_out = Path(args.output_dir) / "plot.png"
        output = analyst.execute_plan(plan=plan, question=req.question, plot_out=plot_out)
    logger.info("Run complete facts=%s", _json_dumps_safe(output.get("facts")))
    artifacts = write_reports(Path(args.output_dir), plan=plan, output=output, question=req.question)
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
