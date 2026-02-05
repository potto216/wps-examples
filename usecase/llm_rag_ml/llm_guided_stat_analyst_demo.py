from __future__ import annotations

import argparse
import json
import os
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

        if ble_path.exists() and wifi_path.exists():
            self.tables = {
                "ble_adv_events": pd.read_csv(ble_path, parse_dates=["timestamp"]),
                "wifi_mgmt_frames": pd.read_csv(wifi_path, parse_dates=["timestamp"]),
            }
        else:
            self.tables = self._demo_tables()

        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as handle:
                self.metadata = json.load(handle)
        else:
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
        if request.planner == "llm":
            plan = self._llm_plan(request)
            if plan:
                return plan
        return self._heuristic_plan(request.question)

    def _llm_plan(self, request: AnalysisRequest) -> Optional[Dict[str, Any]]:
        if OpenAI is None or not os.getenv("OPENAI_API_KEY"):
            return None

        prompt = (
            "Return JSON only for analysis plan with keys: dataset, filters, groupby, metrics, "
            "plot, anomaly. Datasets available: ble_adv_events and wifi_mgmt_frames. "
            f"Question: {request.question}"
        )
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=request.model,
            messages=[
                {"role": "system", "content": "You are a wireless packet analytics planner."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            return None

    def _heuristic_plan(self, question: str) -> Dict[str, Any]:
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

        result_table = frame
        if plan.get("groupby"):
            grouped = frame.groupby(plan["groupby"], dropna=False)
            agg = pd.DataFrame(index=grouped.size().index)
            for metric in plan.get("metrics", []):
                if metric["name"] == "count":
                    agg["count"] = grouped.size()
            result_table = agg.reset_index().sort_values("count", ascending=False)
            facts["top_group"] = result_table.iloc[0].to_dict() if not result_table.empty else {}

        # Additional computed feature for timing-style questions.
        if "timing" in question.lower() or "distribution" in question.lower():
            timed = frame.sort_values("timestamp").copy()
            timed["inter_arrival_sec"] = timed["timestamp"].diff().dt.total_seconds()
            result_table = timed.dropna(subset=["inter_arrival_sec"])
            facts["inter_arrival_mean"] = float(result_table["inter_arrival_sec"].mean()) if not result_table.empty else None
            facts["inter_arrival_p95"] = float(result_table["inter_arrival_sec"].quantile(0.95)) if not result_table.empty else None

        anomaly_report = self._anomaly_detect(plan.get("anomaly", {}), result_table)

        if plot_out:
            self._plot(plan.get("plot", {}), result_table, plot_out)
            facts["plot_path"] = str(plot_out)

        return {
            "table": result_table,
            "facts": facts,
            "anomalies": anomaly_report,
        }

    def execute_python(self, question: str) -> Dict[str, Any]:
        """Safe, tiny code execution mode with prebuilt snippets (extendable)."""
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
    parser.add_argument("--data-dir", default="usecase/llm_rag_ml/data", help="Directory with CSV tables + metadata")
    parser.add_argument("--question", required=True, help="Question in natural language")
    parser.add_argument("--planner", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--execution-mode", choices=["plan", "python"], default="plan")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--plot-out", default="usecase/llm_rag_ml/artifacts/latest_plot.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyst = DemoAnalyst(Path(args.data_dir))
    req = AnalysisRequest(
        question=args.question,
        planner=args.planner,
        execution_mode=args.execution_mode,
        model=args.model,
    )

    if req.execution_mode == "python":
        output = analyst.execute_python(req.question)
        plan: Dict[str, Any] = {"execution_mode": "python"}
    else:
        plan = analyst.generate_plan(req)
        output = analyst.execute_plan(plan=plan, question=req.question, plot_out=Path(args.plot_out))

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


if __name__ == "__main__":
    main()
