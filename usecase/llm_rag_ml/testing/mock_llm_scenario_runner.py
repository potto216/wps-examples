from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd


@dataclass
class ScenarioResult:
    plan: Dict[str, Any]
    facts: Dict[str, Any]
    summary: str
    artifacts: Dict[str, str]


def _load_demo_module() -> Any:
    script_path = Path(__file__).resolve().parents[1] / "llm_guided_stat_analyst_demo.py"
    spec = importlib.util.spec_from_file_location("llm_guided_stat_analyst_demo", script_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("Failed to load llm_guided_stat_analyst_demo module")
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def run_scenario(scenario_file: str | Path) -> ScenarioResult:
    scenario_path = Path(scenario_file).resolve()
    root = scenario_path.parent
    scenario = _read_json(scenario_path)

    module = _load_demo_module()

    with tempfile.TemporaryDirectory(prefix="mock-llm-scenario-") as td:
        tmp = Path(td)
        data_dir = tmp / "data"
        out_dir = tmp / "out"
        data_dir.mkdir(parents=True, exist_ok=True)

        for table in scenario.get("input_tables", []):
            table_name = table["table_name"]
            source = root / table["csv_file"]
            frame = pd.read_csv(source)
            frame.to_parquet(data_dir / f"{table_name}.parquet")

        llm_backend = module.create_llm_backend(
            "mock-file",
            mock_llm_responses_file=str((root / scenario["mock_llm_responses_file"]).resolve()),
        )
        analyst = module.DemoAnalyst(data_dir, llm_backend=llm_backend)
        request = module.AnalysisRequest(
            question=scenario["question"],
            planner="llm",
            execution_mode="plan",
            model=scenario.get("model", "mock-model"),
            dataset=scenario.get("dataset"),
        )

        plan = analyst.generate_plan(request)
        output = analyst.execute_plan(plan=plan, question=request.question, plot_out=None)
        artifacts = module.write_reports(
            out_dir,
            plan=plan,
            output=output,
            question=request.question,
            llm_model=request.model,
            llm_backend=llm_backend,
        )
        summary = Path(artifacts["summary"]).read_text(encoding="utf-8").strip()

        return ScenarioResult(
            plan=plan,
            facts=output["facts"],
            summary=summary,
            artifacts=artifacts,
        )


def assert_scenario(scenario_file: str | Path) -> None:
    scenario_path = Path(scenario_file).resolve()
    root = scenario_path.parent
    scenario = _read_json(scenario_path)
    result = run_scenario(scenario_path)

    expected_plan_path = scenario.get("expected_plan_file")
    if expected_plan_path:
        expected_plan = _read_json(root / expected_plan_path)
        if result.plan != expected_plan:
            raise AssertionError(f"Plan mismatch\nExpected: {expected_plan}\nActual: {result.plan}")

    expected_facts_path = scenario.get("expected_facts_file")
    if expected_facts_path:
        expected_facts = _read_json(root / expected_facts_path)
        for key, value in expected_facts.items():
            if result.facts.get(key) != value:
                raise AssertionError(
                    f"Facts mismatch for key '{key}'. Expected={value!r} Actual={result.facts.get(key)!r}"
                )

    expected_summary_path = scenario.get("expected_summary_file")
    if expected_summary_path:
        expected_summary = (root / expected_summary_path).read_text(encoding="utf-8").strip()
        if result.summary != expected_summary:
            raise AssertionError(
                f"Summary mismatch\nExpected: {expected_summary}\nActual: {result.summary}"
            )
