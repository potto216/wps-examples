# Mock LLM scenario testing

This folder contains a file-driven test harness for `llm_guided_stat_analyst_demo.py`.

## Layout

- `mock_llm_scenario_runner.py`: loads a scenario, builds temporary parquet inputs from CSV fixtures, runs the analyst with mock LLM responses, and validates expected outputs.
- `scenarios/<name>/scenario.json`: scenario definition.
- `scenarios/<name>/input/*.csv`: input table fixtures.
- `scenarios/<name>/mock_llm_responses.json`: deterministic LLM outputs.
- `scenarios/<name>/expected/*`: expected plan/facts/summary.

## Add a scenario

1. Create `scenarios/<scenario_name>/`.
2. Add one or more input CSV files and map them in `input_tables`.
3. Add a mock response file with `plan` and `report` entries.
4. Add expected outputs and point to them from `scenario.json`.
5. Add a unittest that calls `assert_scenario`.
