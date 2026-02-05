# LLM-Guided Statistical Analyst Demo (BLE + Wi-Fi)

This example provides a simplified but complete pipeline that can be extended in coursework:

1. Load capture-derived tables as pandas DataFrames (`ble_adv_events`, `wifi_mgmt_frames`) plus optional `metadata.json`.
2. Convert a natural-language question into a JSON analysis plan (heuristic or LLM-backed).
3. Execute the plan with pandas (or execute a safe Python snippet mode).
4. Produce a concise narrative summary and an incident report when anomalies are detected.

## Files

- `llm_guided_stat_analyst_demo.py` – CLI app for planning, execution, plotting, and reporting.
- Built-in synthetic tables are used by default so the demo runs without extra files.
- Optional `data/` folder can override inputs with `ble_adv_events.csv`, `wifi_mgmt_frames.csv`, and `metadata.json`.

## Run examples

```bash
python usecase/llm_rag_ml/llm_guided_stat_analyst_demo.py \
  --question "How often does device aa:bb:cc:dd:ee:ff advertise?"

python usecase/llm_rag_ml/llm_guided_stat_analyst_demo.py \
  --question "Show scan/connection timing distribution"

python usecase/llm_rag_ml/llm_guided_stat_analyst_demo.py \
  --question "Which AP had the most probe requests yesterday?"
```

Optional LLM plan generation:

```bash
export OPENAI_API_KEY=...
python usecase/llm_rag_ml/llm_guided_stat_analyst_demo.py \
  --planner llm \
  --question "Which AP had the most probe requests yesterday?"
```

The script writes a default plot artifact to `usecase/llm_rag_ml/artifacts/latest_plot.png`.
