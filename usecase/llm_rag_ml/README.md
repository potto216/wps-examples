# LLM-Guided Statistical Analyst Demo (BLE + Wi-Fi)

Wireless packet captures are typically large. Putting raw capture data into an LLM context window is often impractical (or expensive). A better workflow is iterative: use the LLM to propose an analysis approach (queries, aggregations, plots), run it locally against the dataset, inspect the results, and then refine the approach.

This demo explores a few trade-offs:

- Should the system search directly for an answer, or generate a repeatable analysis plan?
- Should the plan be expressed as structured operations (pandas-style steps) or as generated Python code?
- How does an LLM-based planner compare to simple heuristics?

This example provides a simplified but complete pipeline that can be extended in coursework:

1. Load capture-derived tables as pandas DataFrames (`ble_adv_events`, `wifi_mgmt_frames`) plus optional `metadata.json`.
2. Convert a natural-language question into a JSON analysis plan (heuristic or LLM-backed).
3. Execute the plan with pandas (or execute a safe Python snippet mode).
4. Produce a concise narrative summary and an incident report when anomalies are detected.

## Setup
To use a commercial LLM, you’ll need an API key. For OpenAI, create a key at https://platform.openai.com/settings/organization/api-keys and set it as an environment variable (e.g., `OPENAI_API_KEY`).

## Files

- `llm_guided_stat_analyst_demo.py` – CLI app for planning, execution, plotting, and reporting.
- Built-in synthetic tables are used by default so the demo runs without extra files.
- Optional `data/` folder can override inputs with `ble_adv_events.csv`, `wifi_mgmt_frames.csv`, and `metadata.json`.

## Run examples

Example questions:
```bash
python usecase/llm_rag_ml/llm_guided_stat_analyst_demo.py \
  --question "How often does device aa:bb:cc:dd:ee:ff advertise?"

python usecase/llm_rag_ml/llm_guided_stat_analyst_demo.py \
  --question "Show scan/connection timing distribution"

python usecase/llm_rag_ml/llm_guided_stat_analyst_demo.py \
  --question "Which AP had the most probe requests yesterday?"
```

Optional LLM plan generation:

A generic example using OpenAI:
```bash
export OPENAI_API_KEY=...
python usecase/llm_rag_ml/llm_guided_stat_analyst_demo.py \
  --planner llm \
  --question "Which AP had the most probe requests yesterday?"
```

A more specific example with logging:
```bash
export OPENAI_API_KEY=...
python usecase/llm_rag_ml/llm_guided_stat_analyst_demo.py \
  --log-level debug \
  --data-dir "/home/user/workspace/data" \
  --planner "llm" \
  --execution-mode "plan" \
  --model "gpt-5-nano" \
  --plot-out "/home/user/workspace/output" \
  --log-to file \
  --log-file "/home/user/workspace/output/llm_guided_stat_analyst_demo.log" \
  --question "How often does device aa:bb:cc:dd:ee:ff advertise?"
```

The script writes a default plot artifact to `usecase/llm_rag_ml/artifacts/latest_plot.png`.
