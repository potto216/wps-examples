# LLM-Guided Statistical Analyst Demo (BLE + Wi-Fi)

Wireless packet captures are typically large. Putting raw capture data into an LLM context window is often impractical (or expensive). A better workflow is iterative: use the LLM to propose an analysis approach (queries, aggregations, plots), run it locally against the dataset, inspect the results, and then refine the approach.

This demo explores a few trade-offs:

- Should the system search directly for an answer, or generate a repeatable analysis plan?
- Should the plan be expressed as structured operations (pandas-style steps) or as generated Python code?
- How does an LLM-based planner compare to simple heuristics?

This example provides a simplified but complete pipeline that can be extended:

1. Load capture-derived tables as pandas DataFrames (`ble_adv_events`, `wifi_mgmt_frames`) plus optional `metadata.json`.
2. Convert a natural-language question into a JSON analysis plan (heuristic or LLM-backed).
3. Execute the plan with pandas (or execute a safe Python snippet mode).
4. Produce a concise narrative summary and an incident report when anomalies are detected.
5. Compute the token (and cost difference) between a large and small LLM model or between analysis resulting from a plan or code generation.

## Setup
To use a commercial LLM, you’ll need an API key. For OpenAI, create a key at https://platform.openai.com/settings/organization/api-keys and set it as an environment variable (e.g., `OPENAI_API_KEY`).

This demo also supports providing the OpenAI key in the JSON config file (see below). Using an environment variable is still the simplest approach.

## JSON config file

The script reads a JSON configuration file at startup. By default it looks for `./llm_guided_stat_analyst_demo.json`, but you can override both the directory and filename:

```bash
python usecase/llm_rag_ml/llm_guided_stat_analyst_demo.py \
  --config-dir usecase/llm_rag_ml \
  --config-file llm_guided_stat_analyst_demo.json
```

### Example config

Create `usecase/llm_rag_ml/llm_guided_stat_analyst_demo.json`:

```json
{
  "question": "Which AP had the most probe requests yesterday?",
  "data_dir": "usecase/llm_rag_ml/data",
  "planner": "llm",
  "execution_mode": "plan",
  "model": "gpt-5-nano",
  "output_dir": "usecase/llm_rag_ml/artifacts",
  "skip_plot_generation": false,
  "log_level": "info",
  "log_to": "stdout",
  "log_file": null
}
```

Supported keys:

- `question` (required)
- `data_dir`, `planner`, `execution_mode`, `model`, `output_dir`, `skip_plot_generation`
- `dataset` (optional; forces a specific dataset name when multiple Parquet tables are loaded)
- `log_level`, `log_to`, `log_file`
- `openai_api_key` (optional; see next section)

### Forcing a dataset

If your `data_dir` contains multiple `.parquet` tables, you can force the dataset used for planning/execution.

In JSON config:

```json
{
  "question": "Count packets by highest_layer",
  "planner": "llm",
  "dataset": "x240_bredr_le_2m_20260106_064929"
}
```

Or on the CLI:

```bash
python usecase/llm_rag_ml/llm_guided_stat_analyst_demo.py \
  --planner llm \
  --dataset x240_bredr_le_2m_20260106_064929 \
  --question "Count packets by highest_layer"
```

Why you might see `wifi_mgmt_frames`: if the LLM-generated plan is invalid and fails validation, the script falls back to the heuristic planner, which may default to `wifi_mgmt_frames` depending on the question.

### Adding the OpenAI key to the config file

If you prefer, you can put the key in the JSON config file as `openai_api_key`. When set, the script will populate `OPENAI_API_KEY` from the config (unless it is already set in the environment).

```json
{
  "question": "How often does device aa:bb:cc:dd:ee:ff advertise?",
  "planner": "llm",
  "execution_mode": "plan",
  "model": "gpt-5-nano",
  "openai_api_key": "YOUR_KEY_HERE"
}
```

Security note: avoid committing this file to source control; treat it like a secret.

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
  --output-dir "/home/user/workspace/output" \
  --log-to file \
  --log-file "/home/user/workspace/output/llm_guided_stat_analyst_demo.log" \
  --question "How often does device aa:bb:cc:dd:ee:ff advertise?"
```

The script writes a default plot artifact to `usecase/llm_rag_ml/artifacts/latest_plot.png`.
