# Tool API integration notes (companion)

This note maps the current tabular analysis plan to a `ToolInvocation`/`ToolResult` workflow and points to concrete integration anchors in the demo analyst code.

## Search anchors (code navigation)

- `DemoAnalyst._llm_plan` → search for: `def _llm_plan(`
- `plan_schema` (inside planner) → search for: `plan_schema = {`
- `generate_plan` (planner entrypoint) → search for: `def generate_plan(`

Primary file: `usecase/llm_rag_ml/llm_guided_stat_analyst_demo.py`.

## 1) Field mapping: existing plan → `ToolInvocation`

Use one `ToolInvocation` per executable analysis step.

- `dataset` → `capture_selection.capture_id`
  - If multiple datasets are needed, emit multiple `ToolInvocation`s.
- `filters` → `capture_selection.filters`
  - Preserve normalized predicate objects or serialize to the tool's accepted filter syntax.
- `groupby` → `arguments.groupby`
- `metrics` → `arguments.metrics`
- `plot` → `arguments.visualization` (or `arguments.plot` if kept as-is)
- `anomaly` → `arguments.anomaly`

Suggested envelope:

```json
{
  "tool_name": "stat_analyst",
  "tool_version": "v1",
  "capture_selection": {
    "capture_id": "<dataset>",
    "filters": ["<translated filter predicates>"]
  },
  "arguments": {
    "groupby": ["..."],
    "metrics": [{"name": "count"}],
    "visualization": {"type": "timeseries", "x": "timestamp", "y": "count"},
    "anomaly": {"method": "zscore", "on": "count", "threshold": 3.0}
  },
  "request_id": "<uuid>",
  "timeout_ms": 30000
}
```

## 2) Prompt injection location in `DemoAnalyst._llm_plan`

Inject tool metadata in two places in `DemoAnalyst._llm_plan`:

1. **Planner instructions (`prompt` string) before `messages` are built**
   - Add a short "available tools" section to the existing prompt, including:
     - tool names/versions,
     - input schema summary,
     - constraints (required args, max filters, etc.).
2. **`plan_schema` construction block**
   - Extend schema to allow a hybrid output shape (tabular fields + optional `tool_calls[]`) or a pure `tool_calls[]` schema in later phases.

Practical anchor flow:

- `def _llm_plan(`
- prompt assembly (`prompt = (...)`)
- `messages = [...]`
- `plan_schema = {...}`
- API call with `response_format` using that schema

## 3) Representing `ToolResult` for downstream LLM reasoning

Represent each execution response as a compact, LLM-friendly evidence object, then pass a list of those objects into synthesis/refinement prompts.

Recommended shape:

```json
{
  "tool_name": "stat_analyst",
  "tool_version": "v1",
  "request_id": "...",
  "status": "ok|partial|error",
  "summary": "1-3 sentence human-readable finding",
  "confidence": 0.0,
  "warnings": [],
  "errors": [],
  "structured_output": {
    "key_metrics": {"count": 1234},
    "tables": [{"name": "grouped_counts", "rows": []}],
    "artifacts": [{"type": "plot", "uri": "plots/xyz.png"}]
  }
}
```

Guidelines:

- Keep `summary` and top-level `key_metrics` always present when possible.
- Preserve `warnings`/`errors` verbatim to support automatic replanning.
- Avoid dumping full raw tables; send compact slices + stable artifact URIs.

## 4) Migration path

### Phase 1: sidecar tool calls for advanced analysis

- Keep current `generate_plan` and existing tabular plan schema as primary.
- Add optional sidecar execution after normal plan execution for advanced requests (e.g., richer anomaly methods).
- Feed sidecar `ToolResult` into the final narrative/synthesis step.

### Phase 2: planner emits hybrid plans (tabular ops + tool calls)

- Update `plan_schema` in `_llm_plan` to support optional `tool_calls` array.
- `generate_plan` returns a plan that may contain both:
  - existing fields (`dataset`, `filters`, `metrics`, ...), and
  - `tool_calls: ToolInvocation[]`.
- Executor runs tabular steps first, then tool calls (or vice versa when dependencies require).

### Phase 3: optional fully tool-centric planning

- Introduce a tool-centric schema where the main output is ordered `ToolInvocation[]`.
- Keep backward-compatible fallback to heuristic/tabular planning behind a flag.
- Over time, reduce direct tabular planning to a compatibility mode.

