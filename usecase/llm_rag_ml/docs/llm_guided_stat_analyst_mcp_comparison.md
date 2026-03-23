# `llm_guided_stat_analyst_demo.py` vs MCP Server Rewrite

## Purpose
This document compares:
1. The **current script-based implementation** in `llm_guided_stat_analyst_demo.py`.
2. A **hypothetical MCP-server rewrite** that exposes packet-analysis capabilities to an LLM through MCP tools/resources.

---

## 1) Current implementation (script-first architecture)

## What it is
The current file is a standalone CLI workflow that:
- Loads parquet capture tables and optional metadata/context files.
- Uses either a heuristic planner or LLM planner to produce an analysis plan.
- Validates and executes the plan with pandas-based operations.
- Produces facts, anomaly output, optional plots, and a narrative/incident report.

## Core architecture
- **Entry-point + argument/config merge**: `parse_args()` and `main()` orchestrate configuration, backend selection, plan creation, execution mode, and report writing.
- **LLM abstraction**:
  - `LLMBackend` protocol.
  - `OpenAIBackend` for live model calls.
  - `MockFileLLMBackend` for deterministic local/offline behavior.
- **Analysis engine**: `DemoAnalyst` loads datasets, builds context, validates plans, executes filters/grouping/metrics, detects anomalies, and writes plots.
- **Execution modes**:
  - `plan`: LLM/heuristic-generated structured analysis plan.
  - `python`: safe snippet-style execution path.

## Strengths
- Very easy to run locally as a demo (`python ... --question ...`).
- End-to-end flow lives in one place, making it straightforward for experimentation.
- Built-in fallback behavior (LLM failure -> heuristic planner).
- Strong guardrails around planner output (schema + validation + repair attempt).
- Useful operational details (logging options, mock backend support, output artifact writing).

## Limitations
- Tight coupling between orchestration, planning, execution, rendering, and I/O in one script.
- Interface is optimized for **humans running a CLI**, not for external LLM clients that want composable capabilities.
- Harder to reuse from multiple agents/apps simultaneously (single-process script lifecycle).
- Tooling discoverability is implicit (knowledge in code/prompts), not explicit in a machine-discoverable MCP tool catalog.
- Multi-tenant/security boundaries are coarse (process-level controls only unless additional wrapping is added).

---

## 2) MCP-server rewrite (tool-first architecture)

## What changes conceptually
Instead of asking an LLM to run this script directly, we expose core capabilities as MCP primitives:

- **Resources** (read-only context):
  - available datasets
  - table schemas
  - metadata summaries
  - last run reports
- **Tools** (actionable operations):
  - create/validate plan
  - run analysis
  - run anomaly detection
  - generate report
  - render plot

The client LLM then calls these tools iteratively based on user questions.

## Likely MCP surface design

### Resources
- `wireless://datasets` → list dataset names + row counts.
- `wireless://schema/{dataset}` → columns, dtypes, time range.
- `wireless://capture_summary` → existing summary metadata.

### Tools
- `plan_analysis(question, dataset_hint?) -> plan_json`
- `validate_plan(plan_json) -> validation_errors[]`
- `execute_plan(plan_json, output_dir?) -> facts, anomalies, table_preview, artifact_paths`
- `run_python_snippet(question, dataset_hint?) -> guarded_result`
- `generate_report(question, facts, anomalies) -> markdown_report`

### Optional prompts/templates
- A reusable planner prompt template and response schema owned by server code.
- A "wireless analyst" instruction prompt offered as an MCP prompt for clients.

---

## 3) Advantages and disadvantages

## Current script approach

### Advantages
- Fast to prototype and debug end-to-end.
- Minimal deployment complexity.
- Works offline with mock LLM backend and local files.
- Easy to reason about execution order (single call path).

### Disadvantages
- Reuse friction: external LLM systems must shell out or reimplement internals.
- Limited composability for iterative agent workflows.
- Harder concurrency/scaling story for shared service usage.
- Changes in one area can affect whole script due to tight coupling.

## MCP server approach

### Advantages
- Native fit for agentic/LLM orchestration ecosystems.
- Clear separation of concerns: client reasoning vs server-side deterministic analytics.
- Better discoverability and interoperability via explicit tools/resources.
- Easier to enforce policy at tool boundaries (allowlists, quotas, auth, auditing).
- Encourages modularization and testability of analysis functions.

### Disadvantages
- More infrastructure and deployment overhead (server runtime, transport, client integration).
- Requires additional API contracts/versioning discipline.
- Can increase latency due to multi-step tool call loops.
- Operational complexity (concurrency, lifecycle, telemetry, authz/authn) moves center stage.

---

## 4) How the MCP server would be written

## Step 1: Refactor core logic into a reusable library layer
Extract these from the script into importable modules:
- data loading/context (`DemoAnalyst` initialization helpers)
- planning and validation
- execution/anomaly/plot functions
- report generation

Keep CLI as a thin wrapper that calls the same library.

## Step 2: Add MCP server entrypoint
Implement an MCP server process (Python SDK) that:
- initializes data directory + backend configuration at startup.
- registers resources and tools.
- maps tool handlers to the refactored library functions.

Pseudo-structure:

```python
# mcp_server.py (illustrative)
from mcp.server import Server
from analyst.core import AnalystService

server = Server("wireless-stat-analyst")
svc = AnalystService(data_dir="...")

@server.resource("wireless://datasets")
def list_datasets():
    return svc.list_datasets()

@server.tool()
def plan_analysis(question: str, dataset_hint: str | None = None):
    req = svc.make_request(question=question, dataset=dataset_hint, planner="llm")
    return svc.generate_plan(req)

@server.tool()
def execute_plan(plan: dict, output_dir: str | None = None):
    return svc.execute_plan(plan, output_dir=output_dir)

if __name__ == "__main__":
    server.run()
```

## Step 3: Enforce tool-level contracts and safety
- JSON-schema validate every tool input.
- Return structured error objects (not stack traces).
- Restrict filesystem writes to configured output directory.
- Add per-tool timeout and row/aggregation limits.

## Step 4: Add observability and governance
- Correlate logs by request/session id.
- Capture metrics: tool latency, failures, token usage for LLM planner calls.
- Add authentication/authorization appropriate to deployment (local dev vs shared service).

## Step 5: Version for compatibility
- Tool names and payload schemas should be semver-governed.
- Add compatibility tests so agent clients do not break when internals evolve.

---

## 5) Practical migration path

1. **Modularize without behavior changes**: move script internals into modules; keep CLI stable.
2. **Introduce MCP wrapper**: start with read-only resources + one execution tool.
3. **Expand toolset**: add planning/validation/reporting tools incrementally.
4. **Dual-run period**: keep both CLI and MCP server until client adoption is complete.
5. **Harden production concerns**: auth, quotas, audit logs, SLOs, schema/version governance.

---

## Recommendation
If this use case is mostly a local demonstration run by one engineer, keep the script-first model.
If the goal is to let multiple LLM agents/apps ask wireless-capture questions reliably and repeatedly, an MCP server rewrite is the better long-term architecture.

In practice, the best path is **not a rewrite from scratch**, but a **progressive refactor**: keep the current script as a compatibility frontend and expose the same core analysis engine through MCP tools/resources.
