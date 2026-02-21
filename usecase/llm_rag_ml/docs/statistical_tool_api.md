# Statistical Tool API Interface Contract

This document defines a strict, machine-oriented contract for statistical tools invoked by an LLM runtime. The contract is intentionally explicit so an LLM can reliably produce valid invocations and parse results.

## 1) `ToolManifest` schema

`ToolManifest` describes what a tool is, what it expects, what it returns, and how it should be run.

### Canonical JSON shape

```json
{
  "name": "string",
  "version": "string (major.minor.patch)",
  "description": "string",
  "capabilities": ["string"],
  "input_schema": {
    "type": "object",
    "properties": {},
    "required": []
  },
  "output_schema": {
    "type": "object",
    "properties": {},
    "required": []
  },
  "execution_constraints": {
    "max_timeout_ms": "integer",
    "max_payload_bytes": "integer",
    "supports_streaming": "boolean",
    "side_effects": "none|read_only|external_write"
  },
  "cost_hint": {
    "unit": "call|second|record",
    "estimated_cost": "number",
    "currency": "string"
  },
  "deterministic": "boolean"
}
```

### Field semantics

- `name` (string): Stable unique identifier for the tool. Must be lowercase snake_case (for example: `anova_tool`).
- `version` (string): Semantic version in `major.minor.patch` format.
- `description` (string): Human-readable summary of purpose and scope.
- `capabilities` (array of string): Enumerated actions supported by this tool (for example: `"linear_regression"`, `"anova"`, `"summary_stats"`).
- `input_schema` (JSON Schema object): JSON Schema for `ToolInvocation.arguments`.
- `output_schema` (JSON Schema object): JSON Schema for `ToolResult.structured_output` when `status` is `ok` or `partial`.
- `execution_constraints` (object): Runtime limits and behavior guarantees.
  - `max_timeout_ms` (integer): Hard upper bound for allowed timeout.
  - `max_payload_bytes` (integer): Maximum serialized request size.
  - `supports_streaming` (boolean): Whether incremental output is supported.
  - `side_effects` (enum): Side-effect class.
- `cost_hint` (object): Optional pricing guidance.
  - `unit` (enum): Billing unit.
  - `estimated_cost` (number): Approximate cost per unit.
  - `currency` (string): ISO currency code (for example `USD`) or pseudo-unit (`credits`).
- `deterministic` (boolean): `true` if identical invocation inputs are expected to produce identical outputs.

## 2) `ToolInvocation` schema

`ToolInvocation` is the request envelope an orchestrator sends to execute the tool.

### Canonical JSON shape

```json
{
  "tool_name": "string",
  "tool_version": "string (major.minor.patch)",
  "capture_selection": {
    "capture_id": "string",
    "selectors": {
      "time_range": {
        "start_ms": "integer",
        "end_ms": "integer"
      },
      "channels": ["string"],
      "filters": ["string"]
    }
  },
  "arguments": {},
  "request_id": "string",
  "timeout_ms": "integer"
}
```

### Field semantics

- `tool_name` (string): Must exactly match `ToolManifest.name`.
- `tool_version` (string): Version requested by caller; should match a supported version in the runtime.
- `capture_selection` (object): Scope of data to operate on.
  - `capture_id` (string): Identifier of dataset/capture.
  - `selectors` (object): Sub-selection directives.
    - `time_range.start_ms` and `time_range.end_ms` (integer): Inclusive bounds in milliseconds.
    - `channels` (array of string): Optional channel list.
    - `filters` (array of string): Optional predicate list expressed in implementation-specific syntax.
- `arguments` (object): Tool-specific parameters, validated by `ToolManifest.input_schema`.
- `request_id` (string): Caller-provided correlation id; must be unique per invocation attempt.
- `timeout_ms` (integer): Desired timeout; runtime MUST clamp to `execution_constraints.max_timeout_ms`.

## 3) `ToolResult` schema

`ToolResult` is the normalized response envelope.

### Canonical JSON shape

```json
{
  "status": "ok",
  "summary": "string",
  "structured_output": {},
  "artifacts": [
    {
      "name": "string",
      "mime_type": "string",
      "uri": "string",
      "sha256": "string"
    }
  ],
  "warnings": [
    {
      "code": "string",
      "message": "string"
    }
  ],
  "errors": [
    {
      "code": "string",
      "message": "string",
      "field": "string"
    }
  ],
  "confidence": 0.0
}
```

### Field semantics

- `status` (enum): `ok`, `partial`, or `error`.
  - `ok`: Request succeeded with complete expected output.
  - `partial`: Some output produced, but with known gaps/degradations.
  - `error`: Execution failed; `structured_output` should be omitted or empty.
- `summary` (string): Concise natural-language summary suitable for LLM response composition.
- `structured_output` (object): Machine-readable result validated by `ToolManifest.output_schema` when `status` is `ok` or `partial`.
- `artifacts` (array): Optional linked outputs (plots, CSV files, model binaries).
  - `name` (string): Artifact label.
  - `mime_type` (string): MIME type (for example `text/csv`, `image/png`).
  - `uri` (string): Fetchable location.
  - `sha256` (string): Integrity digest for content.
- `warnings` (array): Non-fatal issues.
- `errors` (array): Fatal or validation errors. Must be populated when `status=error`.
  - `field` identifies the problematic input field when applicable.
- `confidence` (number): Range `[0.0, 1.0]`; model/tool confidence in correctness or representativeness.

## 4) JSON examples

Complete request/response pair for a regression run.

### Example `ToolInvocation`

```json
{
  "tool_name": "statistical_regression_tool",
  "tool_version": "1.2.0",
  "capture_selection": {
    "capture_id": "cap_2026_03_14_a",
    "selectors": {
      "time_range": {
        "start_ms": 0,
        "end_ms": 120000
      },
      "channels": ["ch1", "ch2"],
      "filters": ["signal_quality >= 0.95"]
    }
  },
  "arguments": {
    "operation": "linear_regression",
    "target": "latency_ms",
    "features": ["snr", "jitter", "packet_loss"],
    "alpha": 0.05,
    "normalize": true
  },
  "request_id": "req-9f4e2f7a-1182-4c4d-b2e7-c17d2db8a5d1",
  "timeout_ms": 45000
}
```

### Example `ToolResult`

```json
{
  "status": "ok",
  "summary": "Linear regression completed on 18,204 samples. packet_loss and jitter are significant predictors of latency_ms.",
  "structured_output": {
    "model": "linear_regression",
    "sample_count": 18204,
    "r_squared": 0.78,
    "coefficients": {
      "intercept": 2.17,
      "snr": -0.09,
      "jitter": 0.61,
      "packet_loss": 1.44
    },
    "p_values": {
      "snr": 0.031,
      "jitter": 0.0002,
      "packet_loss": 0.00001
    }
  },
  "artifacts": [
    {
      "name": "regression_coefficients",
      "mime_type": "text/csv",
      "uri": "s3://example-bucket/results/req-9f4e2f7a/coefficients.csv",
      "sha256": "3f786850e387550fdab836ed7e6dc881de23001b"
    }
  ],
  "warnings": [],
  "errors": [],
  "confidence": 0.94
}
```

## 5) Validation rules

1. **Required fields**
   - `ToolManifest`: all top-level fields are required.
   - `ToolInvocation`: `tool_name`, `tool_version`, `capture_selection`, `arguments`, `request_id`, `timeout_ms` are required.
   - `ToolResult`: `status`, `summary`, `warnings`, `errors`, `confidence` are required; `structured_output` required when `status` is `ok|partial`.

2. **Type constraints**
   - Strings must be valid UTF-8.
   - Numeric fields marked integer must be whole numbers and non-negative unless stated otherwise.
   - `confidence` must satisfy `0.0 <= confidence <= 1.0`.
   - `status` must be exactly one of `ok`, `partial`, `error`.
   - `version` and `tool_version` must match regex: `^\\d+\\.\\d+\\.\\d+$`.

3. **Cross-field constraints**
   - `tool_name` and `tool_version` must identify an installed manifest entry.
   - `timeout_ms` must be clamped to `execution_constraints.max_timeout_ms`; if below minimum runtime threshold, return `error`.
   - `arguments` must validate against `input_schema`; unknown fields should be rejected unless `input_schema.additionalProperties=true`.
   - For `status=error`, `errors` must be non-empty; `structured_output` should be `{}` or omitted.
   - For `status=partial`, at least one warning is expected explaining incompleteness.

4. **Behavior on missing/invalid arguments**
   - Missing required argument: return `status=error`, error code `MISSING_ARGUMENT`, and `errors[].field` naming missing key.
   - Type mismatch: return `status=error`, error code `INVALID_TYPE` with expected vs actual type in message.
   - Constraint violation (range/enum/pattern): return `status=error`, error code `INVALID_VALUE`.
   - Unknown argument when disallowed: return `status=error`, error code `UNKNOWN_ARGUMENT`.
   - No partial execution should occur after schema validation failure.

## 6) Compatibility policy

This contract uses semantic versioning: `major.minor.patch`.

- **Patch (`x.y.Z`)**: Backward-compatible fixes only (wording clarifications, non-breaking constraints, bug fixes).
- **Minor (`x.Y.z`)**: Backward-compatible additions (new optional fields, additional enum values that old clients may safely ignore, expanded capabilities).
- **Major (`X.y.z`)**: Breaking changes (field removals/renames, required-field additions, type changes, stricter validation that breaks previously valid payloads).

Backward compatibility expectations:

- Clients must send explicit `tool_version` and should pin major version.
- Runtimes should support at least one previous minor version within the same major where feasible.
- New optional response fields must not break old clients; old clients should ignore unknown fields.
- Removing or repurposing existing fields requires a major version increment.
- Deprecations should be announced in advance and preserved for at least one minor release before removal in the next major release.
