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

## 6) Planner/Executor Lifecycle

To keep planning behavior deterministic, orchestrators should run the following fixed lifecycle for every user request:

1. **Discover Tools**: Load all available `ToolManifest` entries (and versions) into planner context.
2. **Plan**: LLM emits an ordered list of `ToolInvocation` objects.
3. **Validate**: Runtime performs JSON-schema validation plus semantic checks (capture availability, time bounds, data sufficiency).
4. **Execute**: Runtime executes only validated invocations.
5. **Summarize**: Runtime/LLM composes user-facing summary from `ToolResult.summary` and `structured_output`.
6. **Repair**: If validation or execution fails, planner regenerates only failed invocations using normalized error payloads.

## 7) Concrete tool spec: `bluetooth_address_analyzer`

`bluetooth_address_analyzer` infers BLE address types and probabilistic identity linkage under configurable temporal assumptions. The tool is intended for investigative analysis, not deterministic attribution.

### 7.1 Input schema

```json
{
  "type": "object",
  "properties": {
    "capture_selection": {
      "type": "object",
      "properties": {
        "capture_ids": {
          "type": "array",
          "items": { "type": "string" },
          "minItems": 1
        },
        "time_window": {
          "type": "object",
          "properties": {
            "start_ms": { "type": "integer", "minimum": 0 },
            "end_ms": { "type": "integer", "minimum": 0 }
          },
          "required": ["start_ms", "end_ms"]
        },
        "filter_expression": {
          "type": "string",
          "description": "Optional implementation-specific predicate, e.g., \"adv_type IN ('ADV_IND','ADV_NONCONN_IND')\"."
        }
      },
      "required": ["capture_ids", "time_window"]
    },
    "address_columns": {
      "type": "object",
      "properties": {
        "advertiser_addr": { "type": "string" },
        "initiator_addr": { "type": "string" },
        "scanner_addr": { "type": "string" }
      },
      "required": ["advertiser_addr", "initiator_addr"],
      "description": "Column names from normalized packet records; scanner_addr is optional."
    },
    "analysis_mode": {
      "type": "string",
      "enum": ["classification", "linkage", "both"]
    },
    "linkage_window_s": {
      "type": "number",
      "minimum": 1,
      "description": "Maximum seconds between related observations when estimating linkage."
    },
    "min_observation_count": {
      "type": "integer",
      "minimum": 1,
      "description": "Minimum per-address observations required before inference."
    },
    "rpa_rotation_model": {
      "type": "string",
      "enum": ["spec_default", "fast_rotation", "slow_rotation", "learned"],
      "description": "Assumption set for random private address rotation cadence."
    }
  },
  "required": [
    "capture_selection",
    "address_columns",
    "analysis_mode",
    "linkage_window_s",
    "min_observation_count",
    "rpa_rotation_model"
  ]
}
```

### 7.2 Output schema

```json
{
  "type": "object",
  "properties": {
    "address_classification": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "address": { "type": "string" },
          "type": {
            "type": "string",
            "enum": ["public", "random_static", "rpa", "nrpa", "unknown"]
          },
          "confidence": { "type": "number", "minimum": 0, "maximum": 1 },
          "evidence": {
            "type": "array",
            "items": { "type": "string" }
          }
        },
        "required": ["address", "type", "confidence", "evidence"]
      }
    },
    "probabilistic_links": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "cluster_id": { "type": "string" },
          "addresses": {
            "type": "array",
            "items": { "type": "string" },
            "minItems": 2
          },
          "pairwise_probability_matrix": {
            "type": "array",
            "items": {
              "type": "array",
              "items": { "type": "number", "minimum": 0, "maximum": 1 }
            }
          },
          "rationale": {
            "type": "array",
            "items": { "type": "string" }
          }
        },
        "required": ["cluster_id", "addresses", "pairwise_probability_matrix", "rationale"]
      }
    },
    "timeline_segments": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "address": { "type": "string" },
          "start_ts": { "type": "string", "description": "RFC3339 timestamp" },
          "end_ts": { "type": "string", "description": "RFC3339 timestamp" },
          "activity_score": { "type": "number", "minimum": 0 }
        },
        "required": ["address", "start_ts", "end_ts", "activity_score"]
      }
    }
  },
  "required": ["address_classification", "probabilistic_links", "timeline_segments"]
}
```

#### Field types + interpretability notes (LLM-facing)

| Field | Type | Interpretability note |
|---|---|---|
| `address_classification[].address` | `string` | Canonical observed BLE address string. Treat as observation key, not a stable device identity. |
| `address_classification[].type` | `enum` | Predicted address class from header-bit patterns and temporal behavior; may shift across sessions. |
| `address_classification[].confidence` | `number [0,1]` | Posterior confidence for the class label, not a legal certainty score. |
| `address_classification[].evidence` | `string[]` | Human-readable features supporting the classification; useful for LLM explanation grounding. |
| `probabilistic_links[].cluster_id` | `string` | Runtime-generated identifier for one inferred identity cluster. |
| `probabilistic_links[].addresses` | `string[]` | Addresses likely belonging to the same emitter under the configured model assumptions. |
| `probabilistic_links[].pairwise_probability_matrix` | `number[][]` | Square matrix aligned to `addresses[]`; each cell is probability two addresses are linked. |
| `probabilistic_links[].rationale` | `string[]` | Explanatory factors (co-occurrence, rotation interval fit, RSSI continuity, etc.) for link probabilities. |
| `timeline_segments[].address` | `string` | Address that was active in the segment. |
| `timeline_segments[].start_ts` | `string (RFC3339)` | Segment start time in absolute wall clock. |
| `timeline_segments[].end_ts` | `string (RFC3339)` | Segment end time in absolute wall clock. |
| `timeline_segments[].activity_score` | `number >= 0` | Relative activity intensity in the segment (packet density/weighted duty); compare within the same run only. |

### 7.3 Statistical interpretation

- **Confidence semantics**: `confidence` and linkage probabilities are model-based posterior-like scores conditioned on observed packets plus `rpa_rotation_model` assumptions; they are not frequentist p-values and should not be interpreted as proof.
- **False positives and caveats**:
  - Similar traffic profiles from nearby devices can inflate linkage probabilities.
  - Capture blind spots can force the model to connect addresses across unobserved intervals.
  - Aggressive `linkage_window_s` values increase recall but may raise false-positive cluster merges.
  - Use `evidence` and `rationale` fields to present uncertainty and alternate explanations in LLM responses.

### 7.4 Example invocation and output

#### Example invocation

```json
{
  "tool_name": "bluetooth_address_analyzer",
  "tool_version": "1.0.0",
  "capture_selection": {
    "capture_ids": ["lab_floor_2026_04_03_a", "lab_floor_2026_04_03_b"],
    "time_window": {
      "start_ms": 1712131200000,
      "end_ms": 1712133000000
    },
    "filter_expression": "channel IN (37,38,39) AND rssi > -92"
  },
  "arguments": {
    "address_columns": {
      "advertiser_addr": "advertiser_addr",
      "initiator_addr": "initiator_addr",
      "scanner_addr": "scanner_addr"
    },
    "analysis_mode": "both",
    "linkage_window_s": 180,
    "min_observation_count": 25,
    "rpa_rotation_model": "spec_default"
  },
  "request_id": "req-bt-71ac9d11",
  "timeout_ms": 60000
}
```

#### Example output

```json
{
  "status": "ok",
  "summary": "Analyzed 4,218 BLE observations; identified one high-confidence 3-address RPA cluster and classified 41 standalone addresses.",
  "structured_output": {
    "address_classification": [
      {
        "address": "7A:4C:21:9D:10:EF",
        "type": "rpa",
        "confidence": 0.93,
        "evidence": [
          "Resolvable private bit pattern detected",
          "Rotation cadence ~14.7 minutes",
          "Stable manufacturer data payload signature"
        ]
      },
      {
        "address": "4E:91:B2:6A:77:03",
        "type": "public",
        "confidence": 0.98,
        "evidence": [
          "Public address bit pattern",
          "No observed rotation across full window"
        ]
      }
    ],
    "probabilistic_links": [
      {
        "cluster_id": "cluster_rpa_004",
        "addresses": [
          "7A:4C:21:9D:10:EF",
          "61:D2:88:3C:A4:19",
          "5F:09:2A:7E:CD:44"
        ],
        "pairwise_probability_matrix": [
          [1.0, 0.91, 0.88],
          [0.91, 1.0, 0.86],
          [0.88, 0.86, 1.0]
        ],
        "rationale": [
          "Sequential non-overlapping activity windows consistent with RPA rotation",
          "Matched GATT service UUID set across all three addresses",
          "RSSI trajectory continuity at receiver rx-north-2"
        ]
      }
    ],
    "timeline_segments": [
      {
        "address": "7A:4C:21:9D:10:EF",
        "start_ts": "2026-04-03T08:01:13Z",
        "end_ts": "2026-04-03T08:16:02Z",
        "activity_score": 0.82
      },
      {
        "address": "61:D2:88:3C:A4:19",
        "start_ts": "2026-04-03T08:16:05Z",
        "end_ts": "2026-04-03T08:30:40Z",
        "activity_score": 0.79
      }
    ]
  },
  "warnings": [],
  "errors": [],
  "confidence": 0.9
}
```

### 7.5 Failure modes

- **Sparse observations**: If an address has fewer than `min_observation_count` events, classification may return `unknown` and linkage may be omitted or downgraded (`status=partial`).
- **Clock gaps**: Sensor clock drift or missing intervals can fragment timeline segments and create artificial handoff patterns.
- **Overlapping devices with similar behavior**: Devices sharing payload templates and similar motion/RSSI profiles can be merged into one cluster with overstated pairwise probabilities.
### Standard error payloads for planner repair

Executor validation and execution failures should be returned in `ToolResult.errors[]` using the canonical shape shown below:

```json
{
  "code": "MISSING_REQUIRED_ARGUMENT",
  "message": "arguments.target is required",
  "field": "arguments.target"
}
```

Use the following required codes and recovery behavior:

| Error code | Field that caused failure | How executor reports it | What planner should regenerate |
|---|---|---|---|
| `MISSING_REQUIRED_ARGUMENT` | Missing required key under `arguments.*` (or required invocation envelope field) | `status=error`; include one `errors[]` item per missing key with `field` path and required-key message. | Re-emit the same invocation with all required fields populated from tool schema and task intent; keep `request_id` new per retry. |
| `INVALID_CAPTURE_SELECTION` | `capture_selection.capture_id`, `capture_selection.selectors.channels[]`, or filter selectors that reference unknown data | `status=error`; set `field` to failing selector path and message naming unknown capture/channel/filter. | Regenerate `capture_selection` using discovered valid capture IDs/channels; preserve operation arguments if still compatible. |
| `UNSUPPORTED_TIME_RANGE` | `capture_selection.selectors.time_range.start_ms` / `end_ms` outside capture bounds or invalid window (`start_ms > end_ms`) | `status=error`; set `field` to `capture_selection.selectors.time_range` and include supported range in message. | Re-plan with a bounded time range that lies within manifest/runtime limits and requested analysis window. |
| `INSUFFICIENT_DATA` | Effective dataset after selection is too small for requested operation (for example sample count below statistical minimum) | `status=error` or `status=partial` (if degraded mode exists); set `field` to operation-specific input (often `arguments.operation` or selection path) and include observed sample count. | Regenerate by broadening selection, switching to a compatible method, or adding fallback operation that can run on available samples. |

### End-to-end example: invalid plan -> corrected plan

The first planner attempt fails validation because `target` is missing and the selected time range exceeds capture bounds.

#### Invalid plan output

```json
[
  {
    "tool_name": "statistical_regression_tool",
    "tool_version": "1.2.0",
    "capture_selection": {
      "capture_id": "cap_2026_03_14_a",
      "selectors": {
        "time_range": {
          "start_ms": 0,
          "end_ms": 999999
        },
        "channels": ["ch1", "ch2"],
        "filters": []
      }
    },
    "arguments": {
      "operation": "linear_regression",
      "features": ["snr", "jitter", "packet_loss"]
    },
    "request_id": "req-invalid-001",
    "timeout_ms": 45000
  }
]
```

#### Executor validation response

```json
{
  "status": "error",
  "summary": "Invocation failed validation.",
  "warnings": [],
  "errors": [
    {
      "code": "MISSING_REQUIRED_ARGUMENT",
      "message": "arguments.target is required",
      "field": "arguments.target"
    },
    {
      "code": "UNSUPPORTED_TIME_RANGE",
      "message": "Requested 0-999999ms but capture cap_2026_03_14_a supports 0-120000ms",
      "field": "capture_selection.selectors.time_range"
    }
  ],
  "confidence": 0.0
}
```

#### Corrected planner output (repair step)

```json
[
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
        "filters": []
      }
    },
    "arguments": {
      "operation": "linear_regression",
      "target": "latency_ms",
      "features": ["snr", "jitter", "packet_loss"]
    },
    "request_id": "req-repair-002",
    "timeout_ms": 45000
  }
]
```

## 7) Compatibility policy

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
