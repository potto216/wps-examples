# Shared Python library (`lib/`)

Use `lib/` for reusable Python code shared by notebooks, scripts, and future tooling in this repository.

## Is `lib` a good name?

Yes. `lib` is a common and understandable name for repository-local shared code.

If this repository grows into a larger Python package, a stronger long-term pattern is to move shared code to a namespaced package under `src/` (for example `src/wps_examples_common/`). For the current script-and-notebook-heavy layout, `lib/` is an appropriate and low-friction choice.

## Is this location good?

Yes. Keeping `lib/` at the repository root makes it easy for code in sibling directories (`analysis/`, `scripts/`, `tests/`, notebooks) to import shared helpers once `PYTHONPATH` is configured to include the repo root.

## Structure recommendations

As you add functions and classes (file conversion, load/save, pandas math/analysis), prefer a package layout instead of putting everything in `lib/__init__.py`:

- `lib/io/` for load/save and format conversion helpers
- `lib/analysis/` for dataframe transformations and statistics
- `lib/util/` for process/system helpers
- `lib/types.py` or `lib/models.py` for shared dataclasses/types

Add module-level docstrings and avoid side effects at import time.

## Dependencies (`requirements.txt`)

Default recommendation: keep one root `requirements.txt` as the source of truth for repo-wide dependencies.

Use a dedicated `lib/requirements.txt` only if `lib/` is intentionally consumed as an independently installable component with a different dependency lifecycle.

A practical middle ground:

- Keep `requirements.txt` at the root.
- Optionally split into `requirements/base.txt` and `requirements/dev.txt` if dependency count grows.
- If/when `lib/` becomes a standalone package, move to `pyproject.toml` with optional dependency groups.

## Import stability recommendation

To reduce import friction over time, consider introducing a top-level namespace package (for example `lib/wps_common/...`) so imports look like:

```python
from lib.wps_common.analysis.packet_math import summarize_packets
```

This avoids ambiguous generic module names and scales better as shared code grows.
