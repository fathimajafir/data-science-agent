# Contributing

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,serve]"
cp .env.example .env
```

Tests and lint run offline on the fallback backend — no keys required:

```bash
ruff check .
python -m pytest -q
```

Please add tests for anything you change; CI (`.github/workflows/ci.yml`)
runs the suite on Python 3.10–3.12 and must stay green.

## Conventions

- Library code logs through `logging.getLogger(__name__)`; only the CLI prints.
- Every model call goes through `backends.make_estimator` so all three
  backends (api / local / fallback) keep working.
- Anything an LLM sees must be JSON-serialisable and bounded in size.
- Secrets belong in `.env` (git-ignored), never in code or reports.

## Adding a dataset

1. Add a loader in `src/tabular_agent/data.py` (real data with a synthetic
   offline fallback, cached under `data/raw/`).
2. Register it in `DATASETS` in `src/tabular_agent/agent/tools.py`
   (task, target, features, description) — the agent can use it immediately.
3. Extend the tool enum in the tool schemas where relevant, and add a test.

## Adding a tool

1. Implement it as a plain function returning a compact dict
   (`src/tabular_agent/agent/tools.py`).
2. Register name + description + JSON schema in `_T`.
3. Keep results small: the planner only sees what the tool returns.

## Reports

`tabular-agent run` regenerates `reports/` — do not commit generated reports;
commit code and let the pipeline produce artifacts.
