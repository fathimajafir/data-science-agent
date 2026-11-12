# Data Science Agent — tabular foundation models (TabPFN)

[![CI](https://github.com/fathimajafir/data-science-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/fathimajafir/data-science-agent/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A production-ready data science agent that makes **classification, regression
and forecasting predictions on tabular data without training a model from
scratch**, using [TabPFN](https://github.com/PriorLabs/tabpfn) — a tabular
foundation model that learns your dataset in context, in a single forward pass.

Instead of training and tuning a separate model per dataset, the agent fits
TabPFN on raw DataFrames (categorical strings, missing values and all), speeds
up repeated predictions with a KV cache, explains individual predictions with
SHAP attributions, and answers natural-language questions by calling these
capabilities as tools.

## Highlights

- **Zero-training predictions** — classification and regression in one forward
  pass; raw pandas frames in, metrics out; gradient-boosting baseline included
  for comparison.
- **KV cache** — `fit_mode="fit_with_cache"` encodes the training context once;
  benchmarked with and without.
- **Interpretation** — global feature importance across a dataset plus the
  drivers behind individual predictions (shapiq SHAP locally, permutation SHAP
  via the API, occlusion offline).
- **Forecasting** — daily series reframed as tabular regression (lags, rolling
  stats, calendar terms, known-future covariates) with quantile intervals and
  rolling-origin backtests.
- **Agent** — tool-calling loop with a JSON-schema tool registry; bring any
  OpenAI-compatible LLM for planning, or use the built-in deterministic
  planner offline.
- **Ops surface** — `tabular-agent` CLI (`run`, `ask`, `doctor`, `datasets`,
  `serve`), optional FastAPI HTTP API, retries with backoff on transient API
  errors, thread-safe tool cache, Docker image, CI.

## Quickstart

```bash
git clone https://github.com/fathimajafir/data-science-agent.git
cd data-science-agent
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # add ,serve for the HTTP API
cp .env.example .env             # optional: add keys here
python -m pytest -q              # runs offline on the fallback backend
tabular-agent doctor             # check your setup
```

Get a free TabPFN API key at
[platform.priorlabs.ai/account/api-keys](https://platform.priorlabs.ai/account/api-keys)
and put it in `.env` as `TABPFN_TOKEN=...` — no GPU or model download needed.
With no key the project runs on a clearly-labelled scikit-learn fallback so
everything stays testable offline.

## The five lessons

| # | Command | What it covers |
|---|---|---|
| 1 | `tabular-agent run 1` | Titanic classification + California-housing regression vs baseline |
| 2 | `tabular-agent run 2` | KV cache benchmark: repeated predictions with vs without cache |
| 3 | `tabular-agent run 3` | SHAP interpretation: global importance + per-row drivers |
| 4 | `tabular-agent run 4` | Forecasting as tabular regression, backtest + 14-day forecast |
| 5 | `tabular-agent run 5` | The agent answering questions through tool calls |

`tabular-agent run` executes all five and writes markdown reports and figures
to `reports/` (start at `reports/summary.md`). The same steps exist as plain
scripts under `scripts/`.

## Using the agent

```bash
tabular-agent ask "How accurate is the survival model compared to boosting?"
tabular-agent ask "Why did the model predict survival for individual passengers?"
tabular-agent ask "Will demand be high next week?"
tabular-agent ask "Is the KV cache worth it for repeated predictions?"
```

The agent picks one of five tools — `inspect_dataset`, `train_and_evaluate`,
`explain_model`, `forecast`, `benchmark_kv_cache` — and answers with numbers
that come only from tool results. To plan with a real LLM, set `LLM_BASE_URL`,
`LLM_API_KEY` and `LLM_MODEL` in `.env` (any OpenAI-compatible endpoint);
without them a deterministic keyword planner is used.

### HTTP API

```bash
pip install -e ".[serve]"
tabular-agent serve --port 8000
```

| Endpoint | Description |
|---|---|
| `GET /health` | liveness, version, active backend |
| `GET /tools` | tool descriptions the agent can call |
| `POST /ask` | `{"question": "...", "max_steps": 6}` → answer + full trace |

```bash
curl -s localhost:8000/ask -H 'Content-Type: application/json' \
     -d '{"question":"What does the housing data look like?"}'
```

### Docker

```bash
docker build -t data-science-agent .
docker run --rm -p 8000:8000 -e TABPFN_TOKEN=$TABPFN_TOKEN data-science-agent
```

## Configuration

Environment variables, or a `.env` file at the repo root:

| Variable | Meaning | Default |
|---|---|---|
| `TABPFN_BACKEND` | `auto` \| `api` \| `local` \| `fallback` | `auto` |
| `TABPFN_TOKEN` | Prior Labs API key (enables the `api` backend) | — |
| `TABPFN_MODEL` | local checkpoint: `v3.5`, `v3.5-fast`, `v3`, `v2.5`, `v2` | `v3.5-fast` |
| `TABPFN_N_ESTIMATORS` | ensemble size (local backend) | `8` |
| `TABPFN_DEVICE` | `auto` \| `cpu` \| `mps` \| `cuda` | `auto` |
| `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` | OpenAI-compatible planner for the agent | — |

### Backends

`auto` resolves in this order:

1. **api** — hosted TabPFN-3.5-Plus through `tabpfn-client`. No GPU needed;
   requests are retried with exponential backoff on transient failures.
2. **local** — the `tabpfn` package with weights on this machine (accept the
   license at [huggingface.co/Prior-Labs](https://huggingface.co/Prior-Labs)
   first; GPU recommended past ~1k rows).
3. **fallback** — scikit-learn gradient boosting behind the same interface, so
   CI and offline development work. Results are always labelled
   `sklearn-fallback` and never presented as TabPFN output.

## Datasets

| Name | Type | Task | Notes |
|---|---|---|---|
| `titanic` | real (seaborn-data) | binary classification | raw strings + missing values, no preprocessing |
| `california` | real (sklearn) | regression | median house value, subsampled to 6k rows |
| `demand` | synthetic, known ground truth | forecasting | trend + weekly/annual seasonality + promo/holiday lifts |

Loaders degrade to schema-compatible synthetic clones when the network is
unavailable, so nothing depends on a download at run time.

## Project layout

```
src/tabular_agent/
  cli.py           tabular-agent command (run/ask/doctor/datasets/serve)
  serve.py         optional FastAPI service
  backends.py      api/local/fallback resolution, retries, diagnostics
  data.py          dataset loaders + schema summaries
  core.py          fit/predict/evaluate wrappers, metrics, baselines
  cache_bench.py   KV cache benchmark
  interpret.py     SHAP explanations (shapiq / API / occlusion)
  forecasting.py   series -> tabular features, recursive backtest + forecast
  agent/           tool registry, planners (LLM + scripted), agent loop
  pipeline.py      the five lessons as report-writing steps
scripts/           one runner per lesson
tests/             25 tests, run offline on the fallback backend
reports/           generated reports + figures (not committed)
```

## Development

```bash
pip install -e ".[dev]"
ruff check .
python -m pytest -q
```

CI runs ruff + the full test suite on Python 3.10–3.12
(`TABPFN_BACKEND=fallback`, no keys needed). See
[CONTRIBUTING.md](CONTRIBUTING.md) for the workflow and for how to add a
dataset or a tool.

## Attribution & licensing

The code in this repository is released under the [MIT License](LICENSE).
TabPFN itself is developed by [Prior Labs](https://priorlabs.ai); the TabPFN
model weights are subject to their own licenses and usage terms, and the
hosted API is metered — see [docs.priorlabs.ai](https://docs.priorlabs.ai).
The project structure follows the ideas taught in the DeepLearning.AI short
course *“Tabular Foundation Models for Data Science Agents”* (with Prior Labs).
