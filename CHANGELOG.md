# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-11-12

### Added

- TabPFN-backed predictions for classification (Titanic) and regression
  (California housing), with gradient-boosting baselines for comparison.
- KV cache benchmark (`fit_mode="fit_with_cache"` vs not) over repeated
  prediction batches.
- Interpretation layer: global feature importance and per-row drivers via
  shapiq SHAP (local weights), permutation SHAP (hosted API) or occlusion
  (offline fallback).
- Forecasting by reframing a daily series as tabular regression: lag/rolling/
  calendar features, known-future covariates, recursive multi-step forecast
  with quantile intervals, rolling-origin backtest.
- Tool-calling data science agent: five tools over a JSON-schema registry,
  OpenAI-compatible LLM planner plus a deterministic offline planner, answer
  grounding limited to tool results.
- `tabular-agent` CLI: `run`, `ask`, `doctor`, `datasets`, `serve`.
- Optional FastAPI HTTP service (`/health`, `/tools`, `/ask`).
- Backend resolution (`api` / `local` / `fallback`) with retries and backoff
  for transient API failures, thread-safe tool cache, diagnostics.
- Offline scikit-learn fallback backend so tests and CI need no keys.
- Test suite (25 tests), ruff lint config, GitHub Actions CI, Dockerfile,
  reports pipeline writing markdown + figures + JSON results.
