"""Tools the data science agent can call.

Each tool is a normal Python function plus a JSON schema. That is the whole
trick behind "giving a model a tool": the LLM never sees your data, it only sees
a short description of the schema and the compact JSON results.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from .. import backends, core, data, forecasting, interpret

# Guards the module-level caches so the HTTP service can serve concurrent
# requests safely: tool calls serialise instead of racing on fits.
_LOCK = threading.RLock()

DATASETS: dict[str, dict[str, Any]] = {
    "titanic": {
        "task": "classification",
        "target": "survived",
        "features": data.TITANIC_FEATURES,
        "description": "Passenger survival (binary). Mixed categorical/numeric with missing values.",
        "loader": lambda: data.load_titanic(),
    },
    "california": {
        "task": "regression",
        "target": "MedHouseVal",
        "features": None,
        "description": "Median house value of census blocks (regression, 8 numeric features).",
        "loader": lambda: data.load_california(),
    },
    "demand": {
        "task": "forecasting",
        "target": "units",
        "features": None,
        "description": "Daily store demand with promotions and holidays (forecasting).",
        "loader": lambda: data.make_demand_series(),
    },
}

_STATE: dict[str, Any] = {}


def get_dataset(name: str):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}. Choose from {list(DATASETS)}")
    with _LOCK:
        if ("df", name) not in _STATE:
            _STATE[("df", name)] = DATASETS[name]["loader"]()
        return _STATE[("df", name)]


def _fitted_model(name: str):
    """Fit once per process and reuse: later tool calls skip the fit cost."""
    spec = DATASETS[name]
    key = ("model", name)
    with _LOCK:
        if key not in _STATE:
            df = get_dataset(name)
            model = (
                core.build_classifier(fit_with_cache=True)
                if spec["task"] == "classification"
                else core.build_regressor(fit_with_cache=True)
            )
            X = df[spec["features"]] if spec["features"] else df.drop(columns=[spec["target"]])
            y = df[spec["target"]]
            model.fit(X, y)
            _STATE[key] = (model, X, y)
        return _STATE[key]


# --------------------------------------------------------------------------
# Tool implementations
# --------------------------------------------------------------------------


def inspect_dataset(dataset: str) -> dict:
    df = get_dataset(dataset)
    spec = DATASETS[dataset]
    summary = data.dataset_summary(df, spec["target"])
    target = df[spec["target"]]
    summary["problem_type"] = spec["task"]
    summary["description"] = spec["description"]
    summary["target_stats"] = (
        target.value_counts(normalize=True).round(3).to_dict()
        if spec["task"] == "classification"
        else {
            "mean": round(float(target.mean()), 3),
            "std": round(float(target.std()), 3),
            "min": round(float(target.min()), 3),
            "max": round(float(target.max()), 3),
        }
    )
    return summary


def train_and_evaluate(dataset: str) -> dict:
    spec = DATASETS[dataset]
    df = get_dataset(dataset)
    if spec["task"] == "classification":
        res = core.run_classification(df, spec["target"], spec["features"])
        baseline = core.run_baseline_classification(
            df, spec["target"], spec["features"]
        )
    elif spec["task"] == "regression":
        res = core.run_regression(df, spec["target"], spec["features"])
        baseline = core.run_baseline_regression(df, spec["target"], spec["features"])
    else:
        raise ValueError("Use the forecast tool for time series datasets")

    return {
        "dataset": dataset,
        "task": spec["task"],
        "backend": res["model"],
        "n_train": res["n_train"],
        "n_test": res["n_test"],
        "n_features": res["n_features"],
        "metrics": {k: round(v, 4) for k, v in res["metrics"].items()},
        "baseline_metrics": {k: round(v, 4) for k, v in baseline["metrics"].items()},
        "fit_seconds": res["fit_seconds"],
        "predict_seconds": res["predict_seconds"],
    }


def explain_model(dataset: str, n_rows: int = 5) -> dict:
    spec = DATASETS[dataset]
    if spec["task"] == "forecasting":
        raise ValueError("Explain a classification or regression dataset instead")
    model, _, _ = _fitted_model(dataset)
    df = get_dataset(dataset)
    res = interpret.explain(
        df,
        spec["target"],
        spec["features"],
        task=spec["task"],
        n_rows=n_rows,
        model=model,
    )
    return {
        "dataset": dataset,
        "method": res["method"],
        "top_features": [
            {
                "feature": f["feature"],
                "importance": round(f["importance"], 4),
                "direction": f["direction"],
            }
            for f in res["global_importance"][:6]
        ],
        "example_rows": [
            {
                "row_index": r["row_index"],
                "prediction": round(r["prediction"], 4),
                "top_drivers": [
                    {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}
                    for d in r["top_drivers"][:3]
                ],
            }
            for r in res["row_explanations"][:2]
        ],
    }


def forecast(dataset: str = "demand", horizon: int = 14) -> dict:
    df = get_dataset(dataset)
    bt = forecasting.backtest(df, horizon=horizon, origins=2)
    fut = forecasting.future_forecast(df, horizon=horizon)
    return {
        "dataset": dataset,
        "horizon": horizon,
        "backend": bt["model"],
        "n_features": bt["n_features"],
        "backtest_metrics": {k: round(v, 4) for k, v in bt["metrics"].items()},
        "next_days": [
            {
                "date": str(r.date.date()),
                "forecast": round(r.forecast, 1),
                "low_80": round(r.low_80, 1),
                "high_80": round(r.high_80, 1),
            }
            for r in fut.head(min(horizon, 7)).itertuples()
        ],
        "total_forecast_units": round(float(fut["forecast"].sum()), 1),
    }


def benchmark_kv_cache(dataset: str = "titanic", n_batches: int = 5) -> dict:
    """Time repeated predictions with and without the KV cache."""
    df = get_dataset(dataset)
    spec = DATASETS[dataset]
    if spec["task"] == "forecasting":
        raise ValueError("Choose titanic or california")
    from ..cache_bench import benchmark

    res = benchmark(df, spec["target"], spec["features"], spec["task"], n_batches)
    return {
        "dataset": dataset,
        "backend": res["backend"],
        "n_batches": n_batches,
        "fit_without_cache_s": res["without"]["fit_seconds"],
        "fit_with_cache_s": res["with"]["fit_seconds"],
        "predict_without_cache_s": res["without"]["total_predict_seconds"],
        "predict_with_cache_s": res["with"]["total_predict_seconds"],
        "speedup_predict": res["speedup_predict"],
        "speedup_total": res["speedup_total"],
    }


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

_T = {
    "inspect_dataset": (
        inspect_dataset,
        "Describe a dataset: rows, columns, dtypes, missing values, target summary.",
        {
            "type": "object",
            "properties": {
                "dataset": {"type": "string", "enum": list(DATASETS)},
            },
            "required": ["dataset"],
        },
    ),
    "train_and_evaluate": (
        train_and_evaluate,
        "Fit a TabPFN model on a classification or regression dataset, score it on a held-out split, "
        "and compare against a gradient-boosting baseline.",
        {
            "type": "object",
            "properties": {"dataset": {"type": "string", "enum": ["titanic", "california"]}},
            "required": ["dataset"],
        },
    ),
    "explain_model": (
        explain_model,
        "Explain a fitted model: global feature importance across the dataset plus the drivers "
        "of individual predictions.",
        {
            "type": "object",
            "properties": {
                "dataset": {"type": "string", "enum": ["titanic", "california"]},
                "n_rows": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["dataset"],
        },
    ),
    "forecast": (
        forecast,
        "Backtest and forecast daily demand. Returns error metrics, 80% interval coverage, "
        "the next days' predictions and the total forecast volume.",
        {
            "type": "object",
            "properties": {
                "dataset": {"type": "string", "enum": ["demand"]},
                "horizon": {"type": "integer", "minimum": 1, "maximum": 60},
            },
            "required": ["dataset"],
        },
    ),
    "benchmark_kv_cache": (
        benchmark_kv_cache,
        "Compare repeated predictions with and without TabPFN's KV cache (fit_with_cache).",
        {
            "type": "object",
            "properties": {
                "dataset": {"type": "string", "enum": ["titanic", "california"]},
                "n_batches": {"type": "integer", "minimum": 2, "maximum": 20},
            },
            "required": ["dataset"],
        },
    ),
}


class ToolRegistry:
    def __init__(self) -> None:
        self._tools = dict(_T)

    def schemas(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": params,
                },
            }
            for name, (_, desc, params) in self._tools.items()
        ]

    def call(self, name: str, args: dict | None = None) -> dict:
        if name not in self._tools:
            return {"error": f"Unknown tool {name!r}", "available": list(self._tools)}
        fn: Callable = self._tools[name][0]
        try:
            return fn(**(args or {}))
        except Exception as exc:  # surface the failure to the agent, do not crash
            return {"error": f"{type(exc).__name__}: {exc}"}

    def describe(self) -> list[dict]:
        return [
            {"name": n, "description": d, "parameters": p}
            for n, (_, d, p) in self._tools.items()
        ]


def backend_notice() -> str:
    if backends.resolve_backend() == "fallback":
        return (
            "TabPFN is not reachable (no TABPFN_TOKEN and no local weights): "
            "results come from the sklearn fallback."
        )
    return f"Backend: {backends.backend_label()}"
