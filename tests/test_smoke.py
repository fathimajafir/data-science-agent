"""Fast end-to-end smoke tests that do not need model weights.

They run against the sklearn fallback so CI (or a machine with a blocked
Hugging Face endpoint) still exercises every code path.
"""

from __future__ import annotations

import os

os.environ.setdefault("TABPFN_BACKEND", "fallback")

from tabular_agent import backends, cache_bench, core, data, forecasting, interpret  # noqa: E402
from tabular_agent.agent import ToolRegistry, run_agent  # noqa: E402


def test_backend_resolution():
    assert backends.resolve_backend() in {"api", "local", "fallback"}
    assert backends.backend_label()


def test_titanic_schema_and_classification():
    df = data.load_titanic()
    assert {"survived", "sex", "pclass"} <= set(df.columns)
    res = core.run_classification(df, "survived", data.TITANIC_FEATURES)
    assert 0.5 <= res["metrics"]["accuracy"] <= 1.0
    assert res["n_features"] == len(data.TITANIC_FEATURES)


def test_regression_and_baseline():
    df = data.load_california(max_rows=1200)
    res = core.run_regression(df, "MedHouseVal")
    base = core.run_baseline_regression(df, "MedHouseVal")
    assert res["metrics"]["r2"] > 0
    assert base["metrics"]["r2"] > 0


def test_kv_cache_benchmark_fields():
    df = data.load_titanic()
    res = cache_bench.benchmark(
        df, "survived", data.TITANIC_FEATURES, "classification", n_batches=2
    )
    assert res["without"]["total_predict_seconds"] > 0
    assert res["with"]["total_predict_seconds"] > 0
    assert res["speedup_predict"] > 0


def test_interpretation_shapes():
    df = data.load_titanic()
    res = interpret.explain(
        df, "survived", data.TITANIC_FEATURES, task="classification", n_rows=3
    )
    shap_df = res["shap_values"]
    assert shap_df.shape == (3, len(data.TITANIC_FEATURES))
    assert len(res["global_importance"]) == len(data.TITANIC_FEATURES)
    assert res["row_explanations"][0]["top_drivers"]


def test_forecast_features_do_not_leak():
    series = data.make_demand_series(days=420)
    table = forecasting.build_features(series).dropna().reset_index(drop=True)
    for i in (40, 90):
        assert table.loc[i, "lag_1"] == table.loc[i - 1, "units"]
        assert table.loc[i, "lag_7"] == table.loc[i - 7, "units"]
        window = table.loc[i - 7 : i - 1, "units"].mean()
        assert abs(table.loc[i, "roll_mean_7"] - window) < 1e-9


def test_forecast_metrics_range():
    series = data.make_demand_series(days=500)
    bt = forecasting.backtest(series, horizon=7, origins=1)
    assert bt["metrics"]["mae"] > 0
    assert 0 <= bt["metrics"]["coverage_80"] <= 1
    fut = forecasting.future_forecast(series, horizon=7)
    assert len(fut) == 7
    assert (fut["forecast"] > 0).all()


def test_agent_scripted_planner_answers_with_numbers():
    out = run_agent("How good is the model on the housing data?")
    assert out.answer
    assert out.trace and out.trace[0]["tool"] == "train_and_evaluate"
    assert "r2" in out.answer or "rmse" in out.answer


def test_registry_unknown_tool_is_handled():
    reg = ToolRegistry()
    res = reg.call("nope", {})
    assert "error" in res
