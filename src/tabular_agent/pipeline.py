"""Runs the five lessons and writes the report artifacts under reports/."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from . import agent as agent_mod
from . import backends, cache_bench, core, data, forecasting, interpret
from .config import FIGURES_DIR, REPORTS_DIR


def _md_table(rows: list[dict]) -> str:
    if not rows:
        return "_(no rows)_\n"
    cols = list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append(
            "| " + " | ".join(str(r.get(c, "")) for c in cols) + " |"
        )
    return "\n".join(lines) + "\n"


def _write(name: str, text: str) -> str:
    path = REPORTS_DIR / name
    path.write_text(text)
    return str(path)


def _save_json(key: str, value: Any) -> None:
    path = REPORTS_DIR / "results.json"
    blob = json.loads(path.read_text()) if path.exists() else {}
    blob[key] = json.loads(json.dumps(value, default=str))
    path.write_text(json.dumps(blob, indent=2))


# --------------------------------------------------------------------------


def step1_predictions() -> dict:
    out: dict[str, Any] = {}

    titanic = data.load_titanic()
    clf = core.run_classification(titanic, "survived", data.TITANIC_FEATURES)
    base_clf = core.run_baseline_classification(
        titanic, "survived", data.TITANIC_FEATURES
    )

    cal = data.load_california()
    reg = core.run_regression(cal, "MedHouseVal", None)
    base_reg = core.run_baseline_regression(cal, "MedHouseVal", None)

    out["classification"] = _slim(clf)
    out["classification_baseline"] = _slim(base_clf)
    out["regression"] = _slim(reg)
    out["regression_baseline"] = _slim(base_reg)

    # figure: predicted vs actual
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = pd.Series(reg["y_test"]).to_numpy(dtype=float)
    p = pd.Series(reg["preds"]).to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=150)
    ax.scatter(y, p, s=7, alpha=0.45, color="#2b6cb0", edgecolors="none")
    lim = [min(y.min(), p.min()), max(y.max(), p.max())]
    ax.plot(lim, lim, "--", color="#c53030", linewidth=1, label="perfect")
    ax.set_xlabel("actual median house value")
    ax.set_ylabel("predicted")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"California housing, TabPFN R2={reg['metrics']['r2']:.3f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "regression_scatter.png")
    plt.close(fig)

    rows = [
        {
            "dataset": "titanic (classification)",
            "model": clf["model"],
            "accuracy": round(clf["metrics"]["accuracy"], 3),
            "f1_macro": round(clf["metrics"]["f1_macro"], 3),
            "roc_auc": round(clf["metrics"]["roc_auc"], 3),
            "fit_s": clf["fit_seconds"],
        },
        {
            "dataset": "titanic (classification)",
            "model": base_clf["model"],
            "accuracy": round(base_clf["metrics"]["accuracy"], 3),
            "f1_macro": round(base_clf["metrics"]["f1_macro"], 3),
            "roc_auc": round(base_clf["metrics"]["roc_auc"], 3),
            "fit_s": base_clf["fit_seconds"],
        },
    ]
    rows2 = [
        {
            "dataset": "california (regression)",
            "model": reg["model"],
            "rmse": round(reg["metrics"]["rmse"], 3),
            "mae": round(reg["metrics"]["mae"], 3),
            "r2": round(reg["metrics"]["r2"], 3),
            "fit_s": reg["fit_seconds"],
        },
        {
            "dataset": "california (regression)",
            "model": base_reg["model"],
            "rmse": round(base_reg["metrics"]["rmse"], 3),
            "mae": round(base_reg["metrics"]["mae"], 3),
            "r2": round(base_reg["metrics"]["r2"], 3),
            "fit_s": base_reg["fit_seconds"],
        },
    ]

    md = f"""# Lesson 1 - Classification and regression in one forward pass

Backend: **{backends.backend_label()}**. Titanic: {len(titanic)} rows, {len(data.TITANIC_FEATURES)} features, raw strings and NaNs passed straight in.
California housing: {len(cal)} rows, 8 numeric features.

## Binary classification (Titanic survival)

{_md_table(rows)}

## Regression (California median house value)

{_md_table(rows2)}

![predicted vs actual](../reports/figures/regression_scatter.png)

No encoding, imputation or scaling was applied: TabPFN handles categorical
strings, missing values and outliers internally.
"""
    _save_json("predictions", out)
    _write("01-predictions.md", md)
    return out


def _slim(res: dict) -> dict:
    return {
        "model": res["model"],
        "task": res["task"],
        "n_train": res["n_train"],
        "n_test": res["n_test"],
        "n_features": res["n_features"],
        "fit_seconds": res["fit_seconds"],
        "predict_seconds": res["predict_seconds"],
        "metrics": {k: round(v, 4) for k, v in res["metrics"].items()},
    }


def step2_kv_cache() -> dict:
    titanic = data.load_titanic()
    cal = data.load_california()

    a = cache_bench.benchmark(
        titanic, "survived", data.TITANIC_FEATURES, "classification", n_batches=5
    )
    b = cache_bench.benchmark(cal, "MedHouseVal", None, "regression", n_batches=5)

    rows = []
    for name, r in (("titanic", a), ("california", b)):
        rows.append(
            {
                "dataset": name,
                "fit no cache (s)": r["without"]["fit_seconds"],
                "fit with cache (s)": r["with"]["fit_seconds"],
                "5x predict no cache (s)": r["without"]["total_predict_seconds"],
                "5x predict with cache (s)": r["with"]["total_predict_seconds"],
                "predict speedup": f"{r['speedup_predict']}x",
                "end-to-end speedup": f"{r['speedup_total']}x",
            }
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=150)
    labels = ["titanic", "california"]
    without = [a["without"]["total_predict_seconds"], b["without"]["total_predict_seconds"]]
    withc = [a["with"]["total_predict_seconds"], b["with"]["total_predict_seconds"]]
    import numpy as np

    x = np.arange(len(labels))
    ax.bar(x - 0.18, without, 0.36, label="no cache", color="#cbd5e0")
    ax.bar(x + 0.18, withc, 0.36, label="fit_with_cache", color="#2b6cb0")
    ax.set_xticks(x, labels)
    ax.set_ylabel("seconds for 5 prediction batches")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("KV cache: training context is encoded once", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "kv_cache.png")
    plt.close(fig)

    md = f"""# Lesson 2 - KV cache

Backend: **{backends.backend_label()}**. Each experiment fits once, then predicts on
{a['n_batches']} batches of {a['batch_size']} rows.

{_md_table(rows)}

![kv cache](../reports/figures/kv_cache.png)

`fit_mode="fit_with_cache"` moves the cost of encoding the training set from
every `predict()` call into `fit()`. Break-even is one fit and a handful of
predicts - exactly the access pattern of an explanation loop or a forecasting
loop, which is why lessons 3 and 4 use it.
"""
    _save_json("kv_cache", {"titanic": a, "california": b})
    _write("02-kv-cache.md", md)
    return {"titanic": a, "california": b}


def step3_interpretation() -> dict:
    titanic = data.load_titanic()
    res = interpret.explain(
        titanic, "survived", data.TITANIC_FEATURES, task="classification", n_rows=6
    )
    interpret.plot_global_importance(res, FIGURES_DIR / "global_importance.png")

    rows = [
        {
            "feature": f["feature"],
            "mean_abs_attribution": round(f["importance"], 4),
            "direction": f["direction"],
        }
        for f in res["global_importance"][:8]
    ]
    local = []
    for r in res["row_explanations"][:3]:
        drivers = "; ".join(
            f"{d['feature']}={d['value']} ({d['shap']:+.3f})" for d in r["top_drivers"][:3]
        )
        local.append(
            {
                "row": r["row_index"],
                "predicted survival": round(r["prediction"], 3),
                "top drivers": drivers,
            }
        )

    md = f"""# Lesson 3 - Interpretation

Backend: **{backends.backend_label()}**. Method: **{res['method']}**.

## Global: what moves the model across the dataset

{_md_table(rows)}

![importance](../reports/figures/global_importance.png)

## Local: what drove individual predictions

{_md_table(local)}

Local attributions are what make a prediction defensible to a human reviewer:
global rankings say "sex matters", a local explanation says "this passenger was
pushed down 0.21 by travelling third class".
"""
    _save_json(
        "interpretation",
        {
            "method": res["method"],
            "global_importance": res["global_importance"],
            "rows": res["row_explanations"],
        },
    )
    _write("03-interpretation.md", md)
    return res


def step4_forecasting() -> dict:
    series = data.make_demand_series()
    bt = forecasting.backtest(series, horizon=14, origins=3)
    fut = forecasting.future_forecast(series, horizon=14)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=150)
    tail = series.tail(120)
    ax.plot(tail["date"], tail["units"], color="#2d3748", linewidth=1, label="actual")
    last = bt["runs"][-1]["forecast"]
    ax.plot(last["date"], last["y_pred"], color="#2b6cb0", linewidth=1.4, label="backtest forecast")
    ax.fill_between(last["date"], last["y_low"], last["y_high"], color="#2b6cb0", alpha=0.18, label="80% interval")
    ax.plot(fut["date"], fut["forecast"], color="#c53030", linewidth=1.4, label="future forecast")
    ax.fill_between(fut["date"], fut["low_80"], fut["high_80"], color="#c53030", alpha=0.15)
    ax.set_title("Daily demand: backtest and 14-day forecast", fontsize=9)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "forecast.png")
    plt.close(fig)

    runs = [
        {
            "origin": r["origin"],
            "MAE": round(r["mae"], 2),
            "MAPE": f"{100 * r['mape']:.2f}%",
            "coverage 80%": f"{100 * r['coverage_80']:.0f}%",
        }
        for r in bt["runs"]
    ]
    metrics = {k: round(v, 4) for k, v in bt["metrics"].items()}

    md = f"""# Lesson 4 - Forecasting without a forecasting model

Backend: **{backends.backend_label()}**. The series is reframed as a table: lags
{forecasting.LAGS}, rolling means/stds {forecasting.ROLLING}, calendar terms, plus
the known-future covariates `promo` and `holiday`. A tabular regressor then
predicts the next day, and the prediction is fed back in as history.

{bt['n_features']} features, {bt['n_train_rows']} training rows, recursive {bt['horizon']}-day horizon.

## Rolling-origin backtest

{_md_table(runs)}

Mean metrics: {metrics}

`coverage_80` should sit near 0.80; it is the honest test of the uncertainty
bands, not just the point forecast.

![forecast](../reports/figures/forecast.png)
"""
    _save_json("forecasting", {"metrics": metrics, "future": fut.assign(date=fut["date"].astype(str)).to_dict("records")})
    _write("04-forecasting.md", md)
    return bt


def step5_agent(questions: list[str] | None = None) -> list[dict]:
    questions = questions or [
        "Which passengers survived? How good is the model and how does it compare to gradient boosting?",
        "Why did the model predict survival for individual passengers?",
        "Will demand be high next week? Give me the forecast.",
        "Is the KV cache worth it for repeated predictions?",
    ]
    results = []
    for q in questions:
        res = agent_mod.run_agent(q)
        results.append({"question": q, **res.to_dict()})

    blocks = []
    for r in results:
        trace = "\n".join(
            f"- step {t['step']}: `{t['tool']}({t['args']})` in {t['seconds']}s"
            for t in r["trace"]
        )
        blocks.append(
            f"### {r['question']}\n\n**Answer.** {r['answer']}\n\nCalls ({r['planner']}, {r['steps']} steps, {r['seconds']}s):\n{trace}\n"
        )

    md = f"""# Lesson 5 - The data science agent

Backend: **{backends.backend_label()}**. Planner: `{results[0]['planner'] if results else 'n/a'}`.
The agent sees tool descriptions only; it never sees the dataframes.

{chr(10).join(blocks)}
"""
    _save_json("agent", results)
    _write("05-agent.md", md)
    return results


def write_summary() -> str:
    parts = [
        "# Tabular foundation models - project summary",
        "",
        f"Backend: **{backends.backend_label()}**",
        "",
    ]
    for name in (
        "01-predictions.md",
        "02-kv-cache.md",
        "03-interpretation.md",
        "04-forecasting.md",
        "05-agent.md",
    ):
        p = REPORTS_DIR / name
        if p.exists():
            parts.append(p.read_text())
            parts.append("\n---\n")
    text = "\n".join(parts)
    _write("summary.md", text)
    return str(REPORTS_DIR / "summary.md")
