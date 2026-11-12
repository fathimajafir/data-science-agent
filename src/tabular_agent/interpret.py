"""Explain TabPFN predictions: global importance and per-row drivers.

Three explanation paths, one result shape:

* local weights    -> shapiq imputation explainer, which reuses the KV cache
* hosted API       -> ``tabpfn_client.explain.permutation_shap`` (batched)
* offline fallback -> occlusion: replace one feature with a background value
                      and measure how far the prediction moves

All of them answer the same two questions: which features move the model across
the dataset, and which features moved it for one particular row.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import backends, core
from .config import FIGURES_DIR, RANDOM_STATE


def _is_classification(y: pd.Series) -> bool:
    return not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 10


def _predict_scalar(model, X: pd.DataFrame, task: str) -> np.ndarray:
    if task == "classification":
        proba = np.asarray(model.predict_proba(X))
        return proba[:, 1] if proba.shape[1] == 2 else proba.max(axis=1)
    return np.asarray(model.predict(X), dtype=float)


def _background(X_train: pd.DataFrame, n: int = 80) -> pd.DataFrame:
    return X_train.sample(min(n, len(X_train)), random_state=RANDOM_STATE)


# --------------------------------------------------------------------------
# Explanation backends
# --------------------------------------------------------------------------


def _shap_local(model, X_train, X_explain, task, budget):
    from tabpfn_extensions.interpretability.shapiq import (
        get_tabpfn_imputation_explainer,
    )

    try:
        explainer = get_tabpfn_imputation_explainer(
            model=model, data=X_train, index="SV", max_order=1
        )
    except Exception:
        explainer = get_tabpfn_imputation_explainer(model=model, data=X_train)
    sv = explainer.explain(X_explain.to_numpy(), budget=budget)
    vals = np.asarray(sv.values, dtype=float)
    # k-SII returns (rows, features, features); keep the first-order diagonal.
    if vals.ndim == 3:
        vals = np.einsum("rii->ri", vals)
    elif vals.ndim == 1:
        vals = vals.reshape(1, -1)
    base = getattr(sv, "baseline_value", None)
    if base is None:
        base = np.full(len(X_explain), np.nan)
    base = np.asarray(base, dtype=float).reshape(-1)
    if base.size == 1:
        base = np.repeat(base, len(X_explain))
    return vals, base


def _shap_api(model, X_train, X_explain, task):
    from tabpfn_client.explain import permutation_shap

    def predict_fn(X):
        return _predict_scalar(model, pd.DataFrame(X, columns=X_train.columns), task)

    exp = permutation_shap(
        predict_fn,
        X_explain,
        background=_background(X_train),
        n_permutations=2,
        random_state=RANDOM_STATE,
    )
    vals = np.asarray(exp.values, dtype=float)
    if vals.ndim == 3:
        vals = vals[..., -1]
    base = np.asarray(exp.base_values, dtype=float).reshape(-1)
    return vals, base


def _occlusion(model, X_train, X_explain, task, n_draws: int = 3):
    """Backend-agnostic attribution: ablate a feature to a background value."""
    bg = _background(X_train, n=max(n_draws, 1))
    base_pred = _predict_scalar(model, bg, task).mean()
    filled = X_explain.copy()
    for col in X_explain.columns:
        if pd.api.types.is_numeric_dtype(X_train[col]):
            filled[col] = X_train[col].median()
        else:
            filled[col] = X_train[col].mode(dropna=True).iloc[0]
    ref = _predict_scalar(model, filled, task)

    vals = np.zeros((len(X_explain), X_explain.shape[1]))
    for i, col in enumerate(X_explain.columns):
        probe = filled.copy()
        probe[col] = X_explain[col].to_numpy()
        moved = _predict_scalar(model, probe, task)
        vals[:, i] = moved - ref
    base = np.full(len(X_explain), base_pred)
    return vals, base


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def explain(
    df: pd.DataFrame,
    target: str,
    features: list[str] | None = None,
    *,
    task: str | None = None,
    n_rows: int = 6,
    budget: int = 32,
    fit_with_cache: bool = True,
    model: Any | None = None,
) -> dict[str, Any]:
    """Fit a model (or reuse ``model``), explain sampled rows, summarise global importance."""
    features = features or [c for c in df.columns if c != target]
    task = task or ("classification" if _is_classification(df[target]) else "regression")
    backend = backends.resolve_backend()

    data = df.dropna(subset=[target])
    X, y = data[features], data[target]
    if model is None:
        model = (
            core.build_classifier(fit_with_cache=fit_with_cache)
            if task == "classification"
            else core.build_regressor(fit_with_cache=fit_with_cache)
        )
        model.fit(X, y)

    X_explain = X.sample(min(n_rows, len(X)), random_state=RANDOM_STATE)

    if backend == "local":
        method = "shapiq imputation SHAP (KV cache)"
        vals, base = _shap_local(model, X, X_explain, task, budget)
    elif backend == "api":
        method = "permutation SHAP (Prior Labs API)"
        vals, base = _shap_api(model, X, X_explain, task)
    else:
        method = "occlusion (offline fallback, not SHAP)"
        vals, base = _occlusion(model, X, X_explain, task)

    shap_df = pd.DataFrame(vals, columns=features, index=X_explain.index)
    importance = (
        shap_df.abs().mean().sort_values(ascending=False).rename("importance")
    )

    # Direction for numeric features: does a higher value push the prediction up?
    direction = {}
    for col in features:
        if pd.api.types.is_numeric_dtype(X[col]):
            corr = shap_df[col].corr(X_explain[col].astype(float), method="spearman")
            if pd.notna(corr):
                direction[col] = "higher value pushes up" if corr >= 0 else "higher value pushes down"

    probs = _predict_scalar(model, X_explain, task)
    row_explanations = []
    for pos, (idx, row) in enumerate(shap_df.iterrows()):
        top = row.abs().sort_values(ascending=False).head(5)
        row_explanations.append(
            {
                "row_index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
                "prediction": float(probs[pos]),
                "base_value": float(base[pos]) if np.isfinite(base[pos]) else None,
                "top_drivers": [
                    {
                        "feature": f,
                        "shap": float(row[f]),
                        "value": _json_value(X_explain.loc[idx, f]),
                        "direction": direction.get(f, ""),
                    }
                    for f in top.index
                ],
            }
        )

    return {
        "task": task,
        "method": method,
        "model": backends.backend_label(),
        "features": features,
        "shap_values": shap_df,
        "global_importance": [
            {"feature": f, "importance": float(v), "direction": direction.get(f, "")}
            for f, v in importance.items()
        ],
        "row_explanations": row_explanations,
        "n_explained": len(X_explain),
    }


def _json_value(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v if isinstance(v, (int, float, str, bool)) or v is None else str(v)


def plot_global_importance(result: dict, path=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    imp = pd.DataFrame(result["global_importance"]).head(12)[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=150)
    ax.barh(imp["feature"], imp["importance"], color="#2b6cb0")
    ax.set_xlabel("mean |attribution|")
    ax.set_title(f"Global feature importance ({result['method']})", fontsize=9)
    fig.tight_layout()
    path = path or FIGURES_DIR / "global_importance.png"
    fig.savefig(path)
    plt.close(fig)
    return path
