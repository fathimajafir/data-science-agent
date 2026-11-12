"""Measure what the KV cache buys you.

Without caching, every ``predict`` call re-encodes the training rows into the
transformer. ``fit_mode="fit_with_cache"`` pays that cost once at fit time and
reuses the attention state, so repeated predictions against the same training
set are much cheaper.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from . import backends, core
from .config import RANDOM_STATE


def _predict(model, X, task):
    if task == "classification":
        return model.predict_proba(X)
    return model.predict(X)


def benchmark(
    df: pd.DataFrame,
    target: str,
    features: list[str] | None,
    task: str,
    n_batches: int = 5,
    batch_size: int = 250,
) -> dict[str, Any]:
    X_train, X_test, y_train, y_test = core.split_data(
        df, target, features, test_size=0.3, stratify=(task == "classification")
    )
    rng = np.random.default_rng(RANDOM_STATE)
    batches = []
    for _ in range(n_batches):
        idx = rng.choice(len(X_test), size=min(batch_size, len(X_test)), replace=False)
        batches.append(X_test.iloc[idx])

    out: dict[str, Any] = {"backend": backends.backend_label(), "batches": []}

    for label, cached in (("without", False), ("with", True)):
        model = core.build_classifier(fit_with_cache=cached) if task == "classification" else core.build_regressor(fit_with_cache=cached)
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        for b in batches:
            _predict(model, b, task)
        pred_s = time.perf_counter() - t0

        out[label] = {
            "fit_seconds": round(fit_s, 3),
            "total_predict_seconds": round(pred_s, 3),
            "per_batch_seconds": round(pred_s / len(batches), 4),
        }
        out["batches"].append(
            {
                "mode": label,
                "fit_seconds": out[label]["fit_seconds"],
                "predict_seconds": out[label]["total_predict_seconds"],
            }
        )

    out["speedup_predict"] = round(
        out["without"]["total_predict_seconds"]
        / max(out["with"]["total_predict_seconds"], 1e-9),
        2,
    )
    out["speedup_total"] = round(
        (out["without"]["fit_seconds"] + out["without"]["total_predict_seconds"])
        / max(
            out["with"]["fit_seconds"] + out["with"]["total_predict_seconds"], 1e-9
        ),
        2,
    )
    out["n_batches"] = n_batches
    out["batch_size"] = min(batch_size, len(X_test))
    out["n_train"] = int(len(X_train))
    return out
