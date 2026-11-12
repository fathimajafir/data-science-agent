"""Thin, reusable wrappers around TabPFN for classification and regression.

TabPFN follows the scikit-learn estimator API, but its "fit" is really an
in-context-learning forward pass. The wrappers here keep the details that matter
in practice: which checkpoint, device placement, timing, and a fair tabular
baseline for comparison.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from . import backends
from .config import RANDOM_STATE


def build_classifier(*, fit_with_cache: bool = False, **overrides):
    return backends.make_estimator(
        "classification", fit_with_cache=fit_with_cache, **overrides
    )


def build_regressor(*, fit_with_cache: bool = False, **overrides):
    return backends.make_estimator(
        "regression", fit_with_cache=fit_with_cache, **overrides
    )


def split_data(
    df: pd.DataFrame,
    target: str,
    features: list[str] | None = None,
    test_size: float = 0.25,
    stratify: bool = True,
):
    """Split into X/y frames. X stays a DataFrame: TabPFN handles raw dtypes."""
    features = features or [c for c in df.columns if c != target]
    data = df.dropna(subset=[target])
    X = data[features]
    y = data[target]
    strat = y if stratify and _is_classification(y) else None
    return train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=strat
    )


def _is_classification(y: pd.Series) -> bool:
    return not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 10


def classification_metrics(y_true, proba, classes) -> dict[str, float]:
    proba = np.asarray(proba)
    preds = np.asarray(classes)[proba.argmax(axis=1)]
    out = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "f1_macro": float(f1_score(y_true, preds, average="macro")),
        "log_loss": float(
            log_loss(y_true, proba, labels=list(classes))
        ),
    }
    try:
        out["roc_auc"] = float(
            roc_auc_score(
                y_true, proba[:, 1] if proba.shape[1] == 2 else proba,
                multi_class="ovr" if proba.shape[1] > 2 else "raise",
            )
        )
    except ValueError:
        out["roc_auc"] = float("nan")
    return out


def regression_metrics(y_true, preds) -> dict[str, float]:
    preds = np.asarray(preds, dtype=float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, preds))),
        "mae": float(mean_absolute_error(y_true, preds)),
        "r2": float(r2_score(y_true, preds)),
    }


def run_classification(
    df: pd.DataFrame,
    target: str,
    features: list[str] | None = None,
    *,
    test_size: float = 0.25,
    fit_with_cache: bool = False,
    model: Any | None = None,
) -> dict[str, Any]:
    X_train, X_test, y_train, y_test = split_data(
        df, target, features, test_size=test_size
    )
    clf = model or build_classifier(fit_with_cache=fit_with_cache)

    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_s = time.perf_counter() - t0

    backend_label = backends.backend_label()

    t0 = time.perf_counter()
    proba = clf.predict_proba(X_test)
    predict_s = time.perf_counter() - t0

    return {
        "task": "classification",
        "model": backend_label,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X_train.shape[1]),
        "fit_seconds": round(fit_s, 2),
        "predict_seconds": round(predict_s, 2),
        "metrics": classification_metrics(y_test, proba, clf.classes_),
        "classes": [str(c) for c in clf.classes_],
        "proba": proba,
        "y_test": y_test,
        "X_test": X_test,
        "X_train": X_train,
        "y_train": y_train,
        "estimator": clf,
    }


def run_regression(
    df: pd.DataFrame,
    target: str,
    features: list[str] | None = None,
    *,
    test_size: float = 0.25,
    fit_with_cache: bool = False,
    model: Any | None = None,
) -> dict[str, Any]:
    X_train, X_test, y_train, y_test = split_data(
        df, target, features, test_size=test_size, stratify=False
    )
    reg = model or build_regressor(fit_with_cache=fit_with_cache)

    t0 = time.perf_counter()
    reg.fit(X_train, y_train)
    fit_s = time.perf_counter() - t0

    backend_label = backends.backend_label()

    t0 = time.perf_counter()
    preds = reg.predict(X_test)
    predict_s = time.perf_counter() - t0

    return {
        "task": "regression",
        "model": backend_label,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X_train.shape[1]),
        "fit_seconds": round(fit_s, 2),
        "predict_seconds": round(predict_s, 2),
        "metrics": regression_metrics(y_test, preds),
        "preds": preds,
        "y_test": y_test,
        "X_test": X_test,
        "X_train": X_train,
        "y_train": y_train,
        "estimator": reg,
    }


# --------------------------------------------------------------------------
# Baselines: a tuned-ish gradient booster on ordinally encoded features.
# TabPFN's selling point is matching or beating this with zero tuning.
# --------------------------------------------------------------------------


def _encode(X_train: pd.DataFrame, X_test: pd.DataFrame):
    cat_cols = [
        c for c in X_train.columns if not pd.api.types.is_numeric_dtype(X_train[c])
    ]
    if not cat_cols:
        return X_train.to_numpy(float), X_test.to_numpy(float)
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    Xtr = X_train.copy()
    Xte = X_test.copy()
    Xtr[cat_cols] = enc.fit_transform(X_train[cat_cols].astype(str))
    Xte[cat_cols] = enc.transform(X_test[cat_cols].astype(str))
    # pandas nullable dtypes -> plain float with NaN, which HGB handles natively
    return (
        Xtr.astype(float).to_numpy(),
        Xte.astype(float).to_numpy(),
    )


def run_baseline_classification(df, target, features=None, *, test_size=0.25):
    X_train, X_test, y_train, y_test = split_data(
        df, target, features, test_size=test_size
    )
    Xtr, Xte = _encode(X_train, X_test)
    clf = HistGradientBoostingClassifier(
        random_state=RANDOM_STATE, learning_rate=0.06, max_iter=400
    )
    t0 = time.perf_counter()
    clf.fit(Xtr, y_train)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    proba = clf.predict_proba(Xte)
    predict_s = time.perf_counter() - t0
    return {
        "task": "classification",
        "model": "HistGradientBoosting (baseline)",
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X_train.shape[1]),
        "fit_seconds": round(fit_s, 2),
        "predict_seconds": round(predict_s, 2),
        "metrics": classification_metrics(y_test, proba, clf.classes_),
    }


def run_baseline_regression(df, target, features=None, *, test_size=0.25):
    X_train, X_test, y_train, y_test = split_data(
        df, target, features, test_size=test_size, stratify=False
    )
    Xtr, Xte = _encode(X_train, X_test)
    reg = HistGradientBoostingRegressor(
        random_state=RANDOM_STATE, learning_rate=0.06, max_iter=400
    )
    t0 = time.perf_counter()
    reg.fit(Xtr, y_train)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    preds = reg.predict(Xte)
    predict_s = time.perf_counter() - t0
    return {
        "task": "regression",
        "model": "HistGradientBoosting (baseline)",
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X_train.shape[1]),
        "fit_seconds": round(fit_s, 2),
        "predict_seconds": round(predict_s, 2),
        "metrics": regression_metrics(y_test, preds),
    }
