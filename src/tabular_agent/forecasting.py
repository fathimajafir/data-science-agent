"""Forecast a daily series by reframing it as tabular regression.

This is the TabPFN-TS idea in one file: turn the series into a supervised table
of lag, rolling and calendar features, then let the tabular model predict the
next value. Because the model is fitted once and queried step by step, the KV
cache from lesson 2 pays off here as well.

Known-future covariates (promotions, holidays) are passed in as columns; that is
the main reason this reframing often beats a pure univariate model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import backends, core

LAGS = (1, 2, 3, 7, 14, 28)
ROLLING = (7, 28)
QUANTILES = [0.1, 0.5, 0.9]


def _calendar(dates: pd.DatetimeIndex) -> pd.DataFrame:
    doy = dates.dayofyear.to_numpy()
    dow = dates.dayofweek.to_numpy()
    return pd.DataFrame(
        {
            "dayofweek": dow,
            "is_weekend": (dow >= 5).astype(int),
            "month": dates.month.to_numpy(),
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
            "trend_index": np.arange(len(dates)),
        },
        index=dates,
    )


def build_features(series: pd.DataFrame, lags=LAGS, rolling=ROLLING) -> pd.DataFrame:
    """Supervised table: row t predicts units at t (features use t-1 and older)."""
    s = series.set_index("date").sort_index()
    feats = _calendar(s.index)
    for name in ("promo", "holiday"):
        if name in s.columns:
            feats[name] = s[name].to_numpy()
    units = s["units"]
    for lag in lags:
        feats[f"lag_{lag}"] = units.shift(lag).to_numpy()
    for w in rolling:
        feats[f"roll_mean_{w}"] = units.shift(1).rolling(w).mean().to_numpy()
        feats[f"roll_std_{w}"] = units.shift(1).rolling(w).std().to_numpy()
    feats["units"] = units.to_numpy()
    feats["date"] = s.index
    return feats.reset_index(drop=True)


def _future_row(
    history: pd.DataFrame, date: pd.Timestamp, promo: float, holiday: float,
    columns: list[str],
) -> pd.DataFrame:
    """One feature row for `date`, using `history` (all data before it)."""
    units = history["units"].to_numpy()
    row: dict[str, Any] = _calendar(pd.DatetimeIndex([date])).iloc[0].to_dict()
    row["promo"] = promo
    row["holiday"] = holiday
    for lag in LAGS:
        row[f"lag_{lag}"] = units[-lag]
    for w in ROLLING:
        tail = units[-w:]
        row[f"roll_mean_{w}"] = float(np.mean(tail))
        row[f"roll_std_{w}"] = float(np.std(tail, ddof=1)) if w > 1 else 0.0
    return pd.DataFrame([row])[columns]


def _predict_quantiles(model, X: pd.DataFrame, backend: str, task: str = "regression"):
    """Return (median, lower, upper) arrays for the requested quantiles."""
    if backend == "fallback":
        med = np.asarray(model.predict(X), dtype=float)
        # crude intervals so plots still render in offline mode
        return med, med - 0.1 * np.abs(med), med + 0.1 * np.abs(med)
    preds = model.predict(X, output_type="quantiles", quantiles=QUANTILES)
    if isinstance(preds, dict):
        low, med, high = (np.asarray(preds[q], dtype=float) for q in QUANTILES)
    else:
        low, med, high = (np.asarray(p, dtype=float) for p in preds)
    return med, low, high


def _pinball(y_true, q, q_pred) -> float:
    diff = y_true - q_pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def backtest(
    series: pd.DataFrame,
    horizon: int = 14,
    *,
    origins: int = 3,
    fit_with_cache: bool = True,
    model=None,
) -> dict[str, Any]:
    """Rolling-origin backtest of a recursive multi-step forecast.

    Each origin refits on everything before the cutoff and forecasts `horizon`
    days one at a time, feeding each prediction back in as history.
    """
    backend = backends.resolve_backend()
    table = build_features(series).dropna().reset_index(drop=True)
    columns = [c for c in table.columns if c not in ("units", "date")]

    n = len(table)
    runs = []
    for k in range(origins):
        cut = n - horizon - k * horizon
        if cut <= 120:
            continue
        train = table.iloc[:cut]
        test = table.iloc[cut : cut + horizon]

        reg = model or core.build_regressor(fit_with_cache=fit_with_cache)
        reg.fit(train[columns], train["units"])

        # All real data strictly before the first forecast step.
        history = series[series["date"] < test["date"].iloc[0]].copy()
        preds = []
        for step in range(len(test)):
            target_date = test["date"].iloc[step]
            row = test.iloc[[step]]
            promo = float(row["promo"].iloc[0]) if "promo" in row else 0.0
            holiday = float(row["holiday"].iloc[0]) if "holiday" in row else 0.0
            X = _future_row(history, target_date, promo, holiday, columns)
            med, low, high = _predict_quantiles(reg, X, backend)
            preds.append(
                {
                    "date": target_date,
                    "y_true": float(row["units"].iloc[0]),
                    "y_pred": float(med[0]),
                    "y_low": float(low[0]),
                    "y_high": float(high[0]),
                }
            )
            history = pd.concat(
                [
                    history,
                    pd.DataFrame(
                        {
                            "date": [target_date],
                            "units": [float(med[0])],
                            "promo": [promo],
                            "holiday": [holiday],
                        }
                    ),
                ],
                ignore_index=True,
            )

        fc = pd.DataFrame(preds)
        y = fc["y_true"].to_numpy()
        yhat = fc["y_pred"].to_numpy()
        runs.append(
            {
                "origin": str(fc["date"].iloc[0].date()),
                "mae": float(np.mean(np.abs(y - yhat))),
                "mape": float(np.mean(np.abs(y - yhat) / y)),
                "rmse": float(np.sqrt(np.mean((y - yhat) ** 2))),
                "coverage_80": float(
                    np.mean((y >= fc["y_low"]) & (y <= fc["y_high"]))
                ),
                "pinball_10": _pinball(y, 0.1, fc["y_low"].to_numpy()),
                "pinball_90": _pinball(y, 0.9, fc["y_high"].to_numpy()),
                "forecast": fc,
            }
        )

    if not runs:
        raise ValueError("Not enough history for the requested backtest")

    agg = {
        m: float(np.mean([r[m] for r in runs]))
        for m in ("mae", "mape", "rmse", "coverage_80", "pinball_10", "pinball_90")
    }
    return {
        "task": "forecasting",
        "model": backends.backend_label(),
        "horizon": horizon,
        "n_features": len(columns),
        "n_train_rows": int(len(train)),
        "runs": runs,
        "metrics": agg,
        "columns": columns,
        "series": series,
    }


def future_forecast(
    series: pd.DataFrame,
    horizon: int = 14,
    *,
    future_promo: float = 0.0,
    future_holiday: float = 0.0,
    model=None,
) -> pd.DataFrame:
    """Forecast past the end of the series (used by the agent)."""
    backend = backends.resolve_backend()
    table = build_features(series).dropna().reset_index(drop=True)
    columns = [c for c in table.columns if c not in ("units", "date")]

    reg = model or core.build_regressor(fit_with_cache=True)
    reg.fit(table[columns], table["units"])

    history = series.copy()
    rows = []
    for step in range(1, horizon + 1):
        d = series["date"].max() + pd.Timedelta(days=step)
        promo = future_promo if step == 1 else 0.0
        holiday = future_holiday if step == 1 else 0.0
        X = _future_row(history, d, promo, holiday, columns)
        med, low, high = _predict_quantiles(reg, X, backend)
        rows.append(
            {
                "date": d,
                "forecast": float(med[0]),
                "low_80": float(low[0]),
                "high_80": float(high[0]),
            }
        )
        history = pd.concat(
            [
                history,
                pd.DataFrame(
                    {
                        "date": [d],
                        "units": [float(med[0])],
                        "promo": [promo],
                        "holiday": [holiday],
                    }
                ),
            ],
            ignore_index=True,
        )
    return pd.DataFrame(rows)
