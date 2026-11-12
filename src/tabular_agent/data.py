"""Dataset loading and preparation.

Three datasets, one per problem type:

* ``titanic``    - binary classification, real data with categoricals and NaNs
* ``california`` - regression, real data, all numeric
* ``demand``     - daily time series, synthetic but with a known generating
                   process so we can check whether the model recovers the
                   real drivers during interpretation

Every loader degrades to a synthetic clone when the network is unavailable so
the project still runs end to end offline.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RANDOM_STATE, RAW_DIR

TITANIC_URL = (
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
)

# Columns dropped from the Titanic model input: direct target copies or
# deterministic functions of the columns we keep.
TITANIC_FEATURES = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked",
    "deck",
]
TITANIC_TARGET = "survived"


def _download(url: str, dest: Path, timeout: int = 20) -> Path | None:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            dest.write_bytes(resp.read())
        return dest
    except Exception:
        return None


def load_titanic() -> pd.DataFrame:
    """Real Titanic passenger data, or a schema-compatible synthetic clone."""
    path = _download(TITANIC_URL, RAW_DIR / "titanic.csv")
    if path is not None:
        try:
            df = pd.read_csv(path)
            if TITANIC_TARGET in df.columns:
                df.attrs["source"] = "real (seaborn-data)"
                return df
        except Exception:
            pass

    rng = np.random.default_rng(RANDOM_STATE)
    n = 900
    pclass = rng.choice([1, 2, 3], size=n, p=[0.24, 0.21, 0.55])
    sex = rng.choice(["male", "female"], size=n, p=[0.65, 0.35])
    age = np.clip(rng.normal(30, 14, n), 0.4, 80)
    sibsp = rng.choice([0, 1, 2, 3], size=n, p=[0.68, 0.23, 0.06, 0.03])
    parch = rng.choice([0, 1, 2], size=n, p=[0.76, 0.13, 0.11])
    fare = np.round(
        np.exp(rng.normal(2.5, 0.9, n)) * (1.0 + 0.8 * (4 - pclass)), 2
    )
    embarked = rng.choice(["S", "C", "Q"], size=n, p=[0.72, 0.19, 0.09])
    deck = rng.choice([None, "C", "B", "D", "E", "F", "G"], size=n,
                      p=[0.77, 0.06, 0.05, 0.04, 0.04, 0.03, 0.01])
    lin = (
        -1.2
        + 1.0 * (sex == "female")
        - 0.9 * (pclass - 1)
        - 0.015 * (age - 30)
        + 0.4 * (embarked == "C")
    )
    survived = (rng.random(n) < 1 / (1 + np.exp(-lin))).astype(int)
    df = pd.DataFrame(
        {
            TITANIC_TARGET: survived,
            "pclass": pclass,
            "sex": sex,
            "age": np.round(age, 1),
            "sibsp": sibsp,
            "parch": parch,
            "fare": fare,
            "embarked": embarked,
            "deck": deck,
        }
    )
    df.loc[rng.random(n) < 0.2, "age"] = np.nan
    df.loc[rng.random(n) < 0.002, "embarked"] = np.nan
    df.attrs["source"] = "synthetic clone (offline)"
    return df


def load_california(max_rows: int = 6000) -> pd.DataFrame:
    """California housing (1990 census blocks). Target: median house value."""
    from sklearn.datasets import fetch_california_housing

    try:
        data = fetch_california_housing(as_frame=True)
        df = data.frame.copy()
        df.attrs["source"] = "real (sklearn / StatLib)"
    except Exception:
        rng = np.random.default_rng(RANDOM_STATE)
        n = 5000
        med_inc = rng.lognormal(1.0, 0.5, n)
        house_age = rng.uniform(1, 52, n)
        ave_rooms = rng.normal(5.4, 2.5, n).clip(0.8, 15)
        ave_occ = rng.normal(3.0, 1.1, n).clip(0.5, 12)
        lat = rng.uniform(32.5, 42.0, n)
        lon = rng.uniform(-124.3, -114.3, n)
        value = (
            0.45 * med_inc
            + 0.35 * ave_rooms
            - 0.25 * ave_occ
            + 0.02 * (42 - lat)
            + rng.normal(0, 0.5, n)
        ).clip(0.15, 5.0)
        df = pd.DataFrame(
            {
                "MedInc": med_inc,
                "HouseAge": house_age,
                "AveRooms": ave_rooms,
                "AveBedrms": rng.normal(1.1, 0.3, n).clip(0.3, 5),
                "Population": rng.integers(3, 30000, n),
                "AveOccup": ave_occ,
                "Latitude": lat,
                "Longitude": lon,
                "MedHouseVal": value,
            }
        )
        df.attrs["source"] = "synthetic clone (offline)"

    df = df.dropna().reset_index(drop=True)
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=RANDOM_STATE).reset_index(drop=True)
    return df


def make_demand_series(days: int = 1095, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Daily store demand with trend, weekly/annual seasonality and promotions.

    The generating process is known, so interpretation can be checked against
    the truth: weekday effect, annual seasonality, promo lift (planned ahead of
    time, so it is a legitimate future covariate), holiday spikes and noise.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=days, freq="D")
    t = np.arange(days)

    trend = 1.0 + 0.00055 * t
    weekly = np.array([1.05, 0.98, 1.0, 1.02, 1.15, 1.28, 1.12])[
        dates.dayofweek
    ]
    annual = 1.0 + 0.18 * np.sin(2 * np.pi * (dates.dayofyear - 100) / 365.25)
    promo = (rng.random(days) < 0.06).astype(int)
    holiday = np.isin(
        dates.strftime("%m-%d"), ["12-24", "12-25", "12-26", "01-01", "07-04"]
    ).astype(int)
    noise = rng.lognormal(0, 0.07, days)

    units = (
        320 * trend * weekly * annual * (1 + 0.38 * promo) * (1 + 0.30 * holiday)
        * noise
    )
    df = pd.DataFrame(
        {
            "date": dates,
            "units": np.round(units).astype(int),
            "promo": promo.astype(float),
            "holiday": holiday.astype(float),
        }
    )
    df.attrs["source"] = "synthetic with known ground truth"
    return df


def dataset_summary(df: pd.DataFrame, target: str) -> dict:
    """Compact schema description, used by the agent's inspect tool."""
    rows = []
    for col in df.columns:
        s = df[col]
        missing = float(s.isna().mean())
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            kind = "numeric"
            extra = f"mean={s.mean():.3g}, std={s.std():.3g}"
        else:
            kind = "categorical"
            nun = s.nunique(dropna=True)
            extra = f"{nun} levels, e.g. {list(s.dropna().astype(str).unique()[:3])}"
        rows.append(
            {
                "column": col,
                "type": kind,
                "missing_pct": round(100 * missing, 1),
                "summary": extra,
                "is_target": col == target,
            }
        )
    return {
        "n_rows": int(len(df)),
        "target": target,
        "columns": rows,
        "source": df.attrs.get("source", "unknown"),
    }
