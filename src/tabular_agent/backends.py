"""One place that decides how TabPFN is reached.

Three backends, chosen from the environment (``TABPFN_BACKEND`` forces one,
``auto`` resolves in order api -> local -> fallback):

``api``       Prior Labs hosted inference through ``tabpfn-client``. No GPU, no
              model download. Needs ``TABPFN_TOKEN`` (free key at
              https://platform.priorlabs.ai/account/api-keys). Estimators are
              wrapped with automatic retries and backoff for transient errors.
``local``     The ``tabpfn`` package with weights on this machine. Needs a GPU
              for anything past ~1k rows and a one-time license acceptance plus
              a Hugging Face download.
``fallback``  scikit-learn gradient boosting behind the same interface. Only for
              offline smoke runs; every result is labelled so it can never be
              mistaken for TabPFN output.

All three expose the same surface: ``fit``, ``predict``, ``predict_proba`` and
``classes_`` for classifiers, which is what every other module relies on.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from typing import Any

from . import config  # noqa: F401  (importing config loads the .env file)
from .config import LLM_BASE_URL, LLM_MODEL, llm_configured

logger = logging.getLogger(__name__)

FALLBACK_LABEL = "sklearn-fallback"
PRIOR_LABS_API = "https://api.priorlabs.ai"
KEY_URL = "https://platform.priorlabs.ai/account/api-keys"

_TRANSIENT_MARKERS = (
    "timeout",
    "timed out",
    "connection",
    "temporarily unavailable",
    "rate limit",
    "too many requests",
    "429",
    "500",
    "502",
    "503",
    "504",
    "overloaded",
)


def requested_backend() -> str:
    return os.getenv("TABPFN_BACKEND", "auto").strip().lower()


def local_weights_cached() -> bool:
    """True when a Prior Labs checkpoint is already in the Hugging Face cache."""
    import pathlib

    hf_home = os.getenv("HF_HOME")
    hub = (
        pathlib.Path(hf_home) / "hub"
        if hf_home
        else pathlib.Path.home() / ".cache" / "huggingface" / "hub"
    )
    if not hub.is_dir():
        return False
    return any(
        p.is_dir() and p.name.lower().startswith("models--prior-labs")
        for p in hub.iterdir()
    )


def backend_available(name: str) -> bool:
    if name == "api":
        return bool(_api_token())
    if name == "local":
        try:
            import tabpfn  # noqa: F401
        except Exception:
            return False
        # Importing works even without weights; only call it available when the
        # checkpoint is already downloaded, otherwise `auto` would hang on a
        # blocked Hugging Face download instead of degrading cleanly.
        return local_weights_cached()
    if name == "fallback":
        return True
    return False


def resolve_backend() -> str:
    """Pick a backend, preferring what the environment can actually do."""
    want = requested_backend()
    if want not in ("auto", "api", "local", "fallback"):
        raise ValueError(
            f"TABPFN_BACKEND={want!r} is invalid; use auto, api, local or fallback."
        )
    if want != "auto":
        return want
    if backend_available("api"):
        return "api"
    if backend_available("local"):
        return "local"
    return "fallback"


# --------------------------------------------------------------------------
# Hosted API
# --------------------------------------------------------------------------

_API_READY = False


def _api_token() -> str | None:
    return os.getenv("TABPFN_TOKEN") or os.getenv("TABPFN_API_KEY")


def _ensure_api() -> None:
    global _API_READY
    if _API_READY:
        return
    token = _api_token()
    if not token:
        raise RuntimeError(
            f"No TabPFN token. Create one at {KEY_URL} and set TABPFN_TOKEN "
            "(or TABPFN_API_KEY) in the environment or .env."
        )
    import tabpfn_client

    tabpfn_client.set_access_token(token)
    _API_READY = True


class RetryingEstimator:
    """Wraps an estimator and retries transient failures with backoff.

    Network hiccups, rate limits and 5xx responses from the hosted API are
    retried a bounded number of times; anything that looks like a permanent
    error (bad input, auth, quota) is raised immediately.
    """

    def __init__(
        self,
        inner: Any,
        *,
        attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 8.0,
    ):
        self._inner = inner
        self.attempts = max(1, attempts)
        self.base_delay = base_delay
        self.max_delay = max_delay

    @staticmethod
    def _transient(exc: BaseException) -> bool:
        if isinstance(exc, (ConnectionError, TimeoutError)):
            return True
        message = str(exc).lower()
        return any(marker in message for marker in _TRANSIENT_MARKERS)

    def _retry(self, op: str, fn: Callable, *args, **kwargs):
        for attempt in range(1, self.attempts + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                if attempt == self.attempts or not self._transient(exc):
                    raise
                delay = min(self.base_delay * 2 ** (attempt - 1), self.max_delay)
                logger.warning(
                    "%s failed with %s: %s; retrying in %.1fs (attempt %d/%d)",
                    op,
                    type(exc).__name__,
                    str(exc)[:160],
                    delay,
                    attempt + 1,
                    self.attempts,
                )
                time.sleep(delay)

    def fit(self, X, y, *args, **kwargs):
        return self._retry("fit", self._inner.fit, X, y, *args, **kwargs)

    def predict(self, X, *args, **kwargs):
        return self._retry("predict", self._inner.predict, X, *args, **kwargs)

    def predict_proba(self, X, *args, **kwargs):
        return self._retry("predict_proba", self._inner.predict_proba, X, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # classes_, feature_names_, model_path, ... delegate to the inner model.
        return getattr(self._inner, name)


# --------------------------------------------------------------------------
# Offline fallback
# --------------------------------------------------------------------------


class SklearnFallback:
    """Minimal sklearn estimator with the TabPFN call surface (offline only)."""

    def __init__(self, task: str, **kwargs):
        self.task = task
        self.kwargs = kwargs
        self._enc = None
        self._model = None

    def _encode(self, X, fit: bool = False):
        import pandas as pd

        X = pd.DataFrame(X).copy()
        cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        if cat:
            from sklearn.preprocessing import OrdinalEncoder

            if fit:
                self._enc = OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                )
                X[cat] = self._enc.fit_transform(X[cat].astype(str))
            else:
                X[cat] = self._enc.transform(X[cat].astype(str))
        return X.astype(float).to_numpy()

    def fit(self, X, y):
        from sklearn.ensemble import (
            HistGradientBoostingClassifier,
            HistGradientBoostingRegressor,
        )

        Xe = self._encode(X, fit=True)
        if self.task == "classification":
            self._model = HistGradientBoostingClassifier(
                random_state=0, learning_rate=0.06, max_iter=300
            )
        else:
            self._model = HistGradientBoostingRegressor(
                random_state=0, learning_rate=0.06, max_iter=300
            )
        self._model.fit(Xe, y)
        self.classes_ = getattr(self._model, "classes_", None)
        return self

    def predict(self, X):
        return self._model.predict(self._encode(X))

    def predict_proba(self, X):
        return self._model.predict_proba(self._encode(X))


# --------------------------------------------------------------------------
# Estimator factory
# --------------------------------------------------------------------------


def make_estimator(
    task: str,
    *,
    backend: str | None = None,
    fit_with_cache: bool = False,
    model_version: str | None = None,
    **overrides,
):
    """Return an unfitted estimator for ``task`` in {"classification","regression"}."""
    backend = backend or resolve_backend()
    if task not in ("classification", "regression"):
        raise ValueError(f"task must be classification or regression, got {task!r}")

    if backend == "fallback":
        return SklearnFallback(task, **overrides)

    fit_mode = "fit_with_cache" if fit_with_cache else "fit_preprocessors"
    version = model_version or os.getenv("TABPFN_MODEL", "v3.5")

    if backend == "api":
        _ensure_api()
        from tabpfn_client import TabPFNClassifier, TabPFNRegressor
        from tabpfn_client.constants import ModelVersion

        cls = TabPFNClassifier if task == "classification" else TabPFNRegressor
        kwargs = {"fit_mode": fit_mode, "random_state": 0}
        kwargs.update(overrides)
        estimator = cls.create_default_for_version(
            getattr(ModelVersion, _enum_name(version)), **kwargs
        )
        return RetryingEstimator(estimator)

    if backend == "local":
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.constants import ModelVersion

        from .config import DEVICE, N_ESTIMATORS, RANDOM_STATE

        cls = TabPFNClassifier if task == "classification" else TabPFNRegressor
        kwargs = {
            "device": DEVICE,
            "n_estimators": N_ESTIMATORS,
            "random_state": RANDOM_STATE,
            "fit_mode": fit_mode,
            "show_progress_bar": False,
        }
        kwargs.update(overrides)
        return cls.create_default_for_version(
            getattr(ModelVersion, _enum_name(version)), **kwargs
        )

    raise ValueError(f"Unknown TABPFN_BACKEND={backend!r}")


def _enum_name(version: str) -> str:
    return "V" + version.lstrip("v").replace(".", "_").replace("-", "_").upper()


def backend_label() -> str:
    """Human-readable description used in reports."""
    backend = resolve_backend()
    if backend == "api":
        return "TabPFN-3.5-Plus (Prior Labs API)"
    if backend == "local":
        return f"TabPFN {os.getenv('TABPFN_MODEL', 'v3.5')} (local)"
    return FALLBACK_LABEL


def api_usage() -> str | None:
    if resolve_backend() != "api":
        return None
    try:
        import tabpfn_client

        _ensure_api()
        return tabpfn_client.get_api_usage()
    except Exception:
        return None


# --------------------------------------------------------------------------
# Diagnostics (used by `tabular-agent doctor` and the /health endpoint)
# --------------------------------------------------------------------------


def diagnose(timeout: float = 6.0) -> dict[str, Any]:
    """Best-effort health report; never raises."""
    checks: dict[str, Any] = {}

    checks["llm_configured"] = llm_configured()
    if llm_configured():
        checks["llm"] = f"{LLM_MODEL} @ {LLM_BASE_URL}"

    backend = resolve_backend()
    checks["backend"] = backend
    checks["backend_label"] = backend_label()
    checks["token_present"] = bool(_api_token())
    checks["local_weights_cached"] = local_weights_cached()

    if backend == "api":
        checks["api_reachable"] = _ping(PRIOR_LABS_API, timeout)
        if not checks["api_reachable"]:
            checks["hint"] = f"{PRIOR_LABS_API} did not respond; check network/proxy."
    elif backend == "local" and not local_weights_cached():
        checks["hint"] = (
            "Local weights are not in the Hugging Face cache. Accept the license "
            "at https://huggingface.co/Prior-Labs and let tabpfn download once."
        )
    elif backend == "fallback":
        checks["hint"] = (
            "Running on the sklearn placeholder. Set TABPFN_TOKEN "
            f"(free key: {KEY_URL}) for real TabPFN predictions."
        )
    return checks


def _ping(url: str, timeout: float) -> bool:
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=timeout):
            return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False
