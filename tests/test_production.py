"""Tests for the production surface: CLI, retries, HTTP service, diagnostics."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("TABPFN_BACKEND", "fallback")

from tabular_agent import __version__  # noqa: E402
from tabular_agent.agent.llm import ScriptPlanner  # noqa: E402
from tabular_agent.backends import RetryingEstimator, diagnose  # noqa: E402
from tabular_agent.cli import main as cli_main  # noqa: E402


def test_version_is_semver():
    parts = __version__.split(".")
    assert len(parts) == 3 and all(p.isdigit() for p in parts)


def test_cli_version_flag():
    with pytest.raises(SystemExit) as exc:
        cli_main(["--version"])
    assert exc.value.code == 0


def test_cli_doctor():
    assert cli_main(["doctor"]) == 0


def test_cli_datasets():
    assert cli_main(["datasets"]) == 0


def test_cli_ask_end_to_end():
    assert cli_main(["ask", "How accurate is the survival model?"]) == 0


def test_diagnose_never_raises():
    checks = diagnose()
    assert checks["backend"] in {"api", "local", "fallback"}
    assert "backend_label" in checks


class FlakyModel:
    def __init__(self, failures: int, exc: Exception | None = None):
        self.failures = failures
        self.calls = 0
        self.exc = exc or ConnectionError("connection reset")

    def predict(self, X):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exc
        return [42.0] * len(X)


def test_retry_proxy_recovers_from_transient_errors(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    inner = FlakyModel(failures=2)
    model = RetryingEstimator(inner, attempts=3, base_delay=0.01)
    assert model.predict([1, 2]) == [42.0, 42.0]
    assert inner.calls == 3


def test_retry_proxy_does_not_retry_permanent_errors():
    inner = FlakyModel(failures=5, exc=ValueError("bad input"))
    model = RetryingEstimator(inner, attempts=3, base_delay=0.0)
    with pytest.raises(ValueError):
        model.predict([1])
    assert inner.calls == 1


def test_retry_proxy_gives_up_after_attempts(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    inner = FlakyModel(failures=99, exc=TimeoutError("timed out"))
    model = RetryingEstimator(inner, attempts=3, base_delay=0.01)
    with pytest.raises(TimeoutError):
        model.predict([1])
    assert inner.calls == 3


def test_retry_proxy_delegates_attributes():
    class Inner:
        classes_ = ["a", "b"]

        def predict(self, X):
            return X

    model = RetryingEstimator(Inner())
    assert model.classes_ == ["a", "b"]


@pytest.mark.parametrize(
    ("question", "tool"),
    [
        ("forecast demand for next week", "forecast"),
        ("what drives survival predictions?", "explain_model"),
        ("is the cache faster?", "benchmark_kv_cache"),
        ("how good are house price predictions?", "train_and_evaluate"),
        ("what columns are in the data?", "inspect_dataset"),
    ],
)
def test_scripted_planner_routing(question, tool):
    planner = ScriptPlanner()
    planner.start(question, [])
    decision = planner.next(None)
    assert decision["type"] == "tool"
    assert decision["name"] == tool


def test_http_service():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from tabular_agent.serve import create_app

    client = TestClient(create_app())
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["backend"] == "fallback"

    tools = client.get("/tools")
    assert tools.status_code == 200
    assert {t["name"] for t in tools.json()} >= {
        "inspect_dataset",
        "train_and_evaluate",
        "forecast",
    }

    resp = client.post("/ask", json={"question": "What does the titanic data look like?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"]
    assert body["trace"]

    bad = client.post("/ask", json={"question": "hi"})
    assert bad.status_code == 422  # min_length=3
