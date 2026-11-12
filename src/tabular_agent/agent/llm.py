"""Planners: the part of the agent that decides which tool to call.

Two implementations, same interface:

* :class:`LLMPlanner` uses any OpenAI-compatible chat endpoint with tool calling.
* :class:`ScriptPlanner` is a deterministic router so the project still works
  with no LLM key at all (useful for tests and offline demos).

The agent has no special access to the data: it sees tool descriptions and
compact JSON results, exactly like a real deployment would.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, llm_configured

SYSTEM_PROMPT = """You are a data science agent for tabular prediction tasks.

You have tools that run a tabular foundation model (TabPFN), explain its
predictions, and benchmark its KV cache. You never invent numbers: every number
in your answer must come from a tool result.

Rules:
- Pick the one tool that answers the question; call more tools only if the first
  result is not enough.
- Then answer in at most 6 short sentences, quoting the key metrics with units.
- State which backend produced the numbers.
"""


class LLMPlanner:
    def __init__(self, model: str | None = None):
        from openai import OpenAI

        self.client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
        self.model = model or LLM_MODEL
        self.messages: list[dict] = []
        self._pending_id: str | None = None
        self._pending_name: str | None = None

    def start(self, request: str, tools: list[dict]) -> None:
        self.tools = tools
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": request},
        ]

    def next(self, tool_result: dict | None) -> dict[str, Any]:
        if tool_result is not None and self._pending_id:
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": self._pending_id,
                    "name": self._pending_name,
                    "content": json.dumps(tool_result, default=str)[:6000],
                }
            )
            self._pending_id = None

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.tools,
            tool_choice="auto",
            temperature=0,
        )
        msg = resp.choices[0].message
        if msg.tool_calls:
            call = msg.tool_calls[0]
            self.messages.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.function.name, "arguments": c.function.arguments},
                        }
                        for c in msg.tool_calls
                    ],
                }
            )
            self._pending_id = call.id
            self._pending_name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            return {"type": "tool", "name": call.function.name, "args": args}

        return {"type": "final", "text": msg.content or ""}


class ScriptPlanner:
    """Keyword router with template answers. No LLM, no network."""

    def __init__(self) -> None:
        self._planned: tuple[str, dict] | None = None
        self._request = ""

    def start(self, request: str, tools: list[dict]) -> None:
        self._request = request.lower()
        self._planned = self._route(self._request)

    def _route(self, q: str) -> tuple[str, dict]:
        if re.search(r"look like|describe|summar|schema|columns|what'?s in|inspect", q):
            ds = (
                "demand"
                if re.search(r"demand|time series|forecast", q)
                else "california" if re.search(r"hous|price|california", q) else "titanic"
            )
            return "inspect_dataset", {"dataset": ds}
        if re.search(r"cache|latency|faster|speed", q):
            ds = "california" if re.search(r"house|price|california", q) else "titanic"
            return "benchmark_kv_cache", {"dataset": ds}
        if re.search(r"forecast|next week|demand|tomorrow|units", q):
            return "forecast", {"dataset": "demand", "horizon": 14}
        if re.search(r"why|explain|driv|important|influenc|feature", q):
            ds = "california" if re.search(r"hous|price|california", q) else "titanic"
            return "explain_model", {"dataset": ds, "n_rows": 4}
        if re.search(r"hous|price|california|value|propert", q):
            return "train_and_evaluate", {"dataset": "california"}
        if re.search(r"column|schema|dataset|data|rows|missing", q) and not re.search(
            r"predict|model|accuracy", q
        ):
            ds = "demand" if re.search(r"demand|time series", q) else "titanic"
            return "inspect_dataset", {"dataset": ds}
        return "train_and_evaluate", {"dataset": "titanic"}

    def next(self, tool_result: dict | None) -> dict[str, Any]:
        if tool_result is None and self._planned:
            name, args = self._planned
            return {"type": "tool", "name": name, "args": args}
        return {"type": "final", "text": _template_answer(self._planned[0], tool_result)}


def _fmt(d: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in d.items())


def _template_answer(tool: str, r: dict) -> str:
    if "error" in r:
        return f"The tool failed: {r['error']}"

    if tool == "inspect_dataset":
        cols = ", ".join(c["column"] for c in r["columns"][:8])
        return (
            f"{r['n_rows']} rows, {len(r['columns'])} columns ({cols}). "
            f"Target '{r['target']}' is a {r['problem_type']} problem; "
            f"source: {r['source']}. Target stats: {r['target_stats']}."
        )

    if tool == "train_and_evaluate":
        return (
            f"TabPFN on {r['dataset']}: {_fmt(r['metrics'])} "
            f"({r['n_train']} train / {r['n_test']} test rows, "
            f"fit {r['fit_seconds']}s, predict {r['predict_seconds']}s). "
            f"Gradient-boosting baseline: {_fmt(r['baseline_metrics'])}. "
            f"Backend: {r['backend']}."
        )

    if tool == "explain_model":
        top = "; ".join(
            f"{f['feature']} ({f['importance']:.3f}, {f['direction']})"
            for f in r["top_features"][:4]
        )
        row = r["example_rows"][0] if r["example_rows"] else None
        extra = ""
        if row:
            drivers = ", ".join(
                f"{d['feature']}={d['value']} ({d['shap']:+.3f})"
                for d in row["top_drivers"]
            )
            extra = f" For row {row['row_index']} (prediction {row['prediction']:.3f}) the drivers were {drivers}."
        return f"Most influential features on {r['dataset']}: {top}.{extra} Method: {r['method']}."

    if tool == "forecast":
        days = ", ".join(
            f"{d['date']}: {d['forecast']} [{d['low_80']}-{d['high_80']}]"
            for d in r["next_days"][:5]
        )
        return (
            f"Demand forecast for {r['horizon']} days: {_fmt(r['backtest_metrics'])} on the backtest. "
            f"Next days: {days}. Total forecast over the horizon: {r['total_forecast_units']} units. "
            f"Backend: {r['backend']}."
        )

    if tool == "benchmark_kv_cache":
        return (
            f"KV cache on {r['dataset']}: fit {r['fit_without_cache_s']}s -> {r['fit_with_cache_s']}s, "
            f"predictions over {r['n_batches']} batches {r['predict_without_cache_s']}s -> "
            f"{r['predict_with_cache_s']}s ({r['speedup_predict']}x faster predictions, "
            f"{r['speedup_total']}x end to end). Backend: {r['backend']}."
        )

    return json.dumps(r, default=str)[:800]


def make_planner():
    if llm_configured():
        try:
            return LLMPlanner()
        except Exception:
            return ScriptPlanner()
    return ScriptPlanner()
