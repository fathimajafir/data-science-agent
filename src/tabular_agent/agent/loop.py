"""The agent loop: request -> tool calls -> grounded answer."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .llm import make_planner
from .tools import ToolRegistry, backend_notice


@dataclass
class AgentResult:
    answer: str
    trace: list[dict] = field(default_factory=list)
    steps: int = 0
    seconds: float = 0.0
    planner: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "steps": self.steps,
            "seconds": round(self.seconds, 2),
            "planner": self.planner,
            "trace": self.trace,
        }


def run_agent(
    request: str,
    *,
    planner=None,
    registry: ToolRegistry | None = None,
    max_steps: int = 6,
) -> AgentResult:
    registry = registry or ToolRegistry()
    planner = planner or make_planner()
    planner.start(request, registry.schemas())

    t0 = time.perf_counter()
    trace: list[dict] = []
    result: dict | None = None

    for step in range(1, max_steps + 1):
        decision = planner.next(result)

        if decision["type"] == "final":
            return AgentResult(
                answer=decision["text"].strip(),
                trace=trace,
                steps=step,
                seconds=time.perf_counter() - t0,
                planner=type(planner).__name__,
            )

        name = decision["name"]
        args = decision.get("args") or {}
        t_tool = time.perf_counter()
        result = registry.call(name, args)
        trace.append(
            {
                "step": step,
                "tool": name,
                "args": args,
                "seconds": round(time.perf_counter() - t_tool, 2),
                "result": result,
            }
        )

    return AgentResult(
        answer=(
            "Stopped after "
            f"{max_steps} steps without a final answer. Last tool result: "
            f"{str(result)[:400]}"
        ),
        trace=trace,
        steps=max_steps,
        seconds=time.perf_counter() - t0,
        planner=type(planner).__name__,
    )


def banner() -> str:
    return backend_notice()
