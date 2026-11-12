"""Optional HTTP API around the agent (requires the [serve] extra).

Endpoints
---------
GET  /health    liveness + active backend
GET  /tools     tool descriptions the agent can call
POST /ask       {"question": "...", "max_steps": 6} -> agent answer + trace

Run with ``tabular-agent serve`` or ``uvicorn tabular_agent.serve:app``.
Tool calls serialise on a lock, so several requests can be in flight at once
without racing on model fits.
"""

# NOTE: no `from __future__ import annotations` here on purpose. FastAPI needs
# the real class objects in endpoint signatures, not postponed string
# annotations, to recognise request-body models.
from typing import Any

from . import __version__
from .backends import backend_label, resolve_backend


def create_app() -> Any:
    from fastapi import FastAPI
    from pydantic import BaseModel, Field

    from .agent import ToolRegistry, run_agent

    app = FastAPI(
        title="tabular-agent",
        version=__version__,
        description="Data science agent powered by tabular foundation models (TabPFN).",
    )

    class AskRequest(BaseModel):
        question: str = Field(min_length=3, max_length=2000)
        max_steps: int = Field(default=6, ge=1, le=10)

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "version": __version__,
            "backend": resolve_backend(),
            "backend_label": backend_label(),
        }

    @app.get("/tools")
    def tools() -> list[dict]:
        return ToolRegistry().describe()

    @app.post("/ask")
    def ask(request: AskRequest) -> dict:
        result = run_agent(request.question, max_steps=request.max_steps)
        return result.to_dict()

    return app


def factory() -> Any:
    """ASGI factory for ``uvicorn tabular_agent.serve:factory``."""
    return create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=8000)
