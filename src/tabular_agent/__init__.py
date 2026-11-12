"""Data science agent powered by tabular foundation models (TabPFN).

Public API:
    run_agent          ask the agent a question
    ToolRegistry       the tools the agent can call
    backends           api / local / fallback resolution
    pipeline           the five lessons as report-writing steps
"""

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "agent",
    "backends",
    "cache_bench",
    "config",
    "core",
    "data",
    "forecasting",
    "interpret",
    "pipeline",
]
