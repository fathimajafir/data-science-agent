"""Command line interface: ``tabular-agent``.

Subcommands
-----------
run       Execute one lesson (or all) and write reports/
ask       Ask the data science agent a question
doctor    Diagnose installation, backend and connectivity
datasets  List the bundled datasets
serve     Start the HTTP API (requires the [serve] extra)
"""

from __future__ import annotations

import argparse
import logging
import sys

from . import __version__
from .backends import FALLBACK_LABEL, backend_label, diagnose, resolve_backend

logger = logging.getLogger("tabular_agent")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tabular-agent",
        description="Data science agent powered by tabular foundation models (TabPFN).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="verbose logging (debug level)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run one lesson or all; writes reports/")
    run.add_argument(
        "lesson",
        nargs="?",
        default="all",
        choices=["1", "2", "3", "4", "5", "all"],
        help="lesson number (1 predictions, 2 KV cache, 3 interpretation, "
        "4 forecasting, 5 agent) or 'all'",
    )
    run.add_argument("--skip-agent", action="store_true", help="skip lesson 5")

    ask = sub.add_parser("ask", help="Ask the data science agent a question")
    ask.add_argument("question", help="the request, in quotes")
    ask.add_argument(
        "--max-steps", type=int, default=6, help="tool-call budget (default 6)"
    )

    sub.add_parser("doctor", help="Diagnose installation, backend and connectivity")
    sub.add_parser("datasets", help="List the bundled datasets")

    serve = sub.add_parser("serve", help="Start the HTTP API (needs [serve] extra)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    return parser


def _print_banner(title: str) -> None:
    print(f"\n=== {title} ===")
    print(f"backend: {backend_label()}")
    if resolve_backend() == "fallback":
        print(
            f"WARNING: {FALLBACK_LABEL} is a placeholder, not TabPFN. "
            "Set TABPFN_TOKEN for real predictions."
        )


def cmd_doctor() -> int:
    import platform

    checks = diagnose()
    print(f"tabular-agent {__version__} | python {platform.python_version()}")
    for key, value in checks.items():
        if key == "hint":
            continue
        print(f"  {key:<22} {value}")
    if "hint" in checks:
        print(f"\n  hint: {checks['hint']}")
    usage = __import__("tabular_agent.backends", fromlist=["api_usage"]).api_usage()
    if usage:
        print(f"\n  API usage: {usage}")
    return 0


def cmd_datasets() -> int:
    from .agent.tools import DATASETS

    for name, spec in DATASETS.items():
        print(f"  {name:<12} {spec['task']:<15} target={spec['target']:<12} {spec['description']}")
    return 0


def cmd_run(lesson: str, skip_agent: bool) -> int:
    from . import pipeline

    _print_banner(f"Lesson {lesson}")

    steps = {
        "1": pipeline.step1_predictions,
        "2": pipeline.step2_kv_cache,
        "3": pipeline.step3_interpretation,
        "4": pipeline.step4_forecasting,
    }
    if lesson == "all":
        for n, fn in steps.items():
            print(f"\n--- lesson {n} ---", flush=True)
            fn()
        if not skip_agent:
            print("\n--- lesson 5 ---", flush=True)
            pipeline.step5_agent()
        print("\nsummary:", pipeline.write_summary())
    else:
        if lesson == "5":
            if skip_agent:
                print("nothing to do: --skip-agent with lesson 5")
                return 1
            pipeline.step5_agent()
        else:
            steps[lesson]()
    return 0


def cmd_ask(question: str, max_steps: int) -> int:
    from .agent import run_agent

    _print_banner("Agent")
    result = run_agent(question, max_steps=max_steps)
    print(f"\nQ: {question}")
    print(f"A: {result.answer}\n")
    for t in result.trace:
        print(f"  step {t['step']}: {t['tool']}({t['args']}) in {t['seconds']}s")
    print(f"  ({result.planner}, {result.steps} steps, {result.seconds:.1f}s)")
    return 0


def cmd_serve(host: str, port: int) -> int:
    try:
        import uvicorn  # noqa: F401
    except ImportError:
        print(
            "The serve command needs the [serve] extra:\n"
            "    pip install 'tabular-agent[serve]'"
        )
        return 2
    from .serve import create_app

    print(f"serving on http://{host}:{port} (backend: {backend_label()})")
    uvicorn.run(create_app(), host=host, port=port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    handlers = {
        "doctor": lambda: cmd_doctor(),
        "datasets": lambda: cmd_datasets(),
        "run": lambda: cmd_run(args.lesson, args.skip_agent),
        "ask": lambda: cmd_ask(args.question, args.max_steps),
        "serve": lambda: cmd_serve(args.host, args.port),
    }
    try:
        return handlers[args.command]()
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        logger.debug("command failed", exc_info=True)
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
