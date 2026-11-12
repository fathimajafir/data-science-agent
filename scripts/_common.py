"""Shared bootstrap for the runnable scripts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tabular_agent import backends  # noqa: E402


def announce(title: str) -> None:
    print(f"\n=== {title} ===")
    print(backends.backend_label())
    if backends.resolve_backend() == "fallback":
        print(
            "WARNING: TabPFN is not reachable. Set TABPFN_TOKEN in .env "
            "(https://platform.priorlabs.ai/account/api-keys) to run the real model; "
            "these numbers are sklearn placeholders."
        )
