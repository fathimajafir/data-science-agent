"""Central configuration and paths for the tabular-agent project."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PREPARED_DIR = DATA_DIR / "prepared"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for _d in (RAW_DIR, PREPARED_DIR, REPORTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

# Checkpoint choice. v3.5 is the strongest; v3.5-fast trades a little accuracy
# for much lower latency, which matters on a laptop GPU.
TABPFN_MODEL = os.getenv("TABPFN_MODEL", "v3.5-fast")
N_ESTIMATORS = os.getenv("TABPFN_N_ESTIMATORS", "8")
DEVICE = os.getenv("TABPFN_DEVICE", "auto")

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


def llm_configured() -> bool:
    return bool(LLM_API_KEY.strip())
