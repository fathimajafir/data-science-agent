#!/usr/bin/env python
"""Lesson 2: what the KV cache buys you."""

import json

from _common import announce

announce("Lesson 2 - KV cache")

from tabular_agent import pipeline  # noqa: E402

out = pipeline.step2_kv_cache()
print(json.dumps({k: {kk: v[kk] for kk in ("without", "with", "speedup_predict", "speedup_total")} for k, v in out.items()}, indent=2))
print("\nwrote reports/02-kv-cache.md")
