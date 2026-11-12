#!/usr/bin/env python
"""Lesson 1: classification and regression with TabPFN."""

import json

from _common import announce

announce("Lesson 1 - predictions")

from tabular_agent import pipeline  # noqa: E402

out = pipeline.step1_predictions()
print(json.dumps(out, indent=2)[:2500])
print("\nwrote reports/01-predictions.md")
