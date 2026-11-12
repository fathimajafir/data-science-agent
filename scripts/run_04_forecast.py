#!/usr/bin/env python
"""Lesson 4: forecasting by reframing as tabular regression."""

from _common import announce

announce("Lesson 4 - forecasting")

from tabular_agent import pipeline  # noqa: E402

bt = pipeline.step4_forecasting()
print({k: round(v, 4) for k, v in bt["metrics"].items()})
print("\nwrote reports/04-forecasting.md")
