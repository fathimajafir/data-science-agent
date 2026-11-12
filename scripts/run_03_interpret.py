#!/usr/bin/env python
"""Lesson 3: explaining predictions."""

from _common import announce

announce("Lesson 3 - interpretation")

from tabular_agent import pipeline  # noqa: E402

res = pipeline.step3_interpretation()
for f in res["global_importance"][:6]:
    print(f"{f['feature']:<10} {f['importance']:.4f}  {f['direction']}")
print("\nwrote reports/03-interpretation.md")
