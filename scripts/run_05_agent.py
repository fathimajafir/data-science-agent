#!/usr/bin/env python
"""Lesson 5: the data science agent.

Usage:
    python scripts/run_05_agent.py
    python scripts/run_05_agent.py "Forecast demand for the next 2 weeks"
"""

import sys

from _common import announce

announce("Lesson 5 - agent")

from tabular_agent import agent, pipeline  # noqa: E402

questions = sys.argv[1:]
if questions:
    for q in questions:
        res = agent.run_agent(q)
        print(
            f"\nQ: {q}\nA: {res.answer}\n"
            f"   ({res.planner}, {res.steps} steps, {res.seconds:.1f}s)"
        )
else:
    for r in pipeline.step5_agent():
        print(
            f"\nQ: {r['question']}\nA: {r['answer']}\n"
            f"   ({r['planner']}, {r['steps']} steps, {r['seconds']}s)"
        )
    print("\nwrote reports/05-agent.md")
