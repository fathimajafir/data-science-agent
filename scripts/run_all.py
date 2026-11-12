#!/usr/bin/env python
"""Run all five lessons and write reports/summary.md."""

import argparse
import sys

from _common import announce


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-agent", action="store_true", help="skip lesson 5")
    args = ap.parse_args()

    announce("Full run")
    from tabular_agent import pipeline

    steps = [
        ("1 predictions", pipeline.step1_predictions),
        ("2 kv cache", pipeline.step2_kv_cache),
        ("3 interpretation", pipeline.step3_interpretation),
        ("4 forecasting", pipeline.step4_forecasting),
    ]
    if not args.skip_agent:
        steps.append(("5 agent", pipeline.step5_agent))

    for name, fn in steps:
        print(f"\n--- {name} ---", flush=True)
        fn()
        print("done", flush=True)

    print("\nsummary:", pipeline.write_summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
