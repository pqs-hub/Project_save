#!/usr/bin/env python3
"""Emit a scalar optimization score for exact-legal ITC99 DeepTPI comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit successfully only when every circuit and the macro average beat DeepTPI.",
    )
    args = parser.parse_args()

    payload = json.loads(args.summary.read_text())
    rows = payload["per_circuit"]
    gaps = [float(row["gap_vs_deeptpi_pp"]) for row in rows]
    macro_tc = float(payload["aggregate"]["macro_filtered_final_tc_pct"])
    macro_gap = float(payload["aggregate"]["macro_gap_vs_deeptpi_pp"])

    if args.check:
        passed = all(gap > 0.0 for gap in gaps) and macro_gap > 0.0
        print("PASS" if passed else "FAIL")
        raise SystemExit(0 if passed else 1)

    # Each remaining percentage-point deficit dominates the small macro-TC
    # tiebreaker. This keeps every lagging circuit visible to the optimizer.
    deficit = sum(max(0.0, -gap) for gap in gaps)
    print(f"{-deficit + macro_tc / 1000.0:.9f}")


if __name__ == "__main__":
    main()
