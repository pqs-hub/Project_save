#!/usr/bin/env python3
"""Select one uniform residual-refresh step using only final b15 coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def incumbent_delta(payload: dict) -> float:
    record = payload.get("winner", payload)
    if "b15_delta_tc" in record:
        return float(record["b15_delta_tc"])
    selected = record.get("selected")
    if selected and "delta_test_coverage" in selected:
        return float(selected["delta_test_coverage"])
    if "delta_test_coverage" in record:
        return float(record["delta_test_coverage"])
    raise KeyError("incumbent manifest has no b15 delta-TC field")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--incumbent-manifest", type=Path, required=True)
    parser.add_argument("--steps", default="64,96,128,160,192,224")
    args = parser.parse_args()

    incumbent = json.loads(args.incumbent_manifest.read_text())
    incumbent_value = incumbent_delta(incumbent)
    records = []
    for step in [int(item) for item in args.steps.split(",") if item.strip()]:
        summary_path = args.base / "evals" / f"step_{step:03d}" / "summary.json"
        rows = json.loads(summary_path.read_text())
        final = rows[-1]
        records.append(
            {
                "refresh_step": step,
                "test_coverage": float(final["test_coverage"]),
                "delta_test_coverage": float(final["delta_test_coverage"]),
                "summary_json": str(summary_path),
                "plan_csv": str(args.base / "plans" / f"step_{step:03d}.csv"),
                "residual_prior_csv": str(
                    args.base / "residual_priors" / f"step_{step:03d}" / "real_fault_priors.csv"
                ),
            }
        )

    best = max(records, key=lambda row: (row["delta_test_coverage"], -row["refresh_step"]))
    strict_win = best["delta_test_coverage"] > incumbent_value + 1e-12
    payload = {
        "selection_circuit": "iscas99__b15_1",
        "selection_metric": "final_delta_test_coverage",
        "tie_break": "retain incumbent; among new strict winners choose earliest refresh",
        "incumbent_delta_test_coverage": incumbent_value,
        "strict_win": strict_win,
        "selected": best if strict_win else None,
        "incumbent_manifest": str(args.incumbent_manifest),
        "records": records,
    }
    args.base.mkdir(parents=True, exist_ok=True)
    (args.base / "b15_selection.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
