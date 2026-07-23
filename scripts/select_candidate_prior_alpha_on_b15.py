#!/usr/bin/env python3
"""Select residual-candidate score fusion strength using only final b15 TC."""

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


def alpha_tag(alpha: float) -> str:
    return f"alpha_{alpha:.3f}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--incumbent-manifest", type=Path, required=True)
    parser.add_argument("--alphas", default="0,0.01,0.03,0.07,0.15,0.30")
    args = parser.parse_args()

    incumbent_payload = json.loads(args.incumbent_manifest.read_text())
    incumbent = incumbent_delta(incumbent_payload)
    records = []
    for alpha in [float(item) for item in args.alphas.split(",") if item.strip()]:
        tag = alpha_tag(alpha)
        summary_path = args.base / "evals" / tag / "summary.json"
        final = json.loads(summary_path.read_text())[-1]
        records.append(
            {
                "candidate_prior_alpha": alpha,
                "test_coverage": float(final["test_coverage"]),
                "delta_test_coverage": float(final["delta_test_coverage"]),
                "summary_json": str(summary_path),
                "plan_csv": str(args.base / "plans" / f"{tag}.csv"),
            }
        )

    best = max(records, key=lambda row: (row["delta_test_coverage"], -row["candidate_prior_alpha"]))
    strict_win = best["delta_test_coverage"] > incumbent + 1e-12
    payload = {
        "selection_circuit": "iscas99__b15_1",
        "selection_metric": "final_delta_test_coverage",
        "tie_break": "retain incumbent; among new strict winners choose smallest candidate-prior alpha",
        "incumbent_delta_test_coverage": incumbent,
        "strict_win": strict_win,
        "selected": best if strict_win else None,
        "incumbent_manifest": str(args.incumbent_manifest),
        "records": records,
    }
    (args.base / "b15_selection.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
