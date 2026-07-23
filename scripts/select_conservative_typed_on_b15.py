#!/usr/bin/env python3
"""Select one frozen-base typed residual and alpha using b15 only."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "autoresearch/loop-260720-0945/model_training_round2"
SELECT = BASE / "b15_selection"
RUNS = BASE / "runs"


def main() -> None:
    candidates: list[dict] = []
    for path in sorted(SELECT.glob("*/alpha_*/results.tsv")):
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        ok = [row for row in rows if row.get("status") == "ok"]
        if len(ok) != 1:
            raise RuntimeError(f"expected one successful b15 result in {path}, got {len(ok)}")
        row = ok[0]
        variant = path.parents[1].name
        config = json.loads((path.parent / "config.json").read_text())
        env = config.get("planner_environment") or {}
        alpha = float(env["TPI_TYPED_RESIDUAL_ALPHA"])
        candidates.append(
            {
                "variant": variant,
                "checkpoint": str((RUNS / variant / "best_final_horizon.pt").relative_to(ROOT)),
                "score_field": "q_typed_residual_context",
                "typed_residual_alpha": alpha,
                "b15_delta_tc": float(row["delta_test_coverage"]),
                "b15_plan_csv": row["plan_csv"],
                "b15_eval_dir": row["eval_dir"],
            }
        )
    if not candidates:
        raise RuntimeError(f"no b15 results found under {SELECT}")
    candidates.sort(
        key=lambda item: (
            -item["b15_delta_tc"],
            item["typed_residual_alpha"],
            item["variant"],
        )
    )
    manifest = {
        "selection_circuit": "b15_C",
        "selection_benchmark": "iscas99__b15_1",
        "rule": "maximize standardized b15 delta TC; prefer smaller residual alpha on ties; inspect no other final circuit",
        "winner": candidates[0],
        "candidates": candidates,
    }
    out = SELECT / "winner.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["winner"], indent=2))


if __name__ == "__main__":
    main()
