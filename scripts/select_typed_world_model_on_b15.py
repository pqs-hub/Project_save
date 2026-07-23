#!/usr/bin/env python3
"""Select one frozen checkpoint/score field using b15_C results only."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = ROOT / "autoresearch/loop-260720-0945/model_training_round1/b15_selection"
RUN_ROOT = ROOT / "autoresearch/loop-260720-0945/model_training_round1/runs"
VARIANTS = ("frozen_balanced", "frozen_long", "joint_low_lr")


def main() -> None:
    candidates: list[dict[str, object]] = []
    for variant in VARIANTS:
        path = EVAL_ROOT / variant / "results.tsv"
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row["status"] != "ok":
                    continue
                candidates.append(
                    {
                        "variant": variant,
                        "checkpoint": str(
                            (RUN_ROOT / variant / "best_final_horizon.pt").relative_to(ROOT)
                        ),
                        "score_field": row["score_field"],
                        "b15_delta_tc": float(row["delta_test_coverage"]),
                        "b15_plan_csv": row["plan_csv"],
                        "b15_eval_dir": row["eval_dir"],
                    }
                )
    if not candidates:
        raise RuntimeError("no successful b15-only selection rows")
    candidates.sort(
        key=lambda item: (
            float(item["b15_delta_tc"]),
            str(item["variant"]),
            str(item["score_field"]),
        ),
        reverse=True,
    )
    payload = {
        "selection_circuit": "b15_C",
        "selection_benchmark": "iscas99__b15_1",
        "rule": "maximize standardized b15 delta TC; no other final circuit is inspected",
        "winner": candidates[0],
        "candidates": candidates,
    }
    output = EVAL_ROOT / "winner.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["winner"], indent=2))


if __name__ == "__main__":
    main()
