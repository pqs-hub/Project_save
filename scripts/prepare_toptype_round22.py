#!/usr/bin/env python3
"""Prepare a top-action/type-focused continuation of the Round21 adapter."""

from __future__ import annotations

import json
from pathlib import Path


LOOP = Path("autoresearch/loop-260720-0945")
ROUND21 = LOOP / "model_training_round21"
ROOT = LOOP / "model_training_round22"
SOURCE_CONFIG = ROUND21 / "configs/horizon_r10_pairwise.json"
SOURCE_CHECKPOINT = ROUND21 / "runs/horizon_r10_pairwise/epoch_010.pt"
ROUND10_CHECKPOINT = LOOP / "model_training_round10/runs/return_within_lr5e4/epoch_008.pt"
ULTRALONG = ROUND21 / "ultralong_prefix_oracle/oracle_actions.tsv"

VARIANTS = {
    "toptype_r21_hard": {
        "init_checkpoint": str(SOURCE_CHECKPOINT),
        "oracle_pairwise_mode": "best_vs_hard_topk",
    },
    "toptype_r21_all": {
        "init_checkpoint": str(SOURCE_CHECKPOINT),
        "oracle_pairwise_mode": "all",
    },
    "toptype_r10_hard": {
        "init_checkpoint": str(ROUND10_CHECKPOINT),
        "oracle_pairwise_mode": "best_vs_hard_topk",
    },
}


def main() -> None:
    for path in (SOURCE_CONFIG, SOURCE_CHECKPOINT, ROUND10_CHECKPOINT, ULTRALONG):
        if not path.is_file():
            raise FileNotFoundError(path)

    config = json.loads(SOURCE_CONFIG.read_text())
    oracle_actions = []
    for source in config["oracle_actions"]:
        row = dict(source)
        if Path(row["path"]) == ULTRALONG:
            row["repeat"] = 8
        oracle_actions.append(row)

    config.update(
        {
            "init_checkpoint_strict": False,
            "epochs": 8,
            "seed": 2150,
            "lr": 5.0e-5,
            "oracle_actions": oracle_actions,
            # The replay audit has reasonable global pairwise accuracy but
            # poor top-1/type accuracy.  Concentrate the continuation on the
            # true top actions and the model's current hard negatives.
            "oracle_positive_topk": 2,
            "oracle_hard_negative_topk": 15,
            "oracle_max_pairs_per_group": 64,
            # Round21 over-weighted within-type node order.  Let cross-type
            # comparisons dominate while retaining a small within-type term.
            "lambda_q_rank": 1.5,
            "lambda_same_type_rank": 0.1,
            "lambda_candidate": 0.25,
            "lambda_ndcg_rank": 0.25,
            "lambda_same_type_ndcg_rank": 0.0,
            "candidate_target_temperature": 0.10,
            "candidate_pred_temperature": 0.75,
            "device": "cuda",
        }
    )

    config_dir = ROOT / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_paths = []
    for name, overrides in VARIANTS.items():
        variant_config = dict(config)
        variant_config.update(overrides)
        variant_config["run_dir"] = str(ROOT / "runs" / name)
        config_path = config_dir / f"{name}.json"
        config_path.write_text(json.dumps(variant_config, indent=2) + "\n")
        config_paths.append(config_path)

    provenance = {
        "round": 22,
        "purpose": "top-action and cross-action-type hard-negative continuation",
        "selection_circuit": "b15_C only",
        "target_circuits_in_training": False,
        "variants": VARIANTS,
        "heldout_non_target_benchmarks": ["subckt_0230", "subckt_0360"],
        "ultralong_repeat": 8,
        "loss": {
            "pairwise_mode": "best_vs_hard_topk",
            "positive_topk": 2,
            "lambda_q_rank": 1.5,
            "lambda_same_type_rank": 0.1,
            "lambda_candidate": 0.25,
            "lambda_ndcg_rank": 0.25,
        },
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "training_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    for config_path in config_paths:
        print(config_path)


if __name__ == "__main__":
    main()
