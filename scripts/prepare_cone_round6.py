#!/usr/bin/env python3
"""Prepare action-cone typed-head configs after validating Round5 ATPG provenance."""

from __future__ import annotations

import json
from pathlib import Path

from prepare_counterfactual_round5 import validate_prefix_oracle


ROUND5 = Path("autoresearch/loop-260720-0945/model_training_round5")
ROOT = Path("autoresearch/loop-260720-0945/model_training_round6")
BASE = Path("autoresearch/loop-260720-0945/model_training_round4/configs/onpolicy_balanced.json")
INIT = Path("autoresearch/loop-260720-0945/model_training_round4/runs/onpolicy_balanced/best_final_horizon.pt")
PREFIX_ORACLE = ROUND5 / "onpolicy_prefix_oracle/oracle_actions.tsv"
INITIAL_ORACLE = Path("autoresearch/oracle-balanced-negative-rich-260629-wide/balanced_train_oracle_actions.tsv")
LABELS = ROUND5 / "../model_training_round4/onpolicy_real_labels/labels.csv"

VARIANTS = {
    "cone_rank": {
        "lambda_q_rank": 0.65,
        "lambda_candidate": 0.15,
        "lambda_ndcg_rank": 0.0,
        "lambda_context_rank": 0.25,
        "lambda_conservative_q": 0.0,
        "lambda_oracle_sa_value": 0.0,
    },
    "cone_toplist": {
        "lambda_q_rank": 0.45,
        "lambda_candidate": 0.20,
        "lambda_ndcg_rank": 0.25,
        "lambda_context_rank": 0.15,
        "lambda_conservative_q": 0.10,
        "lambda_oracle_sa_value": 0.0,
    },
    "cone_toplist_sa": {
        "lambda_q_rank": 0.45,
        "lambda_candidate": 0.20,
        "lambda_ndcg_rank": 0.25,
        "lambda_context_rank": 0.15,
        "lambda_conservative_q": 0.10,
        "lambda_oracle_sa_value": 0.15,
    },
}


def main() -> None:
    manifest = validate_prefix_oracle(
        PREFIX_ORACLE,
        expected_plans_dir=ROUND5 / "onpolicy_plans",
    )
    print(
        "validated prefix oracle "
        f"states={manifest['state_count']} candidates={manifest['candidate_evaluations']}"
    )
    base = json.loads(BASE.read_text())
    config_dir = ROOT / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    for name, overrides in VARIANTS.items():
        config = dict(base)
        config.update(
            {
                "labels": str(LABELS),
                "run_dir": str(ROOT / "runs" / name),
                "init_checkpoint": str(INIT),
                "init_checkpoint_strict": False,
                "seed": 2106,
                "epochs": 12,
                "lr": 1.0e-4,
                "utility_head_type": "typed_cone_film",
                "trainable_modules": "typed_utility_only",
                "oracle_actions": [
                    {"path": str(INITIAL_ORACLE), "repeat": 1},
                    {"path": str(PREFIX_ORACLE), "repeat": 3},
                ],
                "oracle_ranking_score_field": "typed_marginal_pred",
                "oracle_pairwise_mode": "all",
                "oracle_max_pairs_per_group": 64,
                "oracle_batch_groups": 8,
                "oracle_every_n_steps": 1,
                "oracle_pairwise_min_delta": 0.0005,
                "oracle_pairwise_temperature": 0.5,
                "candidate_target_temperature": 0.25,
                "candidate_pred_temperature": 1.0,
                "oracle_ndcg_k": 3,
                "oracle_ndcg_target_temperature": 0.15,
                "oracle_ndcg_pred_temperature": 1.0,
                "oracle_context_top_weight": 0.65,
                "oracle_prefix_detach": True,
                "device": "cuda",
            }
        )
        config.update(overrides)
        path = config_dir / f"{name}.json"
        path.write_text(json.dumps(config, indent=2) + "\n")
        print(path)


if __name__ == "__main__":
    main()
