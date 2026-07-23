#!/usr/bin/env python3
"""Prepare action-type-balanced cone-head continuation training."""

from __future__ import annotations

import json
from pathlib import Path

from prepare_counterfactual_round5 import validate_prefix_oracle


ROUND5 = Path("autoresearch/loop-260720-0945/model_training_round5")
ROUND6 = Path("autoresearch/loop-260720-0945/model_training_round6")
ROOT = Path("autoresearch/loop-260720-0945/model_training_round6_type_balanced")
PREFIX_ORACLE = ROUND5 / "onpolicy_prefix_oracle/oracle_actions.tsv"
INITIAL_ORACLE = Path("autoresearch/oracle-balanced-negative-rich-260629-wide/balanced_train_oracle_actions.tsv")

VARIANTS = {
    "type_balanced_rank": {
        "base": ROUND6 / "configs/cone_rank.json",
        "init": ROUND6 / "runs/cone_rank/epoch_012.pt",
        "lambda_q_value": 0.12,
        "lambda_q_rank": 0.70,
        "lambda_candidate": 0.15,
        "lambda_ndcg_rank": 0.0,
        "lambda_context_rank": 0.25,
        "lambda_conservative_q": 0.0,
    },
    "type_balanced_toplist": {
        "base": ROUND6 / "configs/cone_toplist.json",
        "init": ROUND6 / "runs/cone_toplist/epoch_012.pt",
        "lambda_q_value": 0.12,
        "lambda_q_rank": 0.50,
        "lambda_candidate": 0.20,
        "lambda_ndcg_rank": 0.20,
        "lambda_context_rank": 0.20,
        "lambda_conservative_q": 0.05,
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
    config_dir = ROOT / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    for name, spec in VARIANTS.items():
        config = json.loads(Path(spec["base"]).read_text())
        config.update(
            {
                "run_dir": str(ROOT / "runs" / name),
                "init_checkpoint": str(spec["init"]),
                "init_checkpoint_strict": True,
                "seed": 2111,
                "epochs": 8,
                "lr": 5.0e-5,
                "utility_head_type": "typed_cone_film",
                "trainable_modules": "typed_utility_only",
                "lambda_typed_marginal": 0.60,
                "lambda_typed_return": 0.05,
                "lambda_typed_sa_reduction": 0.0,
                "lambda_oracle_sa_value": 0.0,
                "oracle_group_sampling": "best_type_balanced",
                "oracle_actions": [
                    {"path": str(INITIAL_ORACLE), "repeat": 1},
                    {"path": str(PREFIX_ORACLE), "repeat": 3},
                ],
                "oracle_ranking_score_field": "typed_marginal_pred",
                "device": "cuda",
            }
        )
        config.update(
            {
                key: value
                for key, value in spec.items()
                if key not in {"base", "init"}
            }
        )
        path = config_dir / f"{name}.json"
        path.write_text(json.dumps(config, indent=2) + "\n")
        print(path)


if __name__ == "__main__":
    main()

