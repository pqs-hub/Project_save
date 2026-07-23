#!/usr/bin/env python3
"""Prepare long-prefix action-cone training configs with strict provenance."""

from __future__ import annotations

import json
from pathlib import Path

from prepare_counterfactual_round5 import validate_prefix_oracle


ROUND5 = Path("autoresearch/loop-260720-0945/model_training_round5")
ROUND6 = Path("autoresearch/loop-260720-0945/model_training_round6")
ROUND6_TYPE = Path("autoresearch/loop-260720-0945/model_training_round6_type_balanced")
ROOT = Path("autoresearch/loop-260720-0945/model_training_round7")
LONG_PLANS = ROOT / "long_onpolicy_plans"
LONG_ORACLE = ROOT / "long_prefix_oracle/oracle_actions.tsv"
LONG_STEPS = [32, 48, 64, 96, 127]
BASE = ROUND6 / "configs/cone_rank.json"
INITIAL_ORACLE = Path("autoresearch/oracle-balanced-negative-rich-260629-wide/balanced_train_oracle_actions.tsv")
SHORT_ORACLE = ROUND5 / "onpolicy_prefix_oracle/oracle_actions.tsv"

VARIANTS = {
    "cone_long_rank": {
        "init_checkpoint": str(ROUND6_TYPE / "runs/type_balanced_rank/epoch_008.pt"),
        "lambda_q_value": 0.12,
        "lambda_q_rank": 0.70,
        "lambda_candidate": 0.15,
        "lambda_ndcg_rank": 0.0,
        "lambda_context_rank": 0.25,
        "lambda_conservative_q": 0.0,
    },
    "cone_long_toplist": {
        "init_checkpoint": str(ROUND6_TYPE / "runs/type_balanced_toplist/epoch_008.pt"),
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
        LONG_ORACLE,
        expected_plans_dir=LONG_PLANS,
        expected_prefix_steps=LONG_STEPS,
    )
    print(
        "validated long-prefix oracle "
        f"states={manifest['state_count']} candidates={manifest['candidate_evaluations']}"
    )
    base = json.loads(BASE.read_text())
    config_dir = ROOT / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    for name, overrides in VARIANTS.items():
        config = dict(base)
        config.update(
            {
                "run_dir": str(ROOT / "runs" / name),
                "init_checkpoint_strict": True,
                "seed": 2117,
                "epochs": 10,
                "lr": 7.5e-5,
                "utility_head_type": "typed_cone_film",
                "trainable_modules": "typed_utility_only",
                "lambda_typed_marginal": 0.60,
                "lambda_typed_return": 0.05,
                "lambda_typed_sa_reduction": 0.0,
                "lambda_oracle_sa_value": 0.0,
                "oracle_actions": [
                    {"path": str(INITIAL_ORACLE), "repeat": 1},
                    {"path": str(SHORT_ORACLE), "repeat": 1},
                    {"path": str(LONG_ORACLE), "repeat": 3},
                ],
                "oracle_ranking_score_field": "typed_marginal_pred",
                "oracle_group_sampling": "best_type_balanced",
                "device": "cuda",
            }
        )
        config.update(overrides)
        path = config_dir / f"{name}.json"
        path.write_text(json.dumps(config, indent=2) + "\n")
        print(path)


if __name__ == "__main__":
    main()
