#!/usr/bin/env python3
"""Prepare expanded late-prefix hybrid pairwise/listwise return-rank configs."""

from __future__ import annotations

import json
from pathlib import Path

from prepare_counterfactual_round5 import validate_prefix_oracle


LOOP = Path("autoresearch/loop-260720-0945")
ROUND5 = LOOP / "model_training_round5"
ROUND7 = LOOP / "model_training_round7"
ROUND8 = LOOP / "model_training_round8"
ROOT = LOOP / "model_training_round20"
BASE = LOOP / "model_training_round10/configs/return_within_lr1e4.json"
INIT = ROUND8 / "runs/moe_joint_within/best_final_horizon.pt"
INITIAL_ORACLE = Path(
    "autoresearch/oracle-balanced-negative-rich-260629-wide/balanced_train_oracle_actions.tsv"
)
SHORT_ORACLE = ROUND5 / "onpolicy_prefix_oracle/oracle_actions.tsv"
LONG_ORACLE = ROUND7 / "long_prefix_oracle/oracle_actions.tsv"
EXPANDED_LATE_ORACLE = ROOT / "late_prefix_oracle/oracle_actions.tsv"


VARIANTS = {
    "return_pairwise_expanded": {
        "lr": 5.0e-5,
        "lambda_q_rank": 0.25,
        "lambda_same_type_rank": 1.0,
        "lambda_ndcg_rank": 0.0,
        "lambda_same_type_ndcg_rank": 0.0,
    },
    "return_hybrid_listwise": {
        "lr": 5.0e-5,
        "lambda_q_rank": 0.25,
        "lambda_same_type_rank": 0.75,
        "lambda_ndcg_rank": 0.25,
        "lambda_same_type_ndcg_rank": 1.0,
    },
    "return_top_listwise": {
        "lr": 5.0e-5,
        "lambda_q_rank": 0.10,
        "lambda_same_type_rank": 0.25,
        "lambda_ndcg_rank": 0.40,
        "lambda_same_type_ndcg_rank": 1.50,
    },
}


def main() -> None:
    if not INIT.is_file():
        raise FileNotFoundError(INIT)
    late_manifest = validate_prefix_oracle(
        EXPANDED_LATE_ORACLE,
        expected_plans_dir=ROOT / "late_onpolicy_plans",
        expected_prefix_steps=[144, 176, 208, 240, 255],
        expected_actions_per_prefix=15,
    )
    print(
        "validated expanded late oracle "
        f"states={late_manifest['state_count']} "
        f"candidates={late_manifest['candidate_evaluations']}"
    )

    base = json.loads(BASE.read_text())
    config_dir = ROOT / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    for name, overrides in VARIANTS.items():
        config = dict(base)
        config.update(
            {
                "run_dir": str(ROOT / "runs" / name),
                "init_checkpoint": str(INIT),
                "init_checkpoint_strict": False,
                "utility_head_type": "typed_cone_return_rank_moe",
                "trainable_modules": "typed_return_rank_only",
                "seed": 2138,
                "epochs": 10,
                "oracle_actions": [
                    {"path": str(INITIAL_ORACLE), "repeat": 1},
                    {"path": str(SHORT_ORACLE), "repeat": 1},
                    {"path": str(LONG_ORACLE), "repeat": 2},
                    {"path": str(EXPANDED_LATE_ORACLE), "repeat": 4},
                ],
                # Train and evaluate the return head directly; only its
                # zero-initialized adapter is unfrozen.
                "oracle_ranking_score_field": "typed_return_pred",
                "oracle_aux_ranking_score_field": "typed_marginal_pred",
                "oracle_group_sampling": "best_type_balanced",
                "oracle_prefix_detach": True,
                "oracle_batch_groups": 12,
                "oracle_every_n_steps": 1,
                "oracle_pairwise_mode": "all",
                "oracle_pairwise_min_delta": 0.00025,
                "oracle_pairwise_temperature": 0.35,
                "oracle_ndcg_k": 2,
                "oracle_ndcg_target_temperature": 0.10,
                "oracle_ndcg_pred_temperature": 0.75,
                "lambda_typed_marginal": 0.0,
                "lambda_typed_return": 0.0,
                "lambda_typed_sa_reduction": 0.0,
                "lambda_q_value": 0.0,
                "lambda_aux_rank": 0.0,
                "lambda_aux_same_type_rank": 0.0,
                "lambda_candidate": 0.0,
                "lambda_context_rank": 0.0,
                "lambda_conservative_q": 0.0,
                "lambda_oracle_sa_value": 0.0,
                "device": "cuda",
            }
        )
        config.update(overrides)
        path = config_dir / f"{name}.json"
        path.write_text(json.dumps(config, indent=2) + "\n")
        print(path)

    provenance = {
        "round": 20,
        "purpose": "expanded late-prefix hybrid pairwise/listwise within-type return ranking",
        "selection_circuit": "b15_C only",
        "validation_circuits": ["b15_C", "b20_C", "b21_C", "b22_C", "b17_C"],
        "init_checkpoint": str(INIT),
        "target_circuits_in_training": False,
        "expanded_late_oracle_states": late_manifest["state_count"],
        "expanded_late_oracle_candidates": late_manifest["candidate_evaluations"],
        "frozen_predictions": [
            "typed_marginal_pred",
            "typed_sa_reduction_pred",
            "legacy world-model outputs",
        ],
        "variants": VARIANTS,
    }
    (ROOT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
