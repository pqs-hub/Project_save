#!/usr/bin/env python3
"""Prepare Round10 isolated typed-return ranking adapter configs."""

from __future__ import annotations

import json
from pathlib import Path

from prepare_counterfactual_round5 import validate_prefix_oracle


LOOP = Path("autoresearch/loop-260720-0945")
ROUND5 = LOOP / "model_training_round5"
ROUND7 = LOOP / "model_training_round7"
ROUND8 = LOOP / "model_training_round8"
ROUND9 = LOOP / "model_training_round9"
ROOT = LOOP / "model_training_round10"
BASE = ROUND8 / "configs/moe_joint_within.json"
INIT = ROUND8 / "runs/moe_joint_within/best_final_horizon.pt"
INITIAL_ORACLE = Path(
    "autoresearch/oracle-balanced-negative-rich-260629-wide/balanced_train_oracle_actions.tsv"
)
SHORT_ORACLE = ROUND5 / "onpolicy_prefix_oracle/oracle_actions.tsv"
LONG_ORACLE = ROUND7 / "long_prefix_oracle/oracle_actions.tsv"
LATE_ORACLE = ROUND9 / "late_prefix_oracle/oracle_actions.tsv"


VARIANTS = {
    "return_within_lr5e5": {
        "lr": 5.0e-5,
        "lambda_aux_rank": 0.0,
        "lambda_aux_same_type_rank": 1.0,
    },
    "return_within_lr1e4": {
        "lr": 1.0e-4,
        "lambda_aux_rank": 0.0,
        "lambda_aux_same_type_rank": 1.0,
    },
    "return_dual_lr5e5": {
        "lr": 5.0e-5,
        "lambda_aux_rank": 0.25,
        "lambda_aux_same_type_rank": 1.0,
    },
    "return_within_lr2e4": {
        "lr": 2.0e-4,
        "lambda_aux_rank": 0.0,
        "lambda_aux_same_type_rank": 1.0,
    },
    "return_within_lr5e4": {
        "lr": 5.0e-4,
        "lambda_aux_rank": 0.0,
        "lambda_aux_same_type_rank": 1.0,
    },
    "return_dual_lr2e4": {
        "lr": 2.0e-4,
        "lambda_aux_rank": 0.25,
        "lambda_aux_same_type_rank": 1.0,
    },
}


def main() -> None:
    if not INIT.is_file():
        raise FileNotFoundError(INIT)
    long_manifest = validate_prefix_oracle(
        LONG_ORACLE,
        expected_plans_dir=ROUND7 / "long_onpolicy_plans",
        expected_prefix_steps=[32, 48, 64, 96, 127],
    )
    late_manifest = validate_prefix_oracle(
        LATE_ORACLE,
        expected_plans_dir=ROUND9 / "late_onpolicy_plans",
        expected_prefix_steps=[144, 176, 208, 240, 255],
    )
    print(
        "validated return-rank oracle "
        f"long_states={long_manifest['state_count']} "
        f"late_states={late_manifest['state_count']} "
        f"late_candidates={late_manifest['candidate_evaluations']}"
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
                "seed": 2128,
                "epochs": 8,
                "oracle_actions": [
                    {"path": str(INITIAL_ORACLE), "repeat": 1},
                    {"path": str(SHORT_ORACLE), "repeat": 1},
                    {"path": str(LONG_ORACLE), "repeat": 2},
                    {"path": str(LATE_ORACLE), "repeat": 4},
                ],
                "oracle_ranking_score_field": "typed_marginal_pred",
                "oracle_aux_ranking_score_field": "typed_return_pred",
                "oracle_group_sampling": "best_type_balanced",
                "oracle_prefix_detach": True,
                # The adapter is the sole trainable path.  Primary-head and
                # value/list losses stay off so the Round8 incumbent remains
                # bit-identical outside typed_return_pred.
                "lambda_typed_marginal": 0.0,
                "lambda_typed_return": 0.0,
                "lambda_typed_sa_reduction": 0.0,
                "lambda_q_value": 0.0,
                "lambda_q_rank": 0.0,
                "lambda_same_type_rank": 0.0,
                "lambda_candidate": 0.0,
                "lambda_context_rank": 0.0,
                "lambda_ndcg_rank": 0.0,
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
        "round": 10,
        "purpose": "isolated typed-return candidate ranking with real ATPG marginals",
        "selection_circuit": "b15_C only",
        "validation_circuits": ["b15_C", "b20_C", "b21_C", "b22_C", "b17_C"],
        "init_checkpoint": str(INIT),
        "target_circuits_in_training": False,
        "long_oracle_states": long_manifest["state_count"],
        "late_oracle_states": late_manifest["state_count"],
        "late_oracle_candidates": late_manifest["candidate_evaluations"],
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
