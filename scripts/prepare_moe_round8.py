#!/usr/bin/env python3
"""Prepare Round8 within-type ranking experiments from the frozen Round7 winner."""

from __future__ import annotations

import json
from pathlib import Path

from prepare_counterfactual_round5 import validate_prefix_oracle


LOOP = Path("autoresearch/loop-260720-0945")
ROUND5 = LOOP / "model_training_round5"
ROUND7 = LOOP / "model_training_round7"
ROOT = LOOP / "model_training_round8"
BASE = ROUND7 / "configs/cone_long_rank.json"
INIT = ROUND7 / "runs/cone_long_rank/best_final_horizon.pt"
INITIAL_ORACLE = Path(
    "autoresearch/oracle-balanced-negative-rich-260629-wide/balanced_train_oracle_actions.tsv"
)
SHORT_ORACLE = ROUND5 / "onpolicy_prefix_oracle/oracle_actions.tsv"
LONG_ORACLE = ROUND7 / "long_prefix_oracle/oracle_actions.tsv"


VARIANTS = {
    # A bounded residual experiment: every parameter used by the deployed
    # Round7 shared head stays bit-identical and only the new experts can move.
    "moe_experts_within": {
        "trainable_modules": "typed_experts_only",
        "lr": 1.0e-4,
        "lambda_typed_marginal": 0.20,
        "lambda_typed_return": 0.02,
        "lambda_q_value": 0.05,
        "lambda_q_rank": 0.20,
        "lambda_same_type_rank": 1.00,
        "lambda_candidate": 0.05,
        "lambda_context_rank": 0.05,
    },
    # A less constrained ablation tests whether the same objective needs to
    # reshape the shared cone representation as well as the expert residuals.
    "moe_joint_within": {
        "trainable_modules": "typed_utility_only",
        "lr": 5.0e-5,
        "lambda_typed_marginal": 0.60,
        "lambda_typed_return": 0.05,
        "lambda_q_value": 0.10,
        "lambda_q_rank": 0.50,
        "lambda_same_type_rank": 0.80,
        "lambda_candidate": 0.10,
        "lambda_context_rank": 0.15,
    },
}


def main() -> None:
    if not INIT.is_file():
        raise FileNotFoundError(INIT)
    manifest = validate_prefix_oracle(
        LONG_ORACLE,
        expected_plans_dir=ROUND7 / "long_onpolicy_plans",
        expected_prefix_steps=[32, 48, 64, 96, 127],
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
                "init_checkpoint": str(INIT),
                "init_checkpoint_strict": False,
                "utility_head_type": "typed_cone_moe",
                "seed": 2121,
                "epochs": 10,
                "oracle_actions": [
                    {"path": str(INITIAL_ORACLE), "repeat": 1},
                    {"path": str(SHORT_ORACLE), "repeat": 1},
                    {"path": str(LONG_ORACLE), "repeat": 3},
                ],
                "oracle_ranking_score_field": "typed_marginal_pred",
                "oracle_group_sampling": "best_type_balanced",
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
        "round": 8,
        "purpose": "improve within-action-type node ordering",
        "selection_circuit": "b15_C only",
        "validation_circuits": ["b15_C", "b20_C", "b21_C", "b22_C", "b17_C"],
        "init_checkpoint": str(INIT),
        "target_circuits_in_training": False,
        "long_oracle_states": manifest["state_count"],
        "long_oracle_candidates": manifest["candidate_evaluations"],
        "variants": VARIANTS,
    }
    (ROOT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
