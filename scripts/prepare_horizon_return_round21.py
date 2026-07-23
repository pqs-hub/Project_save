#!/usr/bin/env python3
"""Prepare leak-free explicit-horizon return-adapter training configs."""

from __future__ import annotations

import json
from pathlib import Path

from prepare_counterfactual_round5 import validate_prefix_oracle


LOOP = Path("autoresearch/loop-260720-0945")
ROOT = LOOP / "model_training_round21"
BASE = LOOP / "model_training_round20/configs/return_pairwise_expanded.json"
ROUND8_INIT = LOOP / "model_training_round8/runs/moe_joint_within/best_final_horizon.pt"
ROUND10_INIT = LOOP / "model_training_round10/runs/return_within_lr5e4/epoch_008.pt"
INITIAL_ORACLE = Path(
    "autoresearch/oracle-balanced-negative-rich-260629-wide/balanced_train_oracle_actions.tsv"
)
SHORT_ORACLE = LOOP / "model_training_round5/onpolicy_prefix_oracle/oracle_actions.tsv"
LONG_ORACLE = LOOP / "model_training_round7/long_prefix_oracle/oracle_actions.tsv"
EXPANDED_ORACLE = LOOP / "model_training_round20/late_prefix_oracle/oracle_actions.tsv"
ULTRALONG_ORACLE = ROOT / "ultralong_prefix_oracle/oracle_actions.tsv"

# These two non-target circuits span the low-coverage and largest-graph ends
# of the ultra-long collection.  They are held out from every oracle source so
# a later audit measures genuine horizon transfer rather than memorization.
HOLDOUT_BENCHMARKS = ["subckt_0360", "subckt_0230"]

PAIRWISE = {
    "lambda_q_rank": 0.25,
    "lambda_same_type_rank": 1.0,
    "lambda_ndcg_rank": 0.0,
    "lambda_same_type_ndcg_rank": 0.0,
}
HYBRID = {
    "lambda_q_rank": 0.25,
    "lambda_same_type_rank": 0.75,
    "lambda_ndcg_rank": 0.25,
    "lambda_same_type_ndcg_rank": 1.0,
}
VARIANTS = {
    "horizon_r8_pairwise": {"init_checkpoint": str(ROUND8_INIT), **PAIRWISE},
    "horizon_r8_hybrid": {"init_checkpoint": str(ROUND8_INIT), **HYBRID},
    "horizon_r10_pairwise": {"init_checkpoint": str(ROUND10_INIT), **PAIRWISE},
    "horizon_r10_hybrid": {"init_checkpoint": str(ROUND10_INIT), **HYBRID},
}


def main() -> None:
    for path in (BASE, ROUND8_INIT, ROUND10_INIT):
        if not path.is_file():
            raise FileNotFoundError(path)
    ultra_manifest = validate_prefix_oracle(
        ULTRALONG_ORACLE,
        expected_plans_dir=ROOT / "ultralong_onpolicy_plans",
        expected_prefix_steps=[320, 448, 576, 704, 767],
        expected_actions_per_prefix=15,
    )
    if int(ultra_manifest.get("patterns") or 0) != 300_000:
        raise ValueError(f"unexpected pattern count: {ultra_manifest.get('patterns')}")
    print(
        "validated ultra-long oracle "
        f"states={ultra_manifest['state_count']} "
        f"candidates={ultra_manifest['candidate_evaluations']}"
    )

    base = json.loads(BASE.read_text())
    config_dir = ROOT / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    for name, overrides in VARIANTS.items():
        config = dict(base)
        forbidden = set(config.get("oracle_forbidden_benchmarks") or [])
        forbidden.update(HOLDOUT_BENCHMARKS)
        config.update(
            {
                "run_dir": str(ROOT / "runs" / name),
                "init_checkpoint_strict": False,
                "utility_head_type": "typed_cone_return_horizon_rank_moe",
                "trainable_modules": "typed_return_horizon_only",
                "seed": 2149,
                "epochs": 10,
                "lr": 5.0e-5,
                # Frozen checkpoint paths do not need training-time dropout;
                # planner evaluation was already deterministic/eval-mode.
                "dropout": 0.0,
                "oracle_actions": [
                    {"path": str(INITIAL_ORACLE), "repeat": 1},
                    {"path": str(SHORT_ORACLE), "repeat": 1},
                    {"path": str(LONG_ORACLE), "repeat": 1},
                    {"path": str(EXPANDED_ORACLE), "repeat": 2},
                    {"path": str(ULTRALONG_ORACLE), "repeat": 4},
                ],
                "oracle_forbidden_benchmarks": sorted(forbidden),
                "oracle_ranking_score_field": "typed_return_pred",
                "oracle_aux_ranking_score_field": "typed_marginal_pred",
                "oracle_group_sampling": "best_type_balanced",
                "oracle_prefix_detach": True,
                "oracle_cache_prefix_latents": True,
                # Match the production planner's replay trajectory exactly;
                # without this bound 700-step training latents drift far
                # outside the state distribution used to generate labels.
                "oracle_latent_norm_clip_ratio": 4.0,
                "oracle_batch_groups": 12,
                "oracle_every_n_steps": 1,
                "oracle_warmup_epochs": 0,
                "oracle_ramp_epochs": 2,
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
        "round": 21,
        "purpose": "explicit-horizon isolated return ranking from real ATPG labels",
        "selection_circuit": "b15_C only",
        "validation_circuits": ["b15_C", "b20_C", "b21_C", "b22_C", "b17_C"],
        "target_circuits_in_training": False,
        "heldout_non_target_benchmarks": HOLDOUT_BENCHMARKS,
        "ultralong_oracle_states": ultra_manifest["state_count"],
        "ultralong_oracle_candidates": ultra_manifest["candidate_evaluations"],
        "trainable_branch": "horizon_return_rank_experts only",
        "planner_replay_alignment": {
            "prefix_state_mode": "replay",
            "latent_norm_clip_ratio": 4.0,
            "dropout": 0.0,
            "prefix_latent_cache": "exact because encoder/action_encoder/dynamics are frozen",
        },
        "frozen_predictions": [
            "incumbent return_rank_experts",
            "typed_marginal_pred",
            "typed_sa_reduction_pred",
            "encoder and dynamics",
        ],
        "variants": VARIANTS,
    }
    (ROOT / "training_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
