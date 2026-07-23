#!/usr/bin/env python3
"""Prepare the tie-aware, b15-invariant late action-type calibrator."""

from __future__ import annotations

import json
from pathlib import Path


LOOP = Path("autoresearch/loop-260720-0945")
ROUND21 = LOOP / "model_training_round21"
ROUND23 = LOOP / "model_training_round23"
ROOT = LOOP / "model_training_round24"
SOURCE_CONFIG = ROUND23 / "configs/late_hard_fixed.json"
INIT_CHECKPOINT = ROUND21 / "runs/horizon_r10_pairwise/epoch_010.pt"
ULTRALONG = ROUND21 / "ultralong_prefix_oracle/oracle_actions.tsv"
VARIANT = "late_type_tie_soft_fixed"


def main() -> None:
    for path in (SOURCE_CONFIG, INIT_CHECKPOINT, ULTRALONG):
        if not path.is_file():
            raise FileNotFoundError(path)

    config = json.loads(SOURCE_CONFIG.read_text())
    config.update(
        {
            "run_dir": str(ROOT / "runs" / VARIANT),
            "init_checkpoint": str(INIT_CHECKPOINT),
            "init_checkpoint_strict": False,
            "utility_head_type": "typed_cone_return_late_type_rank_moe",
            "trainable_modules": "typed_return_late_type_only",
            # Fixed before training: the tiny adapter needs a larger step than
            # the 7k-parameter Round23 expert bank, but retains AdamW decay.
            "epochs": 8,
            "seed": 2152,
            "lr": 5.0e-4,
            "oracle_actions": [{"path": str(ULTRALONG), "repeat": 1}],
            # Train only the explicit family-level objective.  R21 continues
            # to own every within-family node comparison.
            "lambda_q_rank": 0.0,
            "lambda_same_type_rank": 0.0,
            "lambda_candidate": 0.0,
            "lambda_action_type_rank": 1.0,
            "lambda_ndcg_rank": 0.0,
            "lambda_same_type_ndcg_rank": 0.0,
            "lambda_conservative_q": 0.0,
            "lambda_context_rank": 0.0,
            "oracle_action_type_aggregate_temperature": 0.10,
            "oracle_action_type_target_temperature": 0.025,
            "oracle_action_type_pred_temperature": 0.35,
            # Uniform sampling avoids the old hard argmax's arbitrary choice
            # among exact ATPG ties.
            "oracle_group_sampling": "uniform",
            "oracle_batch_groups": 15,
            "oracle_every_n_steps": 1,
            "oracle_warmup_epochs": 0,
            "oracle_ramp_epochs": 1,
            "oracle_cache_prefix_latents": True,
            "oracle_latent_norm_clip_ratio": 4.0,
            "dropout": 0.0,
            "device": "cuda",
        }
    )

    config_dir = ROOT / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{VARIANT}.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    provenance = {
        "round": 24,
        "purpose": "tie-aware low-capacity late action-type calibration",
        "selection_rule": "single predeclared epoch-8 checkpoint; b15 equality is mandatory",
        "init_checkpoint": str(INIT_CHECKPOINT),
        "target_circuits_in_training": False,
        "heldout_non_target_benchmarks": ["subckt_0230", "subckt_0360"],
        "oracle_source": str(ULTRALONG),
        "label_rule": "softmax over per-type max delta-TC; exact ties remain soft ties",
        "adapter_scope": "action type plus analytic horizon only; no node features",
        "late_gate": {
            "zero_through_sequence_step": 277,
            "full_from_sequence_step": 320,
        },
        "fixed_epoch": 8,
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "training_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(config_path)


if __name__ == "__main__":
    main()
