#!/usr/bin/env python3
"""Prepare a b15-invariant late-horizon return adapter."""

from __future__ import annotations

import json
from pathlib import Path


LOOP = Path("autoresearch/loop-260720-0945")
ROUND21 = LOOP / "model_training_round21"
ROUND22 = LOOP / "model_training_round22"
ROOT = LOOP / "model_training_round23"
SOURCE_CONFIG = ROUND22 / "configs/toptype_r10_hard.json"
INIT_CHECKPOINT = ROUND21 / "runs/horizon_r10_pairwise/epoch_010.pt"
ULTRALONG = ROUND21 / "ultralong_prefix_oracle/oracle_actions.tsv"
VARIANT = "late_hard_fixed"


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
            "utility_head_type": "typed_cone_return_late_horizon_rank_moe",
            "trainable_modules": "typed_return_late_horizon_only",
            "epochs": 8,
            "seed": 2151,
            "lr": 5.0e-5,
            # Only prefixes beyond b15's zero-based final step (277) can
            # produce a gradient through the structural late gate.
            "oracle_actions": [{"path": str(ULTRALONG), "repeat": 1}],
            "oracle_pairwise_mode": "best_vs_hard_topk",
            "oracle_positive_topk": 2,
            "oracle_hard_negative_topk": 15,
            "oracle_max_pairs_per_group": 64,
            "lambda_q_rank": 1.5,
            "lambda_same_type_rank": 0.1,
            "lambda_candidate": 0.25,
            "lambda_ndcg_rank": 0.25,
            "lambda_same_type_ndcg_rank": 0.0,
            "oracle_group_sampling": "best_type_balanced",
            "oracle_batch_groups": 12,
            "oracle_every_n_steps": 1,
            "oracle_warmup_epochs": 0,
            "oracle_ramp_epochs": 2,
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
        "round": 23,
        "purpose": "b15-invariant late-horizon top/action-type correction",
        "selection_rule": "single predeclared epoch-8 checkpoint; b15 equality is mandatory",
        "init_checkpoint": str(INIT_CHECKPOINT),
        "target_circuits_in_training": False,
        "heldout_non_target_benchmarks": ["subckt_0230", "subckt_0360"],
        "oracle_source": str(ULTRALONG),
        "late_gate": {
            "zero_through_sequence_step": 277,
            "full_from_sequence_step": 320,
        },
        "trainable_branch": "late_return_rank_experts only",
        "fixed_epoch": 8,
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "training_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(config_path)


if __name__ == "__main__":
    main()
