#!/usr/bin/env python3
"""Materialize the fixed-data typed world-model round-1 training configs."""

from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/planner_aligned_q_rank_v5_context_safe.json"
OUT = ROOT / "autoresearch/loop-260720-0945/model_training_round1"
INIT = "runs/planner_aligned_q_rank_v5_context_safe/best_final_horizon.pt"


COMMON = {
    "seed": 2041,
    "epochs": 5,
    "max_train_steps_per_epoch": 400,
    "max_val_steps": 256,
    "init_checkpoint": INIT,
    "init_checkpoint_strict": False,
    "utility_head_type": "typed_film",
    "oracle_ranking_score_field": "typed_marginal_pred",
    "lambda_return": 0.12,
    "lambda_typed_marginal": 0.35,
    "lambda_typed_return": 0.20,
    "lambda_typed_sa_reduction": 0.20,
    "return_gamma": 0.90,
    "train_sample_strategy": "shuffle",
    "oracle_every_n_steps": 20,
    "oracle_batch_groups": 2,
    "rollout_increase_every": 1,
    "save_epoch_checkpoints": True,
    "progress_bar": True,
    "progress_log_every": 20,
    "device": "cuda",
}


VARIANTS = {
    "frozen_balanced": {
        "trainable_modules": "utility_posttrain",
        "lr": 0.00030,
    },
    "frozen_long": {
        "trainable_modules": "utility_posttrain",
        "lr": 0.00030,
        "lambda_return": 0.20,
        "lambda_typed_marginal": 0.25,
        "lambda_typed_return": 0.40,
        "return_gamma": 1.0,
    },
    "joint_low_lr": {
        "trainable_modules": "all",
        "lr": 0.00012,
        "lambda_jepa": 0.35,
        "lambda_typed_marginal": 0.35,
        "lambda_typed_return": 0.25,
        "lambda_typed_sa_reduction": 0.25,
    },
}


def main() -> None:
    base = json.loads(BASE.read_text())
    config_dir = OUT / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    for name, overrides in VARIANTS.items():
        config = copy.deepcopy(base)
        config.update(COMMON)
        config.update(overrides)
        config["run_dir"] = str((OUT / "runs" / name).relative_to(ROOT))
        path = config_dir / f"{name}.json"
        path.write_text(json.dumps(config, indent=2) + "\n")
        print(f"[prepare-typed-round1] wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
