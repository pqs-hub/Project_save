#!/usr/bin/env python3
"""Prepare typed-head-only round-2 configs with a frozen production world model."""

from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/planner_aligned_q_rank_v5_context_safe.json"
OUT = ROOT / "autoresearch/loop-260720-0945/model_training_round2"
INIT = "runs/planner_aligned_q_rank_v5_context_safe/best_final_horizon.pt"


COMMON = {
    "seed": 2042,
    "epochs": 5,
    "max_train_steps_per_epoch": 400,
    "max_val_steps": 256,
    "init_checkpoint": INIT,
    "init_checkpoint_strict": False,
    "utility_head_type": "typed_film",
    "trainable_modules": "typed_utility_only",
    "oracle_ranking_score_field": "typed_marginal_pred",
    "train_sample_strategy": "shuffle",
    "oracle_every_n_steps": 15,
    "oracle_batch_groups": 3,
    "rollout_increase_every": 1,
    "save_epoch_checkpoints": True,
    "progress_bar": True,
    "progress_log_every": 20,
    "device": "cuda",
    # Frozen legacy losses are disabled so the logged objective measures only
    # the action-conditioned residual heads and their real-oracle ordering.
    "lambda_jepa": 0.0,
    "lambda_fc": 0.0,
    "lambda_hard": 0.0,
    "lambda_hard_reduction": 0.0,
    "lambda_hard_soft_f1": 0.0,
    "lambda_return": 0.0,
    "lambda_q_rank": 0.40,
    "lambda_q_value": 0.08,
    "lambda_candidate": 0.12,
    "lambda_context_rank": 0.15,
}


VARIANTS = {
    "typed_balanced": {
        "lr": 0.00035,
        "lambda_typed_marginal": 0.35,
        "lambda_typed_return": 0.25,
        "lambda_typed_sa_reduction": 0.25,
        "return_gamma": 0.90,
    },
    "typed_long": {
        "lr": 0.00030,
        "lambda_typed_marginal": 0.20,
        "lambda_typed_return": 0.50,
        "lambda_typed_sa_reduction": 0.25,
        "return_gamma": 1.0,
    },
    "typed_rank": {
        "lr": 0.00030,
        "lambda_typed_marginal": 0.45,
        "lambda_typed_return": 0.20,
        "lambda_typed_sa_reduction": 0.30,
        "lambda_q_rank": 0.65,
        "lambda_candidate": 0.20,
        "oracle_every_n_steps": 10,
        "return_gamma": 0.90,
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
        print(f"[prepare-typed-round2] wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
