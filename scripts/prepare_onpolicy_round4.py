#!/usr/bin/env python3
"""Prepare typed-head-only training configs for round-4 on-policy labels."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("autoresearch/loop-260720-0945/model_training_round4")
BASE = Path("autoresearch/loop-260720-0945/model_training_round3/configs/long32_balanced.json")
LABELS = ROOT / "onpolicy_real_labels/labels.csv"
INIT = Path("autoresearch/loop-260720-0945/model_training_round2/runs/typed_long/best_final_horizon.pt")
VARIANTS = {
    "onpolicy_balanced": {"lambda_typed_marginal": 0.4, "lambda_typed_return": 0.1, "return_gamma": 0.97},
    "onpolicy_marginal": {"lambda_typed_marginal": 0.5, "lambda_typed_return": 0.0, "return_gamma": 1.0},
}


def main() -> None:
    base = json.loads(BASE.read_text())
    config_dir = ROOT / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    for name, overrides in VARIANTS.items():
        config = dict(base)
        config.update(
            {
                "labels": str(LABELS),
                "run_dir": str(ROOT / "runs" / name),
                "init_checkpoint": str(INIT),
                "init_checkpoint_strict": True,
                "seed": 2084,
                "train_frac": 0.82,
                "val_frac": 0.18,
                "lr": 0.0001,
                "epochs": 12,
                "rollout_max_horizon": 32,
                "rollout_horizon_schedule": [1, 2, 4, 8, 16, 32, 32, 32, 32, 32, 32, 32],
                "require_full_horizon": True,
                "trainable_modules": "typed_utility_only",
                "device": "cuda",
            }
        )
        config.update(overrides)
        path = config_dir / f"{name}.json"
        path.write_text(json.dumps(config, indent=2) + "\n")
        print(path)


if __name__ == "__main__":
    main()
