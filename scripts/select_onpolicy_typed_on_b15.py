#!/usr/bin/env python3
"""Select one on-policy typed checkpoint and global score configuration using b15 only."""

from __future__ import annotations

import csv
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("autoresearch/loop-260720-0945/model_training_round4"),
    )
    parser.add_argument(
        "--incumbent-manifest",
        type=Path,
        default=None,
        help="Optional prior b15-only winner.json to keep in the selection pool.",
    )
    args = parser.parse_args()
    base = args.base if args.base.is_absolute() else ROOT / args.base
    candidates = []
    for path in sorted((base / "b15_selection").rglob("results.tsv")):
        config_path = path.parent / "config.json"
        if not config_path.is_file():
            continue
        config = json.loads(config_path.read_text())
        env = config.get("planner_environment") or {}
        checkpoint_path = Path(str(config.get("checkpoint") or ""))
        if not checkpoint_path:
            continue
        checkpoint = checkpoint_path if checkpoint_path.is_absolute() else ROOT / checkpoint_path
        try:
            checkpoint_value = str(checkpoint.relative_to(ROOT))
        except ValueError:
            checkpoint_value = str(checkpoint)
        variant = checkpoint.parent.name
        checkpoint_tag = checkpoint.stem
        rows = list(csv.DictReader(path.open(newline=""), delimiter="\t"))
        for row in rows:
            if row.get("status") != "ok" or row.get("benchmark_id") != "iscas99__b15_1":
                continue
            planner_environment = {
                str(key): str(value)
                for key, value in env.items()
                if value is not None
            }
            candidates.append(
                {
                    "variant": variant,
                    "checkpoint_tag": checkpoint_tag,
                    "checkpoint": checkpoint_value,
                    "ensemble_checkpoints": str(config.get("ensemble_checkpoints") or ""),
                    "ensemble_lcb_alpha": float(config.get("ensemble_lcb_alpha") or 1.0),
                    "max_candidates": int(config.get("max_candidates") or 48),
                    "candidate_strategy": str(
                        row.get("candidate_strategy")
                        or config.get("candidate_strategies")
                        or "hard_fault_cluster"
                    ),
                    "planner_environment": planner_environment,
                    "adaptive_base_candidates": int(
                        env.get("TPI_ADAPTIVE_BASE_CANDIDATES") or 0
                    ),
                    "adaptive_expansion_margin": float(
                        env.get("TPI_ADAPTIVE_EXPANSION_MARGIN") or 0.0
                    ),
                    "adaptive_margin_mode": str(
                        env.get("TPI_ADAPTIVE_MARGIN_MODE") or "absolute"
                    ),
                    "score_field": row["score_field"],
                    "typed_residual_alpha": float(env.get("TPI_TYPED_RESIDUAL_ALPHA") or 0.0),
                    "typed_residual_decay_steps": float(env.get("TPI_TYPED_RESIDUAL_DECAY_STEPS") or 0.0),
                    "typed_trust_min_heads": int(env.get("TPI_TYPED_TRUST_MIN_HEADS") or 2),
                    "typed_trust_cp0_min_heads": int(env.get("TPI_TYPED_TRUST_CP0_MIN_HEADS") or 3),
                    "typed_trust_head_margin": float(env.get("TPI_TYPED_TRUST_HEAD_MARGIN") or 0.0),
                    "typed_trust_advantage_margin": float(
                        env.get("TPI_TYPED_TRUST_ADVANTAGE_MARGIN") or 0.0
                    ),
                    "typed_reliable_marginal_weight": float(
                        env.get("TPI_TYPED_RELIABLE_MARGINAL_WEIGHT") or 0.75
                    ),
                    "typed_reliable_min_heads": int(
                        env.get("TPI_TYPED_RELIABLE_MIN_HEADS") or 1
                    ),
                    "typed_reliable_cp0_min_heads": int(
                        env.get("TPI_TYPED_RELIABLE_CP0_MIN_HEADS") or 2
                    ),
                    "b15_delta_tc": float(row["delta_test_coverage"]),
                    "is_incumbent": False,
                }
            )
    if args.incumbent_manifest is not None:
        incumbent_path = (
            args.incumbent_manifest
            if args.incumbent_manifest.is_absolute()
            else ROOT / args.incumbent_manifest
        )
        incumbent = json.loads(incumbent_path.read_text())["winner"]
        incumbent_checkpoint = Path(str(incumbent["checkpoint"]))
        incumbent_checkpoint = (
            incumbent_checkpoint
            if incumbent_checkpoint.is_absolute()
            else ROOT / incumbent_checkpoint
        )
        try:
            incumbent_checkpoint_value = str(incumbent_checkpoint.relative_to(ROOT))
        except ValueError:
            incumbent_checkpoint_value = str(incumbent_checkpoint)
        candidates.append(
            {
                "variant": str(incumbent.get("variant") or incumbent_checkpoint.parent.name),
                "checkpoint_tag": incumbent_checkpoint.stem,
                "checkpoint": incumbent_checkpoint_value,
                "ensemble_checkpoints": str(incumbent.get("ensemble_checkpoints") or ""),
                "ensemble_lcb_alpha": float(incumbent.get("ensemble_lcb_alpha", 1.0)),
                "max_candidates": int(incumbent.get("max_candidates") or 48),
                "candidate_strategy": str(
                    incumbent.get("candidate_strategy") or "hard_fault_cluster"
                ),
                "planner_environment": dict(incumbent.get("planner_environment") or {}),
                "adaptive_base_candidates": int(
                    incumbent.get("adaptive_base_candidates") or 0
                ),
                "adaptive_expansion_margin": float(
                    incumbent.get("adaptive_expansion_margin") or 0.0
                ),
                "adaptive_margin_mode": str(
                    incumbent.get("adaptive_margin_mode") or "absolute"
                ),
                "score_field": str(incumbent["score_field"]),
                "typed_residual_alpha": float(incumbent.get("typed_residual_alpha") or 0.0),
                "typed_residual_decay_steps": float(
                    incumbent.get("typed_residual_decay_steps") or 0.0
                ),
                "typed_trust_min_heads": int(incumbent.get("typed_trust_min_heads", 2)),
                "typed_trust_cp0_min_heads": int(
                    incumbent.get("typed_trust_cp0_min_heads", 3)
                ),
                "typed_trust_head_margin": float(
                    incumbent.get("typed_trust_head_margin", 0.0)
                ),
                "typed_trust_advantage_margin": float(
                    incumbent.get("typed_trust_advantage_margin", 0.0)
                ),
                "typed_reliable_marginal_weight": float(
                    incumbent.get("typed_reliable_marginal_weight", 0.75)
                ),
                "typed_reliable_min_heads": int(
                    incumbent.get("typed_reliable_min_heads", 1)
                ),
                "typed_reliable_cp0_min_heads": int(
                    incumbent.get("typed_reliable_cp0_min_heads", 2)
                ),
                "b15_delta_tc": float(incumbent["b15_delta_tc"]),
                "source": str(incumbent_path.relative_to(ROOT)),
                "is_incumbent": True,
            }
        )
    if not candidates:
        raise SystemExit(f"no successful b15 candidates under {base / 'b15_selection'}")
    score_preference = {
        "q_pred_context": 0,
        "q_typed_reliable_context": 1,
        "q_typed_trust_context": 2,
        "q_typed_residual_context": 3,
    }
    candidates.sort(
        key=lambda x: (
            -x["b15_delta_tc"],
            0 if x.get("is_incumbent") else 1,
            score_preference.get(x["score_field"], 3),
            x["typed_residual_alpha"],
            x["typed_residual_decay_steps"] or float("inf"),
            x["variant"],
            x["checkpoint_tag"],
        )
    )
    manifest = {
        "selection_circuit": "b15_C",
        "validation_circuits_read": [],
        "selection_rule": (
            "maximize b15 delta TC; exact ties retain the incumbent, then prefer the frozen context head, "
            "then reliability-filtered residual, three-head trust residual, raw residual, "
            "and smaller residual weight"
        ),
        "winner": candidates[0],
        "candidates": candidates,
    }
    out = base / "b15_selection/winner.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["winner"], indent=2))


if __name__ == "__main__":
    main()
