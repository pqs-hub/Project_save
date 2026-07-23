#!/usr/bin/env python3
"""Prepare typed-head training configs for Round5 prefix counterfactual ranking."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path("autoresearch/loop-260720-0945/model_training_round5")
BASE = Path("autoresearch/loop-260720-0945/model_training_round4/configs/onpolicy_balanced.json")
INIT = Path("autoresearch/loop-260720-0945/model_training_round4/runs/onpolicy_balanced/best_final_horizon.pt")
PREFIX_ORACLE = ROOT / "onpolicy_prefix_oracle/oracle_actions.tsv"
INITIAL_ORACLE = Path("autoresearch/oracle-balanced-negative-rich-260629-wide/balanced_train_oracle_actions.tsv")
VARIANTS = {
    "prefix_rank": {
        "lambda_q_rank": 0.65,
        "lambda_candidate": 0.15,
        "lambda_ndcg_rank": 0.0,
        "lambda_context_rank": 0.25,
        "lambda_conservative_q": 0.0,
        "lambda_oracle_sa_value": 0.0,
    },
    "prefix_rank_sa": {
        "lambda_q_rank": 0.65,
        "lambda_candidate": 0.15,
        "lambda_ndcg_rank": 0.0,
        "lambda_context_rank": 0.25,
        "lambda_conservative_q": 0.0,
        "lambda_oracle_sa_value": 0.20,
    },
    "prefix_cql_sa": {
        "lr": 7.5e-5,
        "lambda_q_rank": 0.55,
        "lambda_candidate": 0.10,
        "lambda_ndcg_rank": 0.0,
        "lambda_context_rank": 0.20,
        "lambda_conservative_q": 0.15,
        "lambda_oracle_sa_value": 0.20,
    },
    "prefix_toplist_sa": {
        "lr": 1.0e-4,
        "lambda_q_rank": 0.45,
        "lambda_candidate": 0.20,
        "lambda_ndcg_rank": 0.25,
        "lambda_context_rank": 0.15,
        "lambda_conservative_q": 0.10,
        "lambda_oracle_sa_value": 0.25,
    },
}

EXPECTED_PREFIX_STEPS = [1, 2, 4, 8, 12, 16, 24, 31]
EXPECTED_PATTERNS = 300_000
EXPECTED_ACTIONS_PER_PREFIX = 9


def validate_prefix_oracle(
    prefix_oracle: Path = PREFIX_ORACLE,
    manifest_path: Path | None = None,
    *,
    expected_plans_dir: Path | None = None,
    expected_prefix_steps: list[int] | None = None,
    expected_patterns: int = EXPECTED_PATTERNS,
    expected_actions_per_prefix: int = EXPECTED_ACTIONS_PER_PREFIX,
) -> dict:
    """Reject stale/incomplete oracle data before a training config can use it."""

    manifest_path = manifest_path or prefix_oracle.parent / "manifest.json"
    expected_plans_dir = expected_plans_dir or ROOT / "onpolicy_plans"
    expected_prefix_steps = expected_prefix_steps or EXPECTED_PREFIX_STEPS
    if not prefix_oracle.is_file():
        raise ValueError(f"missing Round5 prefix oracle labels: {prefix_oracle}")
    if not manifest_path.is_file():
        raise ValueError(f"missing Round5 prefix oracle manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    errors: list[str] = []
    if int(manifest.get("patterns") or 0) != expected_patterns:
        errors.append(f"patterns={manifest.get('patterns')} expected={expected_patterns}")
    if int(manifest.get("actions_per_prefix") or 0) != expected_actions_per_prefix:
        errors.append(
            f"actions_per_prefix={manifest.get('actions_per_prefix')} expected={expected_actions_per_prefix}"
        )
    actual_prefixes = sorted(int(value) for value in manifest.get("prefix_steps") or [])
    if actual_prefixes != sorted(expected_prefix_steps):
        errors.append(f"prefix_steps={actual_prefixes} expected={sorted(expected_prefix_steps)}")
    plans_dir = Path(str(manifest.get("plans_dir") or ""))
    if plans_dir.resolve() != expected_plans_dir.resolve():
        errors.append(f"plans_dir={plans_dir} expected={expected_plans_dir}")

    with prefix_oracle.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    expected_rows = int(manifest.get("candidate_evaluations") or 0)
    if expected_rows <= 0 or len(rows) != expected_rows:
        errors.append(f"oracle_rows={len(rows)} manifest_candidate_evaluations={expected_rows}")
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (str(row.get("benchmark_id") or ""), str(row.get("state_id") or ""))
        groups.setdefault(key, []).append(row)
        if row.get("status") != "ok":
            errors.append(f"non-ok oracle row at {key}: {row.get('status')!r}")
        try:
            prefix_step = int(row.get("prefix_step") or -1)
        except ValueError:
            prefix_step = -1
        if prefix_step not in expected_prefix_steps:
            errors.append(f"unexpected prefix_step={prefix_step} at {key}")
        source_plan = Path(str(row.get("source_plan_csv") or ""))
        if source_plan.parent.resolve() != expected_plans_dir.resolve():
            errors.append(f"stale source_plan_csv={source_plan} at {key}")
    for key, group in groups.items():
        if len(group) != expected_actions_per_prefix:
            errors.append(f"candidate_count={len(group)} expected={expected_actions_per_prefix} at {key}")
        action_keys = {row.get("action_key") for row in group}
        if len(action_keys) != len(group):
            errors.append(f"duplicate action_key at {key}")
    expected_states = int(manifest.get("state_count") or 0)
    if expected_states <= 0 or len(groups) != expected_states:
        errors.append(f"oracle_states={len(groups)} manifest_state_count={expected_states}")
    if errors:
        preview = "; ".join(errors[:8])
        raise ValueError(f"Round5 prefix oracle provenance check failed: {preview}")
    return manifest


def main() -> None:
    try:
        manifest = validate_prefix_oracle()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        "validated Round5 prefix oracle "
        f"states={manifest['state_count']} candidates={manifest['candidate_evaluations']} "
        f"patterns={manifest['patterns']}"
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
                "init_checkpoint_strict": True,
                "seed": 2095,
                "epochs": 12,
                "lr": 5e-5,
                "trainable_modules": "typed_utility_only",
                "oracle_actions": [
                    {"path": str(INITIAL_ORACLE), "repeat": 1},
                    {"path": str(PREFIX_ORACLE), "repeat": 3},
                ],
                "oracle_ranking_score_field": "typed_marginal_pred",
                "oracle_pairwise_mode": "all",
                "oracle_max_pairs_per_group": 64,
                "oracle_batch_groups": 8,
                "oracle_every_n_steps": 1,
                "oracle_pairwise_min_delta": 0.0005,
                "oracle_pairwise_temperature": 0.5,
                "candidate_target_temperature": 0.25,
                "candidate_pred_temperature": 1.0,
                "oracle_ndcg_k": 3,
                "oracle_ndcg_target_temperature": 0.15,
                "oracle_ndcg_pred_temperature": 1.0,
                "oracle_context_top_weight": 0.65,
                "oracle_prefix_detach": True,
                "device": "cuda",
            }
        )
        config.update(overrides)
        path = config_dir / f"{name}.json"
        path.write_text(json.dumps(config, indent=2) + "\n")
        print(path)


if __name__ == "__main__":
    main()
