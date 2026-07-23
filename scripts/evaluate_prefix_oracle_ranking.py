#!/usr/bin/env python3
"""Audit checkpoint ranking on fixed non-initial prefix-oracle groups.

Unlike ``evaluate_oracle_action_values.py`` v1, this evaluator replays the
shared state-action prefix before scoring each candidate.  It never invokes
ATPG and is intended as a diagnostic; target-circuit model selection remains
separate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tpi_jepa.plan import load_checkpoint  # noqa: E402
from tpi_jepa.train import (  # noqa: E402
    _canonical_oracle_action_type,
    _predict_oracle_group_scores,
    load_oracle_groups,
)


def parse_checkpoint_specs(values: list[str]) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(f"checkpoint spec must be NAME=PATH, got {item!r}")
            name, path = (part.strip() for part in item.split("=", 1))
            if not name or not path:
                raise ValueError(f"checkpoint spec must be NAME=PATH, got {item!r}")
            specs.append((name, path))
    names = [name for name, _ in specs]
    if len(names) != len(set(names)):
        raise ValueError("checkpoint names must be unique")
    return specs


def average_ranks(values: list[float]) -> list[float]:
    """Return zero-based average ranks, ascending, with exact-tie handling."""

    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = 0.5 * (start + end - 1)
        for offset in range(start, end):
            ranks[order[offset]] = rank
        start = end
    return ranks


def pearson(left: list[float], right: list[float]) -> float:
    if len(left) < 2 or len(left) != len(right):
        return float("nan")
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_norm = sum((a - left_mean) ** 2 for a in left)
    right_norm = sum((b - right_mean) ** 2 for b in right)
    denominator = math.sqrt(left_norm * right_norm)
    return numerator / denominator if denominator > 0.0 else float("nan")


def pairwise_accuracy(predictions: list[float], targets: list[float]) -> float:
    correct = 0.0
    total = 0
    for left in range(len(targets)):
        for right in range(left + 1, len(targets)):
            target_diff = targets[left] - targets[right]
            if abs(target_diff) <= 1e-12:
                continue
            pred_diff = predictions[left] - predictions[right]
            total += 1
            if pred_diff == 0.0:
                correct += 0.5
            elif pred_diff * target_diff > 0.0:
                correct += 1.0
    return correct / total if total else float("nan")


def same_type_pairwise_accuracy(
    predictions: list[float],
    targets: list[float],
    action_types: list[str],
) -> float:
    """Measure node ordering without giving credit for easy cross-type pairs."""

    correct = 0.0
    total = 0
    for left in range(len(targets)):
        for right in range(left + 1, len(targets)):
            if action_types[left] != action_types[right]:
                continue
            target_diff = targets[left] - targets[right]
            if abs(target_diff) <= 1e-12:
                continue
            pred_diff = predictions[left] - predictions[right]
            total += 1
            if pred_diff == 0.0:
                correct += 0.5
            elif pred_diff * target_diff > 0.0:
                correct += 1.0
    return correct / total if total else float("nan")


def finite_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else float("nan")


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-actions", required=True)
    parser.add_argument("--checkpoints", action="append", default=[])
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--score-fields", default="typed_marginal_pred")
    parser.add_argument("--plan-device", default="cuda")
    parser.add_argument(
        "--latent-norm-clip-ratio",
        type=float,
        default=None,
        help="Override oracle replay clipping to match the production planner.",
    )
    parser.add_argument(
        "--include-benchmarks",
        default="",
        help="Optional comma-separated benchmark allowlist.",
    )
    parser.add_argument(
        "--exclude-benchmarks",
        default="",
        help="Optional comma-separated benchmark denylist.",
    )
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    specs = parse_checkpoint_specs([*args.checkpoints, *args.checkpoint])
    if not specs:
        raise ValueError("at least one NAME=PATH checkpoint is required")
    if args.plan_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")
    device = torch.device(args.plan_device)
    score_fields = [field.strip() for field in args.score_fields.split(",") if field.strip()]
    groups = load_oracle_groups(args.oracle_actions)
    included = {value.strip() for value in args.include_benchmarks.split(",") if value.strip()}
    excluded = {value.strip() for value in args.exclude_benchmarks.split(",") if value.strip()}
    if included:
        groups = [group for group in groups if group[0]["benchmark_id"] in included]
    if excluded:
        groups = [group for group in groups if group[0]["benchmark_id"] not in excluded]
    if not groups:
        raise ValueError("benchmark filters removed every oracle group")
    print(
        f"[prefix-audit] groups={len(groups)} checkpoints={len(specs)} "
        f"include={','.join(sorted(included)) or '*'} "
        f"exclude={','.join(sorted(excluded)) or '-'}",
        flush=True,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    details: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for checkpoint_index, (checkpoint_name, checkpoint_path) in enumerate(specs, start=1):
        print(
            f"[prefix-audit] start checkpoint={checkpoint_name} "
            f"index={checkpoint_index}/{len(specs)}",
            flush=True,
        )
        model, config = load_checkpoint(checkpoint_path, device)
        if args.latent_norm_clip_ratio is not None:
            config["oracle_latent_norm_clip_ratio"] = args.latent_norm_clip_ratio
        model.eval()
        graph_cache: dict[str, Any] = {}
        base_cache: dict[str, torch.Tensor] = {}
        by_field: dict[str, list[dict[str, Any]]] = {field: [] for field in score_fields}
        with torch.no_grad():
            for group in groups:
                scores = _predict_oracle_group_scores(
                    model, config, group, graph_cache, base_cache, device
                )
                targets = [100.0 * float(row["oracle_delta_tc"]) for row in group]
                action_types = [_canonical_oracle_action_type(row["type"]) for row in group]
                best_index = max(range(len(group)), key=lambda index: targets[index])
                for field in score_fields:
                    if field not in scores:
                        valid = ", ".join(sorted(scores))
                        raise ValueError(f"unsupported score field {field!r}; valid: {valid}")
                    predictions = [float(value) for value in scores[field].detach().cpu().tolist()]
                    selected_index = max(range(len(group)), key=lambda index: predictions[index])
                    selected_target = targets[selected_index]
                    best_target = targets[best_index]
                    selected_type = _canonical_oracle_action_type(group[selected_index]["type"])
                    best_type = _canonical_oracle_action_type(group[best_index]["type"])
                    best_type_indices = [
                        index for index, action_type in enumerate(action_types) if action_type == best_type
                    ]
                    within_type_index = max(best_type_indices, key=lambda index: predictions[index])
                    row = {
                        "checkpoint": checkpoint_name,
                        "score_field": field,
                        "benchmark_id": group[0]["benchmark_id"],
                        "state_id": group[0]["state_id"],
                        "prefix_step": group[0].get("prefix_step", ""),
                        "candidate_count": len(group),
                        "selected_action": group[selected_index].get("action_key", ""),
                        "selected_type": selected_type,
                        "best_action": group[best_index].get("action_key", ""),
                        "best_type": best_type,
                        "top1_hit": int(selected_index == best_index),
                        "type_hit": int(selected_type == best_type),
                        "within_best_type_top1_hit": int(within_type_index == best_index),
                        "within_best_type_regret_pp": best_target - targets[within_type_index],
                        "selected_delta_tc_pp": selected_target,
                        "best_delta_tc_pp": best_target,
                        "top1_regret_pp": best_target - selected_target,
                        "negative_top1": int(selected_target < 0.0),
                        "spearman": pearson(average_ranks(predictions), average_ranks(targets)),
                        "pairwise_accuracy": pairwise_accuracy(predictions, targets),
                        "same_type_pairwise_accuracy": same_type_pairwise_accuracy(
                            predictions, targets, action_types
                        ),
                    }
                    details.append(row)
                    by_field[field].append(row)
        for field, rows in by_field.items():
            type_counts = {
                action_type: sum(row["selected_type"] == action_type for row in rows)
                for action_type in ("control0", "control1", "observe")
            }
            summaries.append(
                {
                    "checkpoint": checkpoint_name,
                    "score_field": field,
                    "groups": len(rows),
                    "top1_accuracy": finite_mean([float(row["top1_hit"]) for row in rows]),
                    "type_accuracy": finite_mean([float(row["type_hit"]) for row in rows]),
                    "within_best_type_top1_accuracy": finite_mean(
                        [float(row["within_best_type_top1_hit"]) for row in rows]
                    ),
                    "mean_within_best_type_regret_pp": finite_mean(
                        [float(row["within_best_type_regret_pp"]) for row in rows]
                    ),
                    "mean_selected_delta_tc_pp": finite_mean(
                        [float(row["selected_delta_tc_pp"]) for row in rows]
                    ),
                    "mean_top1_regret_pp": finite_mean(
                        [float(row["top1_regret_pp"]) for row in rows]
                    ),
                    "negative_top1_rate": finite_mean(
                        [float(row["negative_top1"]) for row in rows]
                    ),
                    "mean_spearman": finite_mean([float(row["spearman"]) for row in rows]),
                    "mean_pairwise_accuracy": finite_mean(
                        [float(row["pairwise_accuracy"]) for row in rows]
                    ),
                    "mean_same_type_pairwise_accuracy": finite_mean(
                        [float(row["same_type_pairwise_accuracy"]) for row in rows]
                    ),
                    "selected_control0": type_counts["control0"],
                    "selected_control1": type_counts["control1"],
                    "selected_observe": type_counts["observe"],
                }
            )
        print(
            f"[prefix-audit] done checkpoint={checkpoint_name} "
            f"index={checkpoint_index}/{len(specs)}",
            flush=True,
        )

    detail_fields = list(details[0])
    summary_fields = list(summaries[0])
    write_tsv(out_dir / "details.tsv", details, detail_fields)
    write_tsv(out_dir / "summary.tsv", summaries, summary_fields)
    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    lines = [
        "# Prefix-oracle ranking audit",
        "",
        "This fixed-label diagnostic did not run ATPG and was not used for target-circuit selection.",
        "",
        "| checkpoint | score | groups | top-1 | type | within-type top-1 | regret (pp) | within regret | negative | Spearman | pairwise | same-type pairwise | CP0/CP1/OP |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {checkpoint} | `{score_field}` | {groups} | {top1_accuracy:.3%} | "
            "{type_accuracy:.3%} | {within_best_type_top1_accuracy:.3%} | "
            "{mean_top1_regret_pp:.4f} | {mean_within_best_type_regret_pp:.4f} | "
            "{negative_top1_rate:.3%} | {mean_spearman:.4f} | "
            "{mean_pairwise_accuracy:.4f} | {mean_same_type_pairwise_accuracy:.4f} | "
            "{selected_control0}/{selected_control1}/{selected_observe} |".format(**row)
        )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
