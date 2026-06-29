"""Build negative-action-balanced oracle subsets from labeled action groups."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any


SUMMARY_FIELDS = [
    "split",
    "input_rows",
    "input_groups",
    "kept_rows",
    "kept_groups",
    "dropped_groups",
    "positive",
    "zero",
    "negative",
    "negative_rate",
    "control_negative_groups",
    "observe_negative_groups",
    "meets_group_target",
    "meets_negative_rate_target",
]
GROUP_DECISION_FIELDS = [
    "split",
    "benchmark_id",
    "state_id",
    "candidate_strategy",
    "input_rows",
    "kept_rows",
    "positive",
    "zero",
    "negative",
    "control_negative",
    "observe_negative",
    "decision",
    "reason",
]


def parse_csv_values(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def safe_float(value: Any, default: float = float("nan")) -> float:
    if value in (None, "", "NA"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def merge_fieldnames(*field_lists: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for fields in field_lists:
        for field in fields:
            if field in seen:
                continue
            merged.append(field)
            seen.add(field)
    return merged


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def group_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("benchmark_id", "")),
        str(row.get("state_id", "")),
        str(row.get("candidate_strategy", "")),
    )


def finite_ok(row: dict[str, str]) -> bool:
    status = row.get("status", "ok")
    return status == "ok" and math.isfinite(safe_float(row.get("oracle_delta_tc")))


def sign_counts(rows: list[dict[str, str]]) -> tuple[int, int, int]:
    values = [safe_float(row.get("oracle_delta_tc")) for row in rows]
    return (
        sum(value > 0.0 for value in values),
        sum(value == 0.0 for value in values),
        sum(value < 0.0 for value in values),
    )


def group_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if finite_ok(row):
            grouped[group_key(row)].append(row)
    return dict(grouped)


def control_negative_count(rows: list[dict[str, str]], preferred_types: set[str]) -> int:
    return sum(
        1
        for row in rows
        if row.get("type") in preferred_types and safe_float(row.get("oracle_delta_tc")) < 0.0
    )


def observe_negative_count(rows: list[dict[str, str]]) -> int:
    return sum(1 for row in rows if row.get("type") == "observe" and safe_float(row.get("oracle_delta_tc")) < 0.0)


def select_rows(
    rows: list[dict[str, str]],
    max_actions_per_group: int,
    preferred_types: set[str],
) -> list[dict[str, str]]:
    if max_actions_per_group <= 0 or len(rows) <= max_actions_per_group:
        return list(rows)
    negatives = [row for row in rows if safe_float(row.get("oracle_delta_tc")) < 0.0]
    positives = [row for row in rows if safe_float(row.get("oracle_delta_tc")) > 0.0]
    zeros = [row for row in rows if safe_float(row.get("oracle_delta_tc")) == 0.0]
    preferred_neg = [row for row in negatives if row.get("type") in preferred_types]
    other_neg = [row for row in negatives if row.get("type") not in preferred_types]
    positives_sorted = sorted(positives, key=lambda row: safe_float(row.get("oracle_delta_tc")), reverse=True)
    negatives_sorted = sorted([*preferred_neg, *other_neg], key=lambda row: safe_float(row.get("oracle_delta_tc")))
    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    for pool in [negatives_sorted, positives_sorted, zeros]:
        for row in pool:
            key = row.get("action_key") or f"{row.get('node')}::{row.get('type')}"
            if key in seen:
                continue
            selected.append(row)
            seen.add(key)
            if len(selected) >= max_actions_per_group:
                return selected
    return selected


def build_subset(
    *,
    split: str,
    rows: list[dict[str, str]],
    min_negatives: int,
    min_positives: int,
    preferred_types: set[str],
    max_actions_per_group: int,
) -> tuple[list[dict[str, str]], list[dict[str, Any]], dict[str, Any]]:
    kept: list[dict[str, str]] = []
    decisions: list[dict[str, Any]] = []
    grouped = group_rows(rows)
    for key, group in sorted(grouped.items()):
        positive, zero, negative = sign_counts(group)
        control_neg = control_negative_count(group, preferred_types)
        observe_neg = observe_negative_count(group)
        if negative < min_negatives:
            decision = "drop"
            reason = f"negative<{min_negatives}"
            selected: list[dict[str, str]] = []
        elif positive < min_positives:
            decision = "drop"
            reason = f"positive<{min_positives}"
            selected = []
        else:
            decision = "keep"
            reason = "balanced"
            selected = select_rows(group, max_actions_per_group, preferred_types)
            kept.extend(selected)
        decisions.append(
            {
                "split": split,
                "benchmark_id": key[0],
                "state_id": key[1],
                "candidate_strategy": key[2],
                "input_rows": len(group),
                "kept_rows": len(selected),
                "positive": positive,
                "zero": zero,
                "negative": negative,
                "control_negative": control_neg,
                "observe_negative": observe_neg,
                "decision": decision,
                "reason": reason,
            }
        )

    kept_positive, kept_zero, kept_negative = sign_counts(kept)
    kept_groups = {group_key(row) for row in kept}
    input_groups = len(grouped)
    summary = {
        "split": split,
        "input_rows": sum(len(group) for group in grouped.values()),
        "input_groups": input_groups,
        "kept_rows": len(kept),
        "kept_groups": len(kept_groups),
        "dropped_groups": input_groups - len(kept_groups),
        "positive": kept_positive,
        "zero": kept_zero,
        "negative": kept_negative,
        "negative_rate": kept_negative / len(kept) if kept else float("nan"),
        "control_negative_groups": sum(
            1
            for key in kept_groups
            if any(group_key(row) == key and row.get("type") in preferred_types and safe_float(row.get("oracle_delta_tc")) < 0.0 for row in kept)
        ),
        "observe_negative_groups": sum(
            1
            for key in kept_groups
            if any(group_key(row) == key and row.get("type") == "observe" and safe_float(row.get("oracle_delta_tc")) < 0.0 for row in kept)
        ),
    }
    return kept, decisions, summary


def annotate_targets(summary: dict[str, Any], min_groups: int, low_rate: float, high_rate: float) -> dict[str, Any]:
    negative_rate = safe_float(summary.get("negative_rate"))
    summary["meets_group_target"] = int(int(summary.get("kept_groups", 0)) >= min_groups)
    summary["meets_negative_rate_target"] = int(math.isfinite(negative_rate) and low_rate <= negative_rate <= high_rate)
    return summary


def build_eval_only_summary(
    *,
    split: str,
    rows: list[dict[str, str]],
    preferred_types: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped = group_rows(rows)
    decisions: list[dict[str, Any]] = []
    finite_rows = [row for group in grouped.values() for row in group]
    positive, zero, negative = sign_counts(finite_rows)
    for key, group in sorted(grouped.items()):
        group_positive, group_zero, group_negative = sign_counts(group)
        decisions.append(
            {
                "split": split,
                "benchmark_id": key[0],
                "state_id": key[1],
                "candidate_strategy": key[2],
                "input_rows": len(group),
                "kept_rows": 0,
                "positive": group_positive,
                "zero": group_zero,
                "negative": group_negative,
                "control_negative": control_negative_count(group, preferred_types),
                "observe_negative": observe_negative_count(group),
                "decision": "eval_only",
                "reason": "transfer_not_used_for_training",
            }
        )
    summary = {
        "split": split,
        "input_rows": len(finite_rows),
        "input_groups": len(grouped),
        "kept_rows": 0,
        "kept_groups": 0,
        "dropped_groups": 0,
        "positive": positive,
        "zero": zero,
        "negative": negative,
        "negative_rate": negative / len(finite_rows) if finite_rows else float("nan"),
        "control_negative_groups": sum(
            1
            for group in grouped.values()
            if any(row.get("type") in preferred_types and safe_float(row.get("oracle_delta_tc")) < 0.0 for row in group)
        ),
        "observe_negative_groups": sum(
            1
            for group in grouped.values()
            if any(row.get("type") == "observe" and safe_float(row.get("oracle_delta_tc")) < 0.0 for row in group)
        ),
        "meets_group_target": "eval_only",
        "meets_negative_rate_target": "eval_only",
    }
    return decisions, summary


def markdown_report(
    summaries: list[dict[str, Any]],
    train_target_groups: int,
    val_target_groups: int,
) -> str:
    lines = [
        "# Balanced Oracle Action Groups",
        "",
        f"generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "## Summary",
        "",
        "| split | kept groups | kept rows | negative rate | group target | neg-rate target |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {split} | {groups} | {rows} | {neg:.4f} | {gt} | {rt} |".format(
                split=row["split"],
                groups=row["kept_groups"],
                rows=row["kept_rows"],
                neg=safe_float(row["negative_rate"]),
                gt=row["meets_group_target"],
                rt=row["meets_negative_rate_target"],
            )
        )
    train = next((row for row in summaries if row["split"] == "train"), {})
    val = next((row for row in summaries if row["split"] == "expanded_val"), {})
    needs_more = int(train.get("meets_group_target", 0)) == 0 or int(val.get("meets_group_target", 0)) == 0
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- train minimum target: `{train_target_groups}` groups",
            f"- val minimum target: `{val_target_groups}` groups",
            f"- needs_more_oracle_collection: `{str(needs_more).lower()}`",
            "",
        ]
    )
    if needs_more:
        lines.append("Existing labels are not enough for the balanced-group target; collect negative-rich oracle groups next.")
    else:
        lines.append("Existing labels meet the group-count target; rescore these balanced subsets before rerunning the ranker.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-oracle", required=True)
    parser.add_argument("--val-oracle", required=True)
    parser.add_argument("--transfer-oracle", required=True)
    parser.add_argument("--min-negatives-per-group", type=int, default=3)
    parser.add_argument("--min-positives-per-group", type=int, default=3)
    parser.add_argument("--prefer-negative-types", default="control0,control1")
    parser.add_argument("--max-actions-per-group", type=int, default=18, help="Use 0 to keep every action in each kept group.")
    parser.add_argument("--min-train-groups", type=int, default=80)
    parser.add_argument("--min-val-groups", type=int, default=24)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preferred_types = set(parse_csv_values(args.prefer_negative_types))

    train_fields, train_rows = read_tsv(Path(args.train_oracle))
    val_fields, val_rows = read_tsv(Path(args.val_oracle))
    _, transfer_rows = read_tsv(Path(args.transfer_oracle))
    output_fields = merge_fieldnames(train_fields, val_fields)

    train_kept, train_decisions, train_summary = build_subset(
        split="train",
        rows=train_rows,
        min_negatives=args.min_negatives_per_group,
        min_positives=args.min_positives_per_group,
        preferred_types=preferred_types,
        max_actions_per_group=args.max_actions_per_group,
    )
    val_kept, val_decisions, val_summary = build_subset(
        split="expanded_val",
        rows=val_rows,
        min_negatives=args.min_negatives_per_group,
        min_positives=args.min_positives_per_group,
        preferred_types=preferred_types,
        max_actions_per_group=args.max_actions_per_group,
    )
    transfer_decisions, transfer_summary = build_eval_only_summary(
        split="transfer_eval_only",
        rows=transfer_rows,
        preferred_types=preferred_types,
    )
    train_summary = annotate_targets(train_summary, args.min_train_groups, 0.25, 0.55)
    val_summary = annotate_targets(val_summary, args.min_val_groups, 0.25, 0.60)
    summaries = [train_summary, val_summary, transfer_summary]

    write_tsv(out_dir / "balanced_train_oracle_actions.tsv", train_kept, output_fields)
    write_tsv(out_dir / "balanced_val_oracle_actions.tsv", val_kept, output_fields)
    write_tsv(out_dir / "group_decisions.tsv", [*train_decisions, *val_decisions, *transfer_decisions], GROUP_DECISION_FIELDS)
    write_tsv(out_dir / "balance_summary.tsv", summaries, SUMMARY_FIELDS)
    (out_dir / "balance_report.md").write_text(markdown_report(summaries, args.min_train_groups, args.min_val_groups))

    needs_more = int(train_summary["meets_group_target"]) == 0 or int(val_summary["meets_group_target"]) == 0
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "policy": {
            "min_negatives_per_group": args.min_negatives_per_group,
            "min_positives_per_group": args.min_positives_per_group,
            "prefer_negative_types": sorted(preferred_types),
            "max_actions_per_group": args.max_actions_per_group,
            "min_train_groups": args.min_train_groups,
            "min_val_groups": args.min_val_groups,
        },
        "inputs": {
            "train_oracle": args.train_oracle,
            "val_oracle": args.val_oracle,
            "transfer_oracle": args.transfer_oracle,
        },
        "fieldnames": {
            "train_fields": train_fields,
            "val_fields": val_fields,
            "output_fields": output_fields,
            "train_only": [field for field in train_fields if field not in val_fields],
            "val_only": [field for field in val_fields if field not in train_fields],
        },
        "summary": summaries,
        "needs_more_oracle_collection": needs_more,
        "note": "transfer rows are audited only and are not written into train/val balanced subsets",
    }
    write_json(out_dir / "balanced_manifest.json", manifest)
    handoff = {
        "mode": "fix",
        "status": "completed_needs_more_oracle" if needs_more else "completed",
        "objective": "balanced oracle action subsets",
        "out_dir": str(out_dir),
        "outputs": {
            "balanced_train": str(out_dir / "balanced_train_oracle_actions.tsv"),
            "balanced_val": str(out_dir / "balanced_val_oracle_actions.tsv"),
            "balance_report": str(out_dir / "balance_report.md"),
            "manifest": str(out_dir / "balanced_manifest.json"),
        },
        "needs_more_oracle_collection": needs_more,
        "summary": summaries,
    }
    write_json(out_dir / "handoff.json", handoff)
    print(json.dumps(handoff, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
