"""Audit oracle action group split/sign/action distributions."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any


SUMMARY_FIELDS = [
    "split",
    "rows",
    "groups",
    "positive",
    "zero",
    "negative",
    "negative_rate",
    "groups_with_negative",
    "all_positive_groups",
    "all_negative_groups",
    "min_delta_tc",
    "p05_delta_tc",
    "median_delta_tc",
    "mean_delta_tc",
    "p95_delta_tc",
    "max_delta_tc",
    "mean_group_best_delta_tc",
    "mean_group_worst_delta_tc",
]
BUCKET_FIELDS = [
    "split",
    "bucket",
    "value",
    "rows",
    "groups",
    "positive",
    "zero",
    "negative",
    "negative_rate",
    "mean_delta_tc",
    "min_delta_tc",
    "max_delta_tc",
]
GROUP_FIELDS = [
    "split",
    "checkpoint_name",
    "benchmark_id",
    "state_id",
    "candidate_strategy",
    "rows",
    "positive",
    "zero",
    "negative",
    "negative_rate",
    "control_negative",
    "observe_negative",
    "best_action",
    "best_delta_tc",
    "worst_action",
    "worst_delta_tc",
    "mean_delta_tc",
]
HIST_FIELDS = ["split", "negative_count", "groups"]
WORSENED_FIELDS = [
    "split",
    "variant",
    "checkpoint_name",
    "benchmark_id",
    "state_id",
    "candidate_strategy",
    "baseline_top1_action",
    "baseline_top1_delta_tc",
    "ranker_top1_action",
    "ranker_top1_delta_tc",
    "delta_vs_baseline",
    "ranker_top1_regret",
]


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"expected NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise ValueError(f"expected NAME=PATH, got {value!r}")
    return name, Path(path)


def safe_float(value: Any, default: float = float("nan")) -> float:
    if value in (None, "", "NA"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else float("nan")


def quantile(values: list[float], q: float) -> float:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return float("nan")
    if len(finite) == 1:
        return finite[0]
    pos = q * (len(finite) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return finite[lo]
    return finite[lo] * (hi - pos) + finite[hi] * (pos - lo)


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def group_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("checkpoint_name", "")),
        str(row.get("benchmark_id", "")),
        str(row.get("state_id", "")),
        str(row.get("candidate_strategy", "")),
    )


def finite_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if math.isfinite(safe_float(row.get("oracle_delta_tc")))]


def sign_counts(values: list[float]) -> tuple[int, int, int]:
    positive = sum(1 for value in values if value > 0.0)
    zero = sum(1 for value in values if value == 0.0)
    negative = sum(1 for value in values if value < 0.0)
    return positive, zero, negative


def group_rows(split: str, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in finite_rows(rows):
        grouped[group_key(row)].append(row)
    out = []
    for key, group in sorted(grouped.items()):
        values = [safe_float(row.get("oracle_delta_tc")) for row in group]
        positive, zero, negative = sign_counts(values)
        best = max(group, key=lambda row: safe_float(row.get("oracle_delta_tc")))
        worst = min(group, key=lambda row: safe_float(row.get("oracle_delta_tc")))
        control_negative = sum(
            1
            for row in group
            if row.get("type") in {"control0", "control1"} and safe_float(row.get("oracle_delta_tc")) < 0.0
        )
        observe_negative = sum(
            1 for row in group if row.get("type") == "observe" and safe_float(row.get("oracle_delta_tc")) < 0.0
        )
        out.append(
            {
                "split": split,
                "checkpoint_name": key[0],
                "benchmark_id": key[1],
                "state_id": key[2],
                "candidate_strategy": key[3],
                "rows": len(group),
                "positive": positive,
                "zero": zero,
                "negative": negative,
                "negative_rate": negative / len(group) if group else float("nan"),
                "control_negative": control_negative,
                "observe_negative": observe_negative,
                "best_action": best.get("action_key", ""),
                "best_delta_tc": safe_float(best.get("oracle_delta_tc")),
                "worst_action": worst.get("action_key", ""),
                "worst_delta_tc": safe_float(worst.get("oracle_delta_tc")),
                "mean_delta_tc": mean(values),
            }
        )
    return out


def summary_row(split: str, rows: list[dict[str, str]], groups: list[dict[str, Any]]) -> dict[str, Any]:
    finite = finite_rows(rows)
    values = [safe_float(row.get("oracle_delta_tc")) for row in finite]
    positive, zero, negative = sign_counts(values)
    best_values = [safe_float(group.get("best_delta_tc")) for group in groups]
    worst_values = [safe_float(group.get("worst_delta_tc")) for group in groups]
    return {
        "split": split,
        "rows": len(finite),
        "groups": len(groups),
        "positive": positive,
        "zero": zero,
        "negative": negative,
        "negative_rate": negative / len(finite) if finite else float("nan"),
        "groups_with_negative": sum(1 for group in groups if int(group["negative"]) > 0),
        "all_positive_groups": sum(1 for group in groups if int(group["positive"]) == int(group["rows"])),
        "all_negative_groups": sum(1 for group in groups if int(group["negative"]) == int(group["rows"])),
        "min_delta_tc": min(values) if values else float("nan"),
        "p05_delta_tc": quantile(values, 0.05),
        "median_delta_tc": quantile(values, 0.50),
        "mean_delta_tc": mean(values),
        "p95_delta_tc": quantile(values, 0.95),
        "max_delta_tc": max(values) if values else float("nan"),
        "mean_group_best_delta_tc": mean(best_values),
        "mean_group_worst_delta_tc": mean(worst_values),
    }


def bucket_rows(split: str, rows: list[dict[str, str]], bucket: str) -> list[dict[str, Any]]:
    grouped_by_bucket: dict[str, list[dict[str, str]]] = defaultdict(list)
    group_sets: dict[str, set[tuple[str, str, str, str]]] = defaultdict(set)
    for row in finite_rows(rows):
        value = str(row.get(bucket, ""))
        grouped_by_bucket[value].append(row)
        group_sets[value].add(group_key(row))
    out = []
    for value, bucket_rows_ in sorted(grouped_by_bucket.items()):
        deltas = [safe_float(row.get("oracle_delta_tc")) for row in bucket_rows_]
        positive, zero, negative = sign_counts(deltas)
        out.append(
            {
                "split": split,
                "bucket": bucket,
                "value": value,
                "rows": len(bucket_rows_),
                "groups": len(group_sets[value]),
                "positive": positive,
                "zero": zero,
                "negative": negative,
                "negative_rate": negative / len(bucket_rows_) if bucket_rows_ else float("nan"),
                "mean_delta_tc": mean(deltas),
                "min_delta_tc": min(deltas) if deltas else float("nan"),
                "max_delta_tc": max(deltas) if deltas else float("nan"),
            }
        )
    return out


def negative_count_hist(split: str, groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counter = Counter(int(group["negative"]) for group in groups)
    return [{"split": split, "negative_count": key, "groups": counter[key]} for key in sorted(counter)]


def load_worsened_groups(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    _, rows = read_tsv(path)
    out = []
    for row in rows:
        delta = safe_float(row.get("delta_vs_baseline"))
        if not math.isfinite(delta) or delta >= 0.0:
            continue
        out.append({**row, "delta_vs_baseline": delta})
    return sorted(out, key=lambda row: safe_float(row.get("delta_vs_baseline")))


def markdown_report(
    summaries: list[dict[str, Any]],
    by_type: list[dict[str, Any]],
    group_hist: list[dict[str, Any]],
    worsened: list[dict[str, Any]],
) -> str:
    lines = [
        "# Oracle Action Group Audit",
        "",
        f"generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "## Split Summary",
        "",
        "| split | rows | groups | negative rate | all-positive groups | mean delta | min delta | max delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {split} | {rows} | {groups} | {neg:.4f} | {all_pos} | {mean:.6f} | {minv:.6f} | {maxv:.6f} |".format(
                split=row["split"],
                rows=row["rows"],
                groups=row["groups"],
                neg=safe_float(row["negative_rate"]),
                all_pos=row["all_positive_groups"],
                mean=safe_float(row["mean_delta_tc"]),
                minv=safe_float(row["min_delta_tc"]),
                maxv=safe_float(row["max_delta_tc"]),
            )
        )

    lines.extend(
        [
            "",
            "## Action Type Summary",
            "",
            "| split | type | rows | negative rate | mean delta |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in by_type:
        lines.append(
            f"| {row['split']} | `{row['value']}` | {row['rows']} | {safe_float(row['negative_rate']):.4f} | {safe_float(row['mean_delta_tc']):.6f} |"
        )

    lines.extend(["", "## Group Negative Count Histogram", "", "| split | negative actions in group | groups |", "|---|---:|---:|"])
    for row in group_hist:
        lines.append(f"| {row['split']} | {row['negative_count']} | {row['groups']} |")

    if worsened:
        lines.extend(
            [
                "",
                "## Ranker-Worsened Groups",
                "",
                "| split | benchmark | strategy | baseline top1 | ranker top1 | delta vs baseline |",
                "|---|---|---|---:|---:|---:|",
            ]
        )
        for row in worsened[:20]:
            lines.append(
                "| {split} | `{bench}` | `{strategy}` | {base:.6f} | {ranker:.6f} | {delta:.6f} |".format(
                    split=row.get("split", ""),
                    bench=row.get("benchmark_id", ""),
                    strategy=row.get("candidate_strategy", ""),
                    base=safe_float(row.get("baseline_top1_delta_tc")),
                    ranker=safe_float(row.get("ranker_top1_delta_tc")),
                    delta=safe_float(row.get("delta_vs_baseline")),
                )
            )

    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "- Balance by group, not only by rows.",
            "- Require both positive and negative actions inside a group for rank training.",
            "- Prefer negative `control0` / `control1` examples because transfer failures concentrate there.",
            "- Keep transfer evaluation-only; do not train on `b15_C` or `i2c_aig` transfer rows.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-tsv", action="append", required=True, help="Repeat NAME=PATH.")
    parser.add_argument("--ranker-top1-deltas", default=None)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    all_groups: list[dict[str, Any]] = []
    by_type: list[dict[str, Any]] = []
    by_strategy: list[dict[str, Any]] = []
    by_benchmark: list[dict[str, Any]] = []
    hist_rows: list[dict[str, Any]] = []
    inputs = {}

    for item in args.oracle_tsv:
        split, path = parse_named_path(item)
        inputs[split] = str(path)
        _, rows = read_tsv(path)
        groups = group_rows(split, rows)
        summaries.append(summary_row(split, rows, groups))
        all_groups.extend(groups)
        by_type.extend(bucket_rows(split, rows, "type"))
        by_strategy.extend(bucket_rows(split, rows, "candidate_strategy"))
        by_benchmark.extend(bucket_rows(split, rows, "benchmark_id"))
        hist_rows.extend(negative_count_hist(split, groups))

    worsened = load_worsened_groups(Path(args.ranker_top1_deltas) if args.ranker_top1_deltas else None)

    write_tsv(out_dir / "oracle_group_audit_summary.tsv", summaries, SUMMARY_FIELDS)
    write_tsv(out_dir / "oracle_group_audit_by_group.tsv", all_groups, GROUP_FIELDS)
    write_tsv(out_dir / "oracle_group_audit_by_action_type.tsv", by_type, BUCKET_FIELDS)
    write_tsv(out_dir / "oracle_group_audit_by_strategy.tsv", by_strategy, BUCKET_FIELDS)
    write_tsv(out_dir / "oracle_group_audit_by_benchmark.tsv", by_benchmark, BUCKET_FIELDS)
    write_tsv(out_dir / "oracle_group_negative_count_hist.tsv", hist_rows, HIST_FIELDS)
    write_tsv(out_dir / "ranker_worsened_groups.tsv", worsened, WORSENED_FIELDS)
    (out_dir / "oracle_group_audit_report.md").write_text(markdown_report(summaries, by_type, hist_rows, worsened))
    handoff = {
        "mode": "fix",
        "status": "completed",
        "objective": "oracle action group split audit",
        "out_dir": str(out_dir),
        "inputs": inputs,
        "ranker_top1_deltas": args.ranker_top1_deltas,
        "outputs": {
            "report": str(out_dir / "oracle_group_audit_report.md"),
            "summary": str(out_dir / "oracle_group_audit_summary.tsv"),
            "by_group": str(out_dir / "oracle_group_audit_by_group.tsv"),
        },
        "summary": summaries,
    }
    write_json(out_dir / "handoff.json", handoff)
    print(json.dumps(handoff, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
