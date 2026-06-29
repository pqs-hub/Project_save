"""Evaluate checkpoints on fixed backend-labeled oracle action groups."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oracle_action_value_probe import (  # noqa: E402
    ORACLE_FIELDS,
    PRED_FIELDS,
    SCORE_FIELDS,
    action_key,
    checkpoint_id,
    metric_rows_for_group,
    mean,
    parse_csv_values,
    parse_int_values,
    read_tsv,
    safe_float,
    score_actions,
    write_json,
    write_tsv,
)
from tpi_jepa.bench import parse_bench  # noqa: E402
from tpi_jepa.graph import build_graph  # noqa: E402
from tpi_jepa.labels import find_bench_path  # noqa: E402
from tpi_jepa.plan import load_checkpoint, set_real_fault_context  # noqa: E402


RESCORED_FIELDS = ["checkpoint_name", *ORACLE_FIELDS]
METRIC_FIELDS = ["checkpoint_name", *PRED_FIELDS]
SUMMARY_FIELDS = [
    "checkpoint_name",
    "score_field",
    "groups",
    "mean_spearman",
    "mean_kendall_tau",
    "mean_pearson",
    "mean_top1_real_delta_tc",
    "mean_top1_regret",
    "negative_top1_rate",
    "mean_sign_accuracy",
    "mean_calibration_slope",
    "mean_calibration_intercept",
    "verdict_vs_baseline",
]


def parse_checkpoint_specs(values: list[str]) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    for value in values:
        for item in parse_csv_values(value):
            if "=" not in item:
                raise ValueError(f"checkpoint spec must be NAME=PATH, got {item!r}")
            name, path = item.split("=", 1)
            name = name.strip()
            path = path.strip()
            if not name or not path:
                raise ValueError(f"checkpoint spec must be NAME=PATH, got {item!r}")
            specs.append((name, path))
    seen: set[str] = set()
    duplicates = []
    for name, _ in specs:
        if name in seen:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        raise ValueError(f"duplicate checkpoint names: {sorted(set(duplicates))}")
    return specs


def group_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("benchmark_id", "")),
        str(row.get("state_id", "")),
        str(row.get("candidate_strategy", "")),
    )


def aggregate_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["checkpoint_name"]), str(row["score_field"]))].append(row)
    summary = []
    for (checkpoint_name, score_field), group in sorted(grouped.items()):
        negative_values = [safe_float(row.get("negative_top1")) for row in group]
        negative_rate = mean([value for value in negative_values if math.isfinite(value)])
        summary.append(
            {
                "checkpoint_name": checkpoint_name,
                "score_field": score_field,
                "groups": len(group),
                "mean_spearman": mean([safe_float(row.get("spearman")) for row in group]),
                "mean_kendall_tau": mean([safe_float(row.get("kendall_tau")) for row in group]),
                "mean_pearson": mean([safe_float(row.get("pearson")) for row in group]),
                "mean_top1_real_delta_tc": mean([safe_float(row.get("top1_real_delta_tc")) for row in group]),
                "mean_top1_regret": mean([safe_float(row.get("top1_regret")) for row in group]),
                "negative_top1_rate": negative_rate,
                "mean_sign_accuracy": mean([safe_float(row.get("sign_accuracy")) for row in group]),
                "mean_calibration_slope": mean([safe_float(row.get("calibration_slope")) for row in group]),
                "mean_calibration_intercept": mean([safe_float(row.get("calibration_intercept")) for row in group]),
                "verdict_vs_baseline": "INCONCLUSIVE",
            }
        )
    return summary


def best_summary_for_checkpoint(summary: list[dict[str, Any]], checkpoint_name: str) -> dict[str, Any] | None:
    rows = [row for row in summary if row["checkpoint_name"] == checkpoint_name]
    if not rows:
        return None
    return max(rows, key=lambda row: safe_float(row.get("mean_spearman"), float("-inf")))


def assign_verdicts(summary: list[dict[str, Any]], baseline: str | None) -> dict[str, str]:
    verdicts: dict[str, str] = {}
    if not baseline:
        for row in summary:
            verdicts[row["checkpoint_name"]] = "INCONCLUSIVE"
        return verdicts
    baseline_best = best_summary_for_checkpoint(summary, baseline)
    if baseline_best is None:
        raise ValueError(f"baseline checkpoint {baseline!r} not found in summaries")
    base_spearman = safe_float(baseline_best.get("mean_spearman"))
    base_neg = safe_float(baseline_best.get("negative_top1_rate"))
    base_regret = safe_float(baseline_best.get("mean_top1_regret"))
    checkpoints = sorted({row["checkpoint_name"] for row in summary})
    for checkpoint_name in checkpoints:
        if checkpoint_name == baseline:
            verdicts[checkpoint_name] = "INCONCLUSIVE"
            continue
        candidate = best_summary_for_checkpoint(summary, checkpoint_name)
        if candidate is None:
            verdicts[checkpoint_name] = "INCONCLUSIVE"
            continue
        spearman = safe_float(candidate.get("mean_spearman"))
        neg = safe_float(candidate.get("negative_top1_rate"))
        regret = safe_float(candidate.get("mean_top1_regret"))
        if math.isfinite(spearman) and math.isfinite(base_spearman) and spearman > base_spearman and neg > base_neg:
            verdicts[checkpoint_name] = "REJECT"
        elif (
            math.isfinite(spearman)
            and math.isfinite(neg)
            and math.isfinite(regret)
            and spearman >= base_spearman + 0.10
            and neg <= base_neg - 0.10
            and regret <= base_regret - 0.01
        ):
            verdicts[checkpoint_name] = "PROMOTE"
        else:
            verdicts[checkpoint_name] = "INCONCLUSIVE"
    return verdicts


def apply_verdicts(summary: list[dict[str, Any]], verdicts: dict[str, str]) -> None:
    for row in summary:
        row["verdict_vs_baseline"] = verdicts.get(row["checkpoint_name"], "INCONCLUSIVE")


def markdown_report(summary: list[dict[str, Any]], verdicts: dict[str, str], baseline: str | None) -> str:
    lines = [
        "# Oracle Action-Value Checkpoint Gate",
        "",
        f"generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        f"baseline: `{baseline or 'none'}`",
        "",
        "## Verdicts",
        "",
        "| checkpoint | verdict |",
        "|---|---|",
    ]
    for checkpoint_name, verdict in sorted(verdicts.items()):
        lines.append(f"| `{checkpoint_name}` | `{verdict}` |")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| checkpoint | score_field | groups | mean Spearman | negative top1 rate | mean top1 regret | verdict |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in summary:
        lines.append(
            "| `{checkpoint}` | `{field}` | {groups} | {spearman:.6g} | {neg:.6g} | {regret:.6g} | `{verdict}` |".format(
                checkpoint=row["checkpoint_name"],
                field=row["score_field"],
                groups=row["groups"],
                spearman=safe_float(row.get("mean_spearman")),
                neg=safe_float(row.get("negative_top1_rate")),
                regret=safe_float(row.get("mean_top1_regret")),
                verdict=row["verdict_vs_baseline"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-actions", required=True)
    parser.add_argument("--checkpoints", action="append", default=[], help="Comma-separated NAME=PATH checkpoint specs.")
    parser.add_argument("--checkpoint", action="append", default=[], help="Repeatable NAME=PATH checkpoint spec.")
    parser.add_argument("--bench-root", default=None)
    parser.add_argument("--score-fields", default="reward_pred,guarded_reward,hard_reduction_total_pred,hybrid_pred")
    parser.add_argument("--top-ks", default="8,16,32")
    parser.add_argument("--oracle-top-m", type=int, default=5)
    parser.add_argument("--plan-device", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--baseline", default=None)
    args = parser.parse_args()

    checkpoint_specs = parse_checkpoint_specs([*args.checkpoints, *args.checkpoint])
    if not checkpoint_specs:
        raise ValueError("at least one --checkpoints or --checkpoint NAME=PATH is required")
    if args.bench_root:
        os.environ["TPI_BENCH_ROOT"] = args.bench_root
    score_fields = parse_csv_values(args.score_fields)
    unsupported = [field for field in score_fields if field not in SCORE_FIELDS]
    if unsupported:
        raise ValueError(f"Unsupported score fields: {unsupported}")
    ks = parse_int_values(args.top_ks)
    oracle_rows = read_tsv(Path(args.oracle_actions))
    grouped_rows: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in oracle_rows:
        if row.get("state_id") != "initial":
            raise ValueError("evaluate_oracle_action_values.py v1 supports only state_id=initial")
        grouped_rows[group_key(row)].append(row)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device_name = args.plan_device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    rescored_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    checkpoint_manifest = []
    graph_cache = {}

    for checkpoint_name, checkpoint_path in checkpoint_specs:
        model, config = load_checkpoint(checkpoint_path, device)
        checkpoint_manifest.append(
            {
                "name": checkpoint_name,
                "path": checkpoint_path,
                "checkpoint_id": checkpoint_id(checkpoint_path),
            }
        )
        for (benchmark_id, state_id, strategy), rows in sorted(grouped_rows.items()):
            set_real_fault_context(benchmark_id, None, None)
            if benchmark_id not in graph_cache:
                graph_cache[benchmark_id] = build_graph(parse_bench(find_bench_path(benchmark_id)))
            graph = graph_cache[benchmark_id]
            selected: list[tuple[str, str]] = []
            candidates = [(row["node"], row["type"]) for row in rows]
            scored = score_actions(
                model=model,
                graph=graph,
                selected=selected,
                candidates=candidates,
                config=config,
                device=device,
                candidate_diversity_penalty=0.0,
                candidate_diversity_depth=4,
            )
            group_rescored = []
            for row in rows:
                key = action_key(row["node"], row["type"])
                scored_row = scored[key]
                merged = {
                    **row,
                    **{field: scored_row.get(field) for field in ORACLE_FIELDS if field in scored_row},
                }
                merged["checkpoint_name"] = checkpoint_name
                rescored_rows.append(merged)
                group_rescored.append(merged)
            _, group_pred, _ = metric_rows_for_group(
                rows=group_rescored,
                score_fields=score_fields,
                ks=ks,
                oracle_top_m=args.oracle_top_m,
            )
            for metric in group_pred:
                metric["checkpoint_name"] = checkpoint_name
                metric_rows.append(metric)

    summary_rows = aggregate_metrics(metric_rows)
    verdicts = assign_verdicts(summary_rows, args.baseline)
    apply_verdicts(summary_rows, verdicts)

    write_tsv(out_dir / "rescored_oracle_actions.tsv", rescored_rows, RESCORED_FIELDS)
    write_tsv(out_dir / "oracle_action_value_metrics.tsv", metric_rows, METRIC_FIELDS)
    write_tsv(out_dir / "oracle_action_value_summary.tsv", summary_rows, SUMMARY_FIELDS)
    (out_dir / "oracle_action_value_report.md").write_text(markdown_report(summary_rows, verdicts, args.baseline))
    handoff = {
        "mode": "fix",
        "status": "progressed",
        "objective": "fixed oracle action-value checkpoint comparison gate",
        "out_dir": str(out_dir),
        "oracle_actions": str(args.oracle_actions),
        "baseline": args.baseline,
        "checkpoints": checkpoint_manifest,
        "verdicts": verdicts,
        "records": {
            "rescored_oracle_actions": len(rescored_rows),
            "oracle_action_value_metrics": len(metric_rows),
            "oracle_action_value_summary": len(summary_rows),
        },
        "outputs": {
            "rescored_oracle_actions": str(out_dir / "rescored_oracle_actions.tsv"),
            "oracle_action_value_metrics": str(out_dir / "oracle_action_value_metrics.tsv"),
            "oracle_action_value_summary": str(out_dir / "oracle_action_value_summary.tsv"),
            "oracle_action_value_report": str(out_dir / "oracle_action_value_report.md"),
        },
    }
    write_json(out_dir / "handoff.json", handoff)
    print(json.dumps(handoff, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
