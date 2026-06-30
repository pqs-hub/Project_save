"""Evaluate post-hoc Q-score calibration on fixed oracle action groups."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from oracle_action_value_probe import (  # noqa: E402
    PRED_FIELDS,
    metric_rows_for_group,
    parse_csv_values,
    read_tsv,
    safe_float,
    spearman,
    write_tsv,
)


CAL_SCORE_FIELD = "q_calibrated"
DEFAULT_CHECKPOINTS = ["Q_v0_rank0p5", "Q_v0_rank1p0", "Q_v0_rank2p0", "Q_v0_value1_rank1"]
DEFAULT_METHODS = ["raw", "group_center", "group_zscore", "group_rank_pct", "circuit_zscore", "global_zscore", "platt"]
GROUP_KEY = ["checkpoint_name", "benchmark_id", "state_id", "candidate_strategy"]
SUMMARY_FIELDS = [
    "split",
    "checkpoint_name",
    "method",
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
    "rank_changed_groups",
    "rank_changed_rate",
]
PROMOTION_FIELDS = [
    "checkpoint_name",
    "method",
    "expanded_spearman",
    "expanded_negative_top1",
    "expanded_top1_real_delta",
    "expanded_top1_regret",
    "transfer_spearman",
    "transfer_negative_top1",
    "transfer_top1_real_delta",
    "transfer_top1_regret",
    "transfer_sign_accuracy",
    "rank_changed_groups_expanded",
    "rank_changed_groups_transfer",
    "verdict",
    "reasons",
]
ACTION_FIELDS = [
    "split",
    "checkpoint_name",
    "method",
    "benchmark_id",
    "state_id",
    "candidate_strategy",
    "candidate_rank",
    "node",
    "type",
    "action_key",
    "oracle_delta_tc",
    "q_pred",
    CAL_SCORE_FIELD,
]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def finite_values(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def mean(values: list[float]) -> float:
    vals = finite_values(values)
    return sum(vals) / len(vals) if vals else float("nan")


def std(values: list[float]) -> float:
    vals = finite_values(values)
    if len(vals) <= 1:
        return 0.0
    mu = mean(vals)
    return math.sqrt(sum((value - mu) ** 2 for value in vals) / len(vals))


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def compute_scores(values: Any, transform: str = "raw") -> list[float]:
    """Pure calibration helper used by unit tests for ordering invariants."""

    qs = [float(value) for value in values]
    if transform == "raw":
        return qs
    if transform == "group_center":
        mu = mean(qs)
        return [q - mu for q in qs]
    if transform in {"group_zscore", "global_zscore", "circuit_zscore"}:
        mu = mean(qs)
        sigma = std(qs) or 1.0
        return [(q - mu) / sigma for q in qs]
    if transform == "group_rank_pct":
        if not qs:
            return []
        order = sorted(range(len(qs)), key=lambda idx: qs[idx])
        denom = max(1, len(qs) - 1)
        out = [0.0 for _ in qs]
        for rank, idx in enumerate(order):
            out[idx] = float(rank) / float(denom)
        return out
    if transform == "platt":
        return [sigmoid(q) for q in qs]
    raise ValueError(f"unsupported transform {transform!r}")


def compute_spearman(pred: Any, target: Any) -> float:
    return spearman([float(value) for value in pred], [float(value) for value in target])


def list_variants(
    checkpoints: list[str] | None = None,
    methods: list[str] | None = None,
) -> list[tuple[str, str]]:
    checkpoint_values = checkpoints or DEFAULT_CHECKPOINTS
    method_values = methods or DEFAULT_METHODS
    return [(checkpoint, method) for checkpoint in checkpoint_values for method in method_values]


def group_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return tuple(str(row.get(field, "")) for field in GROUP_KEY)  # type: ignore[return-value]


def circuit_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("checkpoint_name", "")), str(row.get("benchmark_id", ""))


def checkpoint_key(row: dict[str, Any]) -> str:
    return str(row.get("checkpoint_name", ""))


def fit_platt(rows: list[dict[str, str]], score_field: str) -> dict[str, tuple[float, float]]:
    """Fit one logistic q->P(delta>0) calibrator per checkpoint on expanded rows."""

    by_checkpoint: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        q = safe_float(row.get(score_field))
        delta = safe_float(row.get("oracle_delta_tc"))
        if math.isfinite(q) and math.isfinite(delta):
            by_checkpoint[checkpoint_key(row)].append((q, 1.0 if delta > 0.0 else 0.0))

    fitted: dict[str, tuple[float, float]] = {}
    for checkpoint, pairs in by_checkpoint.items():
        if len(pairs) < 2 or len({label for _, label in pairs}) < 2:
            fitted[checkpoint] = (1.0, 0.0)
            continue
        qs = [q for q, _ in pairs]
        q_mu = mean(qs)
        q_sigma = std(qs) or 1.0
        norm_pairs = [((q - q_mu) / q_sigma, label) for q, label in pairs]
        a = 1.0
        b = 0.0
        lr = 0.05
        l2 = 1e-3
        for _ in range(800):
            grad_a = l2 * a
            grad_b = 0.0
            for q, label in norm_pairs:
                p = sigmoid(a * q + b)
                err = p - label
                grad_a += err * q / len(norm_pairs)
                grad_b += err / len(norm_pairs)
            a -= lr * grad_a
            b -= lr * grad_b
        # Store parameters in the original q scale.
        fitted[checkpoint] = (a / q_sigma, b - a * q_mu / q_sigma)
    return fitted


def add_context_stats(rows_by_split: dict[str, list[dict[str, str]]], score_field: str) -> dict[str, Any]:
    expanded = rows_by_split["expanded"]
    global_stats: dict[str, tuple[float, float]] = {}
    for checkpoint in sorted({checkpoint_key(row) for row in expanded}):
        vals = [safe_float(row.get(score_field)) for row in expanded if checkpoint_key(row) == checkpoint]
        vals = finite_values(vals)
        global_stats[checkpoint] = (mean(vals), std(vals) or 1.0)

    stats = {
        "global": global_stats,
        "platt": fit_platt(expanded, score_field),
        "group": {},
        "circuit": {},
    }
    for split, rows in rows_by_split.items():
        by_group: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
        by_circuit: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in rows:
            q = safe_float(row.get(score_field))
            if not math.isfinite(q):
                continue
            by_group[group_key(row)].append(q)
            by_circuit[circuit_key(row)].append(q)
        stats["group"][split] = {key: (mean(vals), std(vals) or 1.0) for key, vals in by_group.items()}
        stats["circuit"][split] = {key: (mean(vals), std(vals) or 1.0) for key, vals in by_circuit.items()}
    return stats


def rank_percentiles(group: list[dict[str, Any]], score_field: str) -> dict[str, float]:
    scored = [
        (idx, safe_float(row.get(score_field)))
        for idx, row in enumerate(group)
        if math.isfinite(safe_float(row.get(score_field)))
    ]
    if not scored:
        return {}
    scored.sort(key=lambda item: item[1])
    denom = max(1, len(scored) - 1)
    values: dict[str, float] = {}
    for rank, (idx, _) in enumerate(scored):
        values[str(idx)] = float(rank) / float(denom)
    return values


def calibrated_value(
    row: dict[str, str],
    method: str,
    split: str,
    score_field: str,
    stats: dict[str, Any],
    rank_pct: float | None,
) -> float:
    q = safe_float(row.get(score_field))
    if not math.isfinite(q):
        return float("nan")
    if method == "raw":
        return q
    if method == "group_center":
        mu, _ = stats["group"][split].get(group_key(row), (0.0, 1.0))
        return q - mu
    if method == "group_zscore":
        mu, sigma = stats["group"][split].get(group_key(row), (0.0, 1.0))
        return (q - mu) / sigma
    if method == "group_rank_pct":
        return rank_pct if rank_pct is not None else float("nan")
    if method == "circuit_zscore":
        mu, sigma = stats["circuit"][split].get(circuit_key(row), (0.0, 1.0))
        return (q - mu) / sigma
    if method == "global_zscore":
        mu, sigma = stats["global"].get(checkpoint_key(row), (0.0, 1.0))
        return (q - mu) / sigma
    if method == "platt":
        a, b = stats["platt"].get(checkpoint_key(row), (1.0, 0.0))
        return sigmoid(a * q + b)
    raise ValueError(f"unsupported method {method!r}")


def top_action(group: list[dict[str, Any]], field: str) -> str:
    valid = [row for row in group if math.isfinite(safe_float(row.get(field)))]
    if not valid:
        return ""
    return str(max(valid, key=lambda row: safe_float(row.get(field))).get("action_key", ""))


def aggregate_pred_rows(split: str, method: str, pred_rows: list[dict[str, Any]], rank_changed: int) -> dict[str, Any]:
    return {
        "split": split,
        "checkpoint_name": pred_rows[0]["checkpoint_name"] if pred_rows else "",
        "method": method,
        "groups": len(pred_rows),
        "mean_spearman": mean([safe_float(row.get("spearman")) for row in pred_rows]),
        "mean_kendall_tau": mean([safe_float(row.get("kendall_tau")) for row in pred_rows]),
        "mean_pearson": mean([safe_float(row.get("pearson")) for row in pred_rows]),
        "mean_top1_real_delta_tc": mean([safe_float(row.get("top1_real_delta_tc")) for row in pred_rows]),
        "mean_top1_regret": mean([safe_float(row.get("top1_regret")) for row in pred_rows]),
        "negative_top1_rate": mean([safe_float(row.get("negative_top1")) for row in pred_rows]),
        "mean_sign_accuracy": mean([safe_float(row.get("sign_accuracy")) for row in pred_rows]),
        "mean_calibration_slope": mean([safe_float(row.get("calibration_slope")) for row in pred_rows]),
        "mean_calibration_intercept": mean([safe_float(row.get("calibration_intercept")) for row in pred_rows]),
        "rank_changed_groups": rank_changed,
        "rank_changed_rate": rank_changed / max(1, len(pred_rows)),
    }


def evaluate_split(
    split: str,
    rows: list[dict[str, str]],
    checkpoints: set[str],
    methods: list[str],
    score_field: str,
    stats: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if checkpoint_key(row) in checkpoints:
            grouped[group_key(row)].append(row)

    action_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    pred_rows_by_method: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    rank_changed_by_method: dict[tuple[str, str], int] = defaultdict(int)

    for key, group in sorted(grouped.items()):
        raw_top = top_action(group, score_field)
        rank_pct = rank_percentiles(group, score_field)
        for method in methods:
            calibrated_group: list[dict[str, Any]] = []
            for idx, row in enumerate(group):
                q_cal = calibrated_value(row, method, split, score_field, stats, rank_pct.get(str(idx)))
                new_row = dict(row)
                new_row[CAL_SCORE_FIELD] = q_cal
                calibrated_group.append(new_row)
                action_rows.append(
                    {
                        "split": split,
                        "checkpoint_name": row.get("checkpoint_name", ""),
                        "method": method,
                        "benchmark_id": row.get("benchmark_id", ""),
                        "state_id": row.get("state_id", ""),
                        "candidate_strategy": row.get("candidate_strategy", ""),
                        "candidate_rank": row.get("candidate_rank", ""),
                        "node": row.get("node", ""),
                        "type": row.get("type", ""),
                        "action_key": row.get("action_key", ""),
                        "oracle_delta_tc": row.get("oracle_delta_tc", ""),
                        "q_pred": row.get(score_field, ""),
                        CAL_SCORE_FIELD: q_cal,
                    }
                )
            _, group_pred, _ = metric_rows_for_group(
                rows=calibrated_group,
                score_fields=[CAL_SCORE_FIELD],
                ks=[1, 3, 5],
                oracle_top_m=1,
            )
            for pred in group_pred:
                pred = dict(pred)
                pred["split"] = split
                pred["checkpoint_name"] = key[0]
                pred["method"] = method
                pred["score_field"] = CAL_SCORE_FIELD
                metric_rows.append(pred)
                pred_rows_by_method[(key[0], method)].append(pred)
            if raw_top != top_action(calibrated_group, CAL_SCORE_FIELD):
                rank_changed_by_method[(key[0], method)] += 1

    for (checkpoint, method), pred_rows in sorted(pred_rows_by_method.items()):
        summary_rows.append(
            aggregate_pred_rows(split, method, pred_rows, rank_changed_by_method.get((checkpoint, method), 0))
            | {"checkpoint_name": checkpoint}
        )
    return action_rows, metric_rows, summary_rows


def summary_lookup(rows: list[dict[str, Any]], split: str, checkpoint: str, method: str) -> dict[str, Any]:
    for row in rows:
        if row.get("split") == split and row.get("checkpoint_name") == checkpoint and row.get("method") == method:
            return row
    return {}


def f(row: dict[str, Any], key: str) -> float:
    return safe_float(row.get(key))


def promotion_rows(summary_rows: list[dict[str, Any]], checkpoints: list[str], methods: list[str]) -> list[dict[str, Any]]:
    rows = []
    for checkpoint in checkpoints:
        for method in methods:
            erow = summary_lookup(summary_rows, "expanded", checkpoint, method)
            trow = summary_lookup(summary_rows, "transfer", checkpoint, method)
            row = {
                "checkpoint_name": checkpoint,
                "method": method,
                "expanded_spearman": f(erow, "mean_spearman"),
                "expanded_negative_top1": f(erow, "negative_top1_rate"),
                "expanded_top1_real_delta": f(erow, "mean_top1_real_delta_tc"),
                "expanded_top1_regret": f(erow, "mean_top1_regret"),
                "transfer_spearman": f(trow, "mean_spearman"),
                "transfer_negative_top1": f(trow, "negative_top1_rate"),
                "transfer_top1_real_delta": f(trow, "mean_top1_real_delta_tc"),
                "transfer_top1_regret": f(trow, "mean_top1_regret"),
                "transfer_sign_accuracy": f(trow, "mean_sign_accuracy"),
                "rank_changed_groups_expanded": erow.get("rank_changed_groups", ""),
                "rank_changed_groups_transfer": trow.get("rank_changed_groups", ""),
            }
            reasons = []
            if row["expanded_spearman"] < 0.50:
                reasons.append("expanded_spearman_below_0p50")
            if row["expanded_negative_top1"] > 0.162:
                reasons.append("expanded_negative_top1_worse")
            if row["transfer_negative_top1"] > 0.167:
                reasons.append("transfer_negative_top1_worse")
            if row["transfer_top1_regret"] > 0.012552:
                reasons.append("transfer_regret_worse_than_incumbent")
            if row["transfer_spearman"] < 0.20:
                reasons.append("transfer_spearman_below_0p20")
            row["verdict"] = "PROMOTE_Q_CALIBRATED" if not reasons else "REJECT"
            row["reasons"] = ",".join(reasons)
            rows.append(row)
    return rows


def markdown_report(promo_rows: list[dict[str, Any]], out_dir: Path) -> str:
    promoted = [row for row in promo_rows if row["verdict"] == "PROMOTE_Q_CALIBRATED"]
    best_expanded = max(promo_rows, key=lambda row: safe_float(row.get("expanded_spearman")), default={})
    best_transfer = min(promo_rows, key=lambda row: safe_float(row.get("transfer_top1_regret")), default={})
    lines = [
        "# Q Calibration Fixed-Candidate Ablation",
        "",
        f"generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "## Summary",
        "",
        f"promoted: `{len(promoted)}`",
        "",
        "| checkpoint | method | verdict | expanded Spearman | expanded neg top1 | transfer Spearman | transfer neg top1 | transfer regret | rank changes expanded/transfer |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in promo_rows:
        lines.append(
            "| {checkpoint_name} | `{method}` | {verdict} | {expanded_spearman:.6f} | {expanded_negative_top1:.6f} | "
            "{transfer_spearman:.6f} | {transfer_negative_top1:.6f} | {transfer_top1_regret:.6f} | {rank_changed_groups_expanded}/{rank_changed_groups_transfer} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Best Expanded",
            "",
            f"- checkpoint: `{best_expanded.get('checkpoint_name', '')}`",
            f"- method: `{best_expanded.get('method', '')}`",
            f"- expanded Spearman: `{best_expanded.get('expanded_spearman', '')}`",
            "",
            "## Best Transfer Regret",
            "",
            f"- checkpoint: `{best_transfer.get('checkpoint_name', '')}`",
            f"- method: `{best_transfer.get('method', '')}`",
            f"- transfer regret: `{best_transfer.get('transfer_top1_regret', '')}`",
            "",
            "## Important Note",
            "",
            "Most calibration methods here are monotonic transforms of `q_pred` within each fixed candidate group. "
            "They can improve score scale and sign calibration, but they cannot change top1 selection unless the transform changes action order. "
            "`rank_changed_groups` explicitly records whether action ordering changed.",
            "",
            "## Outputs",
            "",
            f"- `{out_dir / 'q_calibration_promotion.tsv'}`",
            f"- `{out_dir / 'q_calibration_summary.tsv'}`",
            f"- `{out_dir / 'q_calibrated_actions.tsv'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expanded-rescored", required=True)
    parser.add_argument("--transfer-rescored", required=True)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--score-field", default="q_pred")
    parser.add_argument(
        "--methods",
        default="raw,group_center,group_zscore,group_rank_pct,circuit_zscore,global_zscore,platt",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("autoresearch/q-calibration-260630"))
    args = parser.parse_args()

    checkpoints = parse_csv_values(args.checkpoints)
    methods = parse_csv_values(args.methods)
    checkpoint_set = set(checkpoints)
    rows_by_split = {
        "expanded": read_tsv(Path(args.expanded_rescored)),
        "transfer": read_tsv(Path(args.transfer_rescored)),
    }
    stats = add_context_stats(rows_by_split, args.score_field)

    all_actions: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []
    all_summary: list[dict[str, Any]] = []
    for split, rows in rows_by_split.items():
        actions, metrics, summary = evaluate_split(split, rows, checkpoint_set, methods, args.score_field, stats)
        all_actions.extend(actions)
        all_metrics.extend(metrics)
        all_summary.extend(summary)

    promo = promotion_rows(all_summary, checkpoints, methods)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(args.out_dir / "q_calibrated_actions.tsv", all_actions, ACTION_FIELDS)
    write_tsv(args.out_dir / "q_calibration_metrics.tsv", all_metrics, ["split", "checkpoint_name", "method", *PRED_FIELDS])
    write_tsv(args.out_dir / "q_calibration_summary.tsv", all_summary, SUMMARY_FIELDS)
    write_tsv(args.out_dir / "q_calibration_promotion.tsv", promo, PROMOTION_FIELDS)
    (args.out_dir / "q_calibration_report.md").write_text(markdown_report(promo, args.out_dir))
    write_json(
        args.out_dir / "handoff.json",
        {
            "mode": "fix",
            "objective": "post-hoc Q calibration fixed-candidate ablation",
            "status": "completed",
            "out_dir": str(args.out_dir),
            "inputs": {
                "expanded_rescored": args.expanded_rescored,
                "transfer_rescored": args.transfer_rescored,
                "checkpoints": checkpoints,
                "score_field": args.score_field,
                "methods": methods,
            },
            "outputs": {
                "actions": str(args.out_dir / "q_calibrated_actions.tsv"),
                "metrics": str(args.out_dir / "q_calibration_metrics.tsv"),
                "summary": str(args.out_dir / "q_calibration_summary.tsv"),
                "promotion": str(args.out_dir / "q_calibration_promotion.tsv"),
                "report": str(args.out_dir / "q_calibration_report.md"),
            },
            "promoted": [
                {"checkpoint": row["checkpoint_name"], "method": row["method"]}
                for row in promo
                if row["verdict"] == "PROMOTE_Q_CALIBRATED"
            ],
        },
    )
    print((args.out_dir / "handoff.json").read_text())


if __name__ == "__main__":
    main()
