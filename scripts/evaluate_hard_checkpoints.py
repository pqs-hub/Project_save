"""Evaluate hard-fault pretraining checkpoints on validation targets."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import math
from pathlib import Path
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tpi_jepa.dataset import TPIDataset, split_by_benchmark  # noqa: E402
from tpi_jepa.features import SCOAP_END, SCOAP_START  # noqa: E402
from tpi_jepa.labels import load_labels  # noqa: E402
from tpi_jepa.model import TPIWorldModel  # noqa: E402
from tpi_jepa.protocol import excluded_benchmarks_from_config, filter_rows_by_excluded_benchmarks  # noqa: E402
from tpi_jepa.train import load_config  # noqa: E402


FIELDS = [
    "checkpoint",
    "epoch",
    "samples",
    "latent_smooth_l1",
    "scoap_mae",
    "delta_scoap_mae",
    "hard_bce",
    "hard_sa0_precision",
    "hard_sa0_recall",
    "hard_sa0_f1",
    "hard_sa0_pr_auc",
    "hard_sa1_precision",
    "hard_sa1_recall",
    "hard_sa1_f1",
    "hard_sa1_pr_auc",
    "hard_macro_f1",
    "hard_macro_f1_at_0p5",
    "hard_sa0_f1_at_0p5",
    "hard_sa1_f1_at_0p5",
    "hard_macro_f1_tuned",
    "hard_threshold_sa0",
    "hard_threshold_sa1",
    "brier_sa0",
    "brier_sa1",
    "ece_sa0",
    "ece_sa1",
    "temperature_sa0",
    "temperature_sa1",
    "temperature_scaled_brier_sa0",
    "temperature_scaled_brier_sa1",
    "temperature_scaled_ece_sa0",
    "temperature_scaled_ece_sa1",
    "delta_ece_sa0_after_temperature",
    "delta_ece_sa1_after_temperature",
    "positive_rate_sa0",
    "positive_rate_sa1",
    "hard_micro_accuracy",
    "hard_recall_at_top_1pct",
    "hard_recall_at_top_5pct",
    "hard_recall_at_top_10pct",
    "hard_count_mae",
    "hard_count_spearman",
    "hard_count_top10_overlap",
    "hard_reduction_mae",
    "hard_reduction_sign_acc",
    "hard_reduction_score",
    "reward_mae",
    "latent_cosine",
    "scoap_acc_at_005",
    "delta_scoap_acc_at_001",
    "hard_count_acc_at_010",
    "hard_reduction_acc_at_005",
    "reward_sign_acc",
    "predictive_score",
]


def checkpoint_epoch(path: Path) -> int:
    match = re.search(r"epoch_(\d+)\.pt$", path.name)
    if match:
        return int(match.group(1))
    if path.name == "best.pt":
        return -2
    if path.name == "latest.pt":
        return -1
    return 0


def load_model(path: Path, device: torch.device) -> tuple[TPIWorldModel, dict]:
    ckpt = torch.load(path, map_location=device)
    config = ckpt["config"]
    model = TPIWorldModel(
        feature_dim=int(ckpt["feature_dim"]),
        latent_dim=int(config["latent_dim"]),
        encoder_layers=int(config["encoder_layers"]),
        action_type_dim=int(config["action_type_dim"]),
        dropout=float(config["dropout"]),
        head_context=bool(config.get("head_context", False)),
        relation_dim=int(ckpt.get("relation_dim", config.get("relation_dim", 4))),
        edge_weight_mode=str(config.get("edge_weight_mode", "mean")),
        edge_keep_ratio=float(config.get("edge_keep_ratio", 1.0)),
        residual_dynamics=bool(config.get("residual_dynamics", False)),
        relation_gate=bool(config.get("relation_gate", False)),
        hard_head_type=str(config.get("hard_head_type", "mlp")),
        encoder_type=str(config.get("encoder_type", "mean")),
        summary_mode=str(config.get("summary_mode", "global")),
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    return model, config


def safe_div(num: float, den: float) -> float:
    return num / den if den > 0.0 else 0.0


def f1_from_counts(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    return precision, recall, f1


def binary_pr_auc(scores: list[float], targets: list[float]) -> float:
    """Compute average precision style PR-AUC without sklearn."""

    if not scores:
        return 0.0
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    y = np.asarray(targets, dtype=np.float64)[order]
    total_pos = float(y.sum())
    if total_pos <= 0.0:
        return 0.0
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall_step = y / total_pos
    return float((precision * recall_step).sum())


def tuned_binary_f1(scores: list[float], targets: list[float]) -> tuple[float, float, float, float]:
    """Return precision, recall, F1, and best threshold from score-derived candidates."""

    best = (0.0, 0.0, 0.0, 0.5)
    s = np.asarray(scores, dtype=np.float64)
    t = np.asarray(targets, dtype=np.float64)
    if s.size == 0 or t.sum() <= 0.0:
        return best
    quantile_thresholds = np.quantile(s, np.linspace(0.01, 0.99, 99))
    positive_thresholds = s[t >= 0.5]
    thresholds = np.unique(np.concatenate([quantile_thresholds, positive_thresholds, np.asarray([0.5])]))
    if thresholds.size > 512:
        thresholds = np.quantile(thresholds, np.linspace(0.0, 1.0, 512))
    for threshold in thresholds:
        p = s >= threshold
        tp = float(((p == 1) & (t == 1)).sum())
        fp = float(((p == 1) & (t == 0)).sum())
        fn = float(((p == 0) & (t == 1)).sum())
        precision, recall, f1 = f1_from_counts(tp, fp, fn)
        if f1 > best[2]:
            best = (precision, recall, f1, threshold)
    return best


def top_recall(scores: torch.Tensor, targets: torch.Tensor, ratio: float) -> float:
    """Recall of positive hard nodes in the top scoring nodes for one graph."""

    hard = targets > 0.5
    positives = int(hard.sum().item())
    if positives <= 0:
        return 0.0
    k = max(1, int(math.ceil(scores.numel() * ratio)))
    top_idx = torch.topk(scores, k=min(k, scores.numel())).indices
    hits = int(hard[top_idx].sum().item())
    return hits / float(positives)


def rankdata(values: torch.Tensor) -> torch.Tensor:
    """Return simple ordinal ranks for Spearman diagnostics."""

    order = torch.argsort(values)
    ranks = torch.empty_like(values, dtype=torch.float32)
    ranks[order] = torch.arange(values.numel(), dtype=torch.float32, device=values.device)
    return ranks


def spearman(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute per-graph Spearman correlation with stable zero-variance handling."""

    if pred.numel() < 2:
        return 0.0
    pr = rankdata(pred.flatten())
    tr = rankdata(target.flatten())
    pr = pr - pr.mean()
    tr = tr - tr.mean()
    denom = pr.norm() * tr.norm()
    if float(denom.item()) <= 0.0:
        return 0.0
    return float((pr * tr).sum().div(denom).clamp(-1.0, 1.0).cpu().item())


def top_overlap(pred: torch.Tensor, target: torch.Tensor, ratio: float = 0.10) -> float:
    """Fraction of target top-k nodes recovered by predicted top-k nodes."""

    k = max(1, int(math.ceil(pred.numel() * ratio)))
    k = min(k, pred.numel())
    pred_top = set(torch.topk(pred, k=k).indices.cpu().tolist())
    target_top = set(torch.topk(target, k=k).indices.cpu().tolist())
    return len(pred_top & target_top) / float(k)


def threshold_metrics(scores: list[float], targets: list[float], threshold: float) -> dict:
    s = np.asarray(scores, dtype=np.float64)
    t = np.asarray(targets, dtype=np.float64)
    if s.size == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
    pred = s >= float(threshold)
    target = t >= 0.5
    tp = float((pred & target).sum())
    fp = float((pred & ~target).sum())
    fn = float((~pred & target).sum())
    tn = float((~pred & ~target).sum())
    precision, recall, f1 = f1_from_counts(tp, fp, fn)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def calibration_stats(scores: list[float], targets: list[float], bins: int = 10) -> tuple[float, float, list[dict]]:
    s = np.asarray(scores, dtype=np.float64)
    t = np.asarray(targets, dtype=np.float64)
    if s.size == 0:
        return 0.0, 0.0, []
    brier = float(np.mean((s - t) ** 2))
    rows = []
    ece = 0.0
    bins = max(1, int(bins))
    edges = np.linspace(0.0, 1.0, bins + 1)
    for idx in range(bins):
        lo = float(edges[idx])
        hi = float(edges[idx + 1])
        if idx == bins - 1:
            mask = (s >= lo) & (s <= hi)
        else:
            mask = (s >= lo) & (s < hi)
        count = int(mask.sum())
        if count <= 0:
            conf = 0.0
            acc = 0.0
        else:
            conf = float(s[mask].mean())
            acc = float(t[mask].mean())
            ece += (count / float(s.size)) * abs(conf - acc)
        rows.append(
            {
                "bin": idx,
                "low": lo,
                "high": hi,
                "count": count,
                "mean_confidence": conf,
                "positive_rate": acc,
                "abs_gap": abs(conf - acc),
            }
        )
    return brier, float(ece), rows


def sigmoid_np(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def temperature_scale_scores(logits: list[float], temperature: float) -> list[float]:
    if not logits:
        return []
    temp = max(float(temperature), 1e-6)
    return sigmoid_np(np.asarray(logits, dtype=np.float64) / temp).tolist()


def fit_temperature_by_brier(logits: list[float], targets: list[float], grid: list[float]) -> tuple[float, float, float, list[float]]:
    """Fit a scalar temperature by minimizing Brier score on the provided grid."""

    if not logits or not targets:
        return 1.0, 0.0, 0.0, []
    candidates = grid or [1.0]
    best_temperature = 1.0
    best_brier = float("inf")
    best_ece = 0.0
    best_scores: list[float] = []
    for temperature in candidates:
        scores = temperature_scale_scores(logits, temperature)
        brier, ece, _ = calibration_stats(scores, targets)
        if brier < best_brier:
            best_temperature = float(temperature)
            best_brier = brier
            best_ece = ece
            best_scores = scores
    return best_temperature, best_brier, best_ece, best_scores


def hard_calibration_temperature_rows(
    checkpoint: str,
    epoch: int,
    hard_logits: list[list[float]],
    hard_targets_all: list[list[float]],
    temperature_grid: list[float],
    calibration_bins: int,
) -> tuple[list[dict], dict]:
    rows = []
    summary: dict[str, float] = {}
    for idx, class_name in [(0, "sa0"), (1, "sa1")]:
        raw_scores = sigmoid_np(np.asarray(hard_logits[idx], dtype=np.float64)).tolist() if hard_logits[idx] else []
        raw_brier, raw_ece, _ = calibration_stats(raw_scores, hard_targets_all[idx], calibration_bins)
        temp, scaled_brier, scaled_ece, scaled_scores = fit_temperature_by_brier(
            hard_logits[idx],
            hard_targets_all[idx],
            temperature_grid,
        )
        raw_f1 = threshold_metrics(raw_scores, hard_targets_all[idx], 0.5)
        scaled_f1 = threshold_metrics(scaled_scores, hard_targets_all[idx], 0.5)
        rows.append(
            {
                "checkpoint": checkpoint,
                "epoch": epoch,
                "class": class_name,
                "temperature": temp,
                "raw_brier": raw_brier,
                "scaled_brier": scaled_brier,
                "raw_ece": raw_ece,
                "scaled_ece": scaled_ece,
                "delta_ece": scaled_ece - raw_ece,
                "raw_f1_at_0p5": raw_f1["f1"],
                "scaled_f1_at_0p5": scaled_f1["f1"],
                "delta_f1_at_0p5": scaled_f1["f1"] - raw_f1["f1"],
            }
        )
        summary[f"temperature_{class_name}"] = temp
        summary[f"temperature_scaled_brier_{class_name}"] = scaled_brier
        summary[f"temperature_scaled_ece_{class_name}"] = scaled_ece
        summary[f"delta_ece_{class_name}_after_temperature"] = scaled_ece - raw_ece
    return rows, summary


def threshold_sweep(scores: list[float], targets: list[float], class_name: str, checkpoint: str, epoch: int) -> list[dict]:
    rows = []
    for threshold in np.linspace(0.0, 1.0, 101):
        metrics = threshold_metrics(scores, targets, float(threshold))
        rows.append(
            {
                "checkpoint": checkpoint,
                "epoch": epoch,
                "class": class_name,
                "threshold": float(threshold),
                **metrics,
            }
        )
    return rows


def write_tsv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def node_count_bucket(num_nodes: int) -> str:
    if num_nodes < 500:
        return "<500"
    if num_nodes < 1000:
        return "500-999"
    if num_nodes < 2000:
        return "1000-1999"
    if num_nodes < 4000:
        return "2000-3999"
    return ">=4000"


def positive_rate_bucket(rate: float) -> str:
    if rate <= 0.0:
        return "0"
    if rate < 0.001:
        return "(0,0.001)"
    if rate < 0.01:
        return "[0.001,0.01)"
    if rate < 0.05:
        return "[0.01,0.05)"
    return ">=0.05"


def ndcg_at_k(scores: list[float], targets: list[float], k: int) -> float:
    if not scores or not targets:
        return 0.0
    s = np.asarray(scores, dtype=np.float64)
    t = np.asarray(targets, dtype=np.float64)
    rel = t - min(0.0, float(t.min()))
    if float(rel.max()) <= 0.0:
        return 0.0
    k = min(max(1, int(k)), s.size)
    order = np.argsort(-s)[:k]
    ideal = np.argsort(-rel)[:k]

    def dcg(indices: np.ndarray) -> float:
        discounts = 1.0 / np.log2(np.arange(2, indices.size + 2))
        return float((rel[indices] * discounts).sum())

    ideal_dcg = dcg(ideal)
    return safe_div(dcg(order), ideal_dcg)


def pairwise_accuracy(scores: list[float], targets: list[float], eps: float = 1e-9) -> tuple[float, int]:
    correct = 0
    total = 0
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            target_delta = targets[i] - targets[j]
            if abs(target_delta) <= eps:
                continue
            score_delta = scores[i] - scores[j]
            total += 1
            if score_delta == 0.0:
                correct += 0.5
            elif (score_delta > 0.0) == (target_delta > 0.0):
                correct += 1
    return safe_div(float(correct), float(total)), total


def spearman_np(scores: list[float], targets: list[float]) -> float:
    if len(scores) < 2:
        return 0.0
    s = torch.tensor(scores, dtype=torch.float32)
    t = torch.tensor(targets, dtype=torch.float32)
    return spearman(s, t)


def action_score_from_record(record: dict, field: str) -> float:
    mapping = {
        "hard_reduction_total": "hard_reduction_pred_total",
        "hard_reduction_sa0": "hard_reduction_pred_sa0",
        "hard_reduction_sa1": "hard_reduction_pred_sa1",
        "reward": "reward_pred",
    }
    return float(record.get(mapping.get(field, field), 0.0))


def action_ranking_diagnostics(
    action_records: list[dict],
    action_score_field: str,
    min_group_size: int,
) -> tuple[list[dict], list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for record in action_records:
        key = (
            record["checkpoint"],
            record["epoch"],
            record["benchmark_id"],
            record["state_key"],
        )
        groups[key].append(record)

    metric_rows = []
    example_rows = []
    min_group_size = max(2, int(min_group_size))
    for key, records in sorted(groups.items()):
        if len(records) < min_group_size:
            continue
        scores = [action_score_from_record(record, action_score_field) for record in records]
        targets = [float(record["hard_reduction_target_total"]) for record in records]
        pair_acc, pair_count = pairwise_accuracy(scores, targets)
        target_spread = max(targets) - min(targets) if targets else 0.0
        best_target = max(targets) if targets else 0.0
        pred_top = int(np.argmax(scores)) if scores else 0
        target_top = int(np.argmax(targets)) if targets else 0
        row = {
            "checkpoint": key[0],
            "epoch": key[1],
            "benchmark_id": key[2],
            "state_key": key[3],
            "group_size": len(records),
            "action_score_field": action_score_field,
            "action_pairwise_acc": pair_acc,
            "pair_count": pair_count,
            "action_spearman": spearman_np(scores, targets),
            "action_ndcg_at_5": ndcg_at_k(scores, targets, 5),
            "action_ndcg_at_10": ndcg_at_k(scores, targets, 10),
            "action_top1_hit": float(pred_top == target_top),
            "target_gain_spread": target_spread,
            "best_target_gain": best_target,
            "pred_top_action": f"{records[pred_top]['action_node_name']}:{records[pred_top]['action_type']}",
            "target_top_action": f"{records[target_top]['action_node_name']}:{records[target_top]['action_type']}",
        }
        metric_rows.append(row)

        ordered = sorted(records, key=lambda record: action_score_from_record(record, action_score_field), reverse=True)
        for rank, record in enumerate(ordered[: min(5, len(ordered))], start=1):
            example_rows.append(
                {
                    "checkpoint": key[0],
                    "epoch": key[1],
                    "benchmark_id": key[2],
                    "state_key": key[3],
                    "rank_by_score": rank,
                    "action_node_name": record["action_node_name"],
                    "action_type": record["action_type"],
                    "action_score": action_score_from_record(record, action_score_field),
                    "hard_reduction_target_total": record["hard_reduction_target_total"],
                    "reward_pred": record["reward_pred"],
                    "delta_fault_coverage": record["delta_fault_coverage"],
                    "group_pairwise_acc": pair_acc,
                    "group_top1_hit": float(pred_top == target_top),
                }
            )
    return metric_rows, example_rows


def parse_float_list(value: str) -> list[float]:
    if not value.strip():
        return []
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def selected_calibration_policies(policy: str, shrinkages: list[float]) -> list[tuple[str, float | None]]:
    policies: list[tuple[str, float | None]] = []
    if policy in ("all", "global_0p5"):
        policies.append(("global_0p5", None))
    if policy in ("all", "class_tuned"):
        policies.append(("class_tuned", None))
    if policy in ("all", "benchmark_tuned"):
        policies.append(("benchmark_tuned", None))
    if policy in ("all", "benchmark_shrinkage"):
        for shrinkage in shrinkages:
            policies.append((f"benchmark_shrinkage_{shrinkage:g}", shrinkage))
    return policies


def two_class_threshold_metrics(values: dict, threshold_sa0: float, threshold_sa1: float) -> dict:
    sa0 = threshold_metrics(values["sa0_scores"], values["sa0_targets"], threshold_sa0)
    sa1 = threshold_metrics(values["sa1_scores"], values["sa1_targets"], threshold_sa1)
    total = sum(sa0[key] + sa1[key] for key in ["tp", "fp", "fn", "tn"])
    nodes = len(values["sa0_scores"])
    return {
        "nodes_evaluated": nodes,
        "hard_macro_f1": (sa0["f1"] + sa1["f1"]) * 0.5,
        "hard_sa0_precision": sa0["precision"],
        "hard_sa0_recall": sa0["recall"],
        "hard_sa0_f1": sa0["f1"],
        "hard_sa1_precision": sa1["precision"],
        "hard_sa1_recall": sa1["recall"],
        "hard_sa1_f1": sa1["f1"],
        "threshold_sa0": threshold_sa0,
        "threshold_sa1": threshold_sa1,
        "fp_rate": safe_div(sa0["fp"] + sa1["fp"], total),
        "fn_rate": safe_div(sa0["fn"] + sa1["fn"], total),
        "sa0_tp": sa0["tp"],
        "sa0_fp": sa0["fp"],
        "sa0_fn": sa0["fn"],
        "sa0_tn": sa0["tn"],
        "sa1_tp": sa1["tp"],
        "sa1_fp": sa1["fp"],
        "sa1_fn": sa1["fn"],
        "sa1_tn": sa1["tn"],
    }


def counts_to_two_class_metrics(counts_row: dict) -> dict:
    sa0_p, sa0_r, sa0_f1 = f1_from_counts(counts_row["sa0_tp"], counts_row["sa0_fp"], counts_row["sa0_fn"])
    sa1_p, sa1_r, sa1_f1 = f1_from_counts(counts_row["sa1_tp"], counts_row["sa1_fp"], counts_row["sa1_fn"])
    total = sum(counts_row[key] for key in ["sa0_tp", "sa0_fp", "sa0_fn", "sa0_tn", "sa1_tp", "sa1_fp", "sa1_fn", "sa1_tn"])
    return {
        "hard_macro_f1": (sa0_f1 + sa1_f1) * 0.5,
        "hard_sa0_precision": sa0_p,
        "hard_sa0_recall": sa0_r,
        "hard_sa0_f1": sa0_f1,
        "hard_sa1_precision": sa1_p,
        "hard_sa1_recall": sa1_r,
        "hard_sa1_f1": sa1_f1,
        "fp_rate": safe_div(counts_row["sa0_fp"] + counts_row["sa1_fp"], total),
        "fn_rate": safe_div(counts_row["sa0_fn"] + counts_row["sa1_fn"], total),
    }


def calibration_policy_diagnostics(
    checkpoint: str,
    epoch: int,
    hard_scores: list[list[float]],
    hard_targets_all: list[list[float]],
    per_benchmark: dict[str, dict],
    tuned_sa0: tuple[float, float, float, float],
    tuned_sa1: tuple[float, float, float, float],
    brier_sa0: float,
    brier_sa1: float,
    ece_sa0: float,
    ece_sa1: float,
    calibration_policy: str,
    shrinkages: list[float],
) -> tuple[list[dict], list[dict], list[dict]]:
    global_thresholds = {
        "global_0p5": (0.5, 0.5),
        "class_tuned": (float(tuned_sa0[3]), float(tuned_sa1[3])),
    }
    selected = selected_calibration_policies(calibration_policy, shrinkages)
    per_bench_thresholds: dict[str, tuple[float, float]] = {}
    for bench, values in per_benchmark.items():
        sa0 = tuned_binary_f1(values["sa0_scores"], values["sa0_targets"])
        sa1 = tuned_binary_f1(values["sa1_scores"], values["sa1_targets"])
        per_bench_thresholds[bench] = (float(sa0[3]), float(sa1[3]))

    per_bench_rows = []
    mode_rows = []
    for policy_name, shrinkage in selected:
        aggregate_counts = {
            "sa0_tp": 0.0,
            "sa0_fp": 0.0,
            "sa0_fn": 0.0,
            "sa0_tn": 0.0,
            "sa1_tp": 0.0,
            "sa1_fp": 0.0,
            "sa1_fn": 0.0,
            "sa1_tn": 0.0,
        }
        bench_f1s = []
        nodes_evaluated = 0
        for bench, values in sorted(per_benchmark.items()):
            if policy_name in global_thresholds:
                threshold_sa0, threshold_sa1 = global_thresholds[policy_name]
            elif policy_name == "benchmark_tuned":
                threshold_sa0, threshold_sa1 = per_bench_thresholds[bench]
            else:
                bench_sa0, bench_sa1 = per_bench_thresholds[bench]
                global_sa0, global_sa1 = global_thresholds["class_tuned"]
                shrink = float(shrinkage or 0.0)
                threshold_sa0 = (1.0 - shrink) * bench_sa0 + shrink * global_sa0
                threshold_sa1 = (1.0 - shrink) * bench_sa1 + shrink * global_sa1

            metrics = two_class_threshold_metrics(values, threshold_sa0, threshold_sa1)
            bench_f1s.append(float(metrics["hard_macro_f1"]))
            nodes_evaluated += int(metrics["nodes_evaluated"])
            for key in aggregate_counts:
                aggregate_counts[key] += float(metrics[key])
            per_bench_rows.append(
                {
                    "checkpoint": checkpoint,
                    "epoch": epoch,
                    "policy": policy_name,
                    "shrinkage": "" if shrinkage is None else shrinkage,
                    "benchmark_id": bench,
                    "num_nodes": values["num_nodes"],
                    "positive_rate_sa0": float(np.mean(values["sa0_targets"])) if values["sa0_targets"] else 0.0,
                    "positive_rate_sa1": float(np.mean(values["sa1_targets"])) if values["sa1_targets"] else 0.0,
                    **metrics,
                }
            )
        aggregate_metrics = counts_to_two_class_metrics(aggregate_counts)
        mode_rows.append(
            {
                "checkpoint": checkpoint,
                "epoch": epoch,
                "policy": policy_name,
                "shrinkage": "" if shrinkage is None else shrinkage,
                "benchmarks": len(per_benchmark),
                "nodes_evaluated": nodes_evaluated,
                "threshold_sa0": "mixed" if policy_name.startswith("benchmark") else global_thresholds[policy_name][0],
                "threshold_sa1": "mixed" if policy_name.startswith("benchmark") else global_thresholds[policy_name][1],
                "mean_benchmark_f1": float(np.mean(bench_f1s)) if bench_f1s else 0.0,
                "worst_benchmark_f1": float(np.min(bench_f1s)) if bench_f1s else 0.0,
                "best_benchmark_f1": float(np.max(bench_f1s)) if bench_f1s else 0.0,
                "brier_sa0": brier_sa0,
                "brier_sa1": brier_sa1,
                "ece_sa0": ece_sa0,
                "ece_sa1": ece_sa1,
                **aggregate_metrics,
            }
        )

    baseline_class = next((row for row in mode_rows if row["policy"] == "class_tuned"), None)
    baseline_global = next((row for row in mode_rows if row["policy"] == "global_0p5"), None)
    if baseline_class is None:
        class_values = {
            "sa0_scores": hard_scores[0],
            "sa0_targets": hard_targets_all[0],
            "sa1_scores": hard_scores[1],
            "sa1_targets": hard_targets_all[1],
        }
        class_metrics = two_class_threshold_metrics(class_values, tuned_sa0[3], tuned_sa1[3])
        baseline_class = {"hard_macro_f1": class_metrics["hard_macro_f1"], "worst_benchmark_f1": 0.0}
    if baseline_global is None:
        global_values = {
            "sa0_scores": hard_scores[0],
            "sa0_targets": hard_targets_all[0],
            "sa1_scores": hard_scores[1],
            "sa1_targets": hard_targets_all[1],
        }
        global_metrics = two_class_threshold_metrics(global_values, 0.5, 0.5)
        baseline_global = {"hard_macro_f1": global_metrics["hard_macro_f1"], "worst_benchmark_f1": 0.0}

    comparison_rows = []
    for row in mode_rows:
        delta_class = float(row["hard_macro_f1"]) - float(baseline_class["hard_macro_f1"])
        delta_global = float(row["hard_macro_f1"]) - float(baseline_global["hard_macro_f1"])
        worst_delta = float(row["worst_benchmark_f1"]) - float(baseline_class.get("worst_benchmark_f1", 0.0))
        if delta_class >= 0.03 or worst_delta >= 0.10:
            decision = "promote"
        elif delta_class <= -0.01:
            decision = "reject"
        else:
            decision = "neutral"
        comparison_rows.append(
            {
                "checkpoint": checkpoint,
                "epoch": epoch,
                "policy": row["policy"],
                "shrinkage": row["shrinkage"],
                "hard_macro_f1": row["hard_macro_f1"],
                "delta_vs_class_tuned": delta_class,
                "delta_vs_global_0p5": delta_global,
                "mean_benchmark_f1": row["mean_benchmark_f1"],
                "worst_benchmark_f1": row["worst_benchmark_f1"],
                "best_benchmark_f1": row["best_benchmark_f1"],
                "ece_sa0": row["ece_sa0"],
                "ece_sa1": row["ece_sa1"],
                "decision": decision,
            }
        )
    return mode_rows, per_bench_rows, comparison_rows


@torch.no_grad()
def evaluate_checkpoint(
    path: Path,
    dataset: TPIDataset,
    device: torch.device,
    max_steps: int | None,
    diagnostics: bool = False,
    calibration_bins: int = 10,
    action_score_field: str = "hard_reduction_total",
    min_action_group_size: int = 2,
    calibration_policy: str = "all",
    benchmark_threshold_shrinkage: list[float] | None = None,
    temperature_scale_hard: bool = False,
    temperature_grid: list[float] | None = None,
) -> dict | tuple[dict, dict]:
    model, config = load_model(path, device)
    coverage_scale = float(config.get("coverage_scale", 100.0))
    totals = {
        "latent_smooth_l1": 0.0,
        "scoap_mae": 0.0,
        "delta_scoap_mae": 0.0,
        "hard_bce": 0.0,
        "hard_count_mae": 0.0,
        "hard_reduction_mae": 0.0,
        "reward_mae": 0.0,
        "latent_cosine": 0.0,
        "scoap_acc_at_005": 0.0,
        "delta_scoap_acc_at_001": 0.0,
        "hard_count_acc_at_010": 0.0,
        "hard_reduction_acc_at_005": 0.0,
        "hard_reduction_sign_acc": 0.0,
        "reward_sign_acc": 0.0,
        "hard_recall_at_top_1pct": 0.0,
        "hard_recall_at_top_5pct": 0.0,
        "hard_recall_at_top_10pct": 0.0,
        "hard_count_spearman": 0.0,
        "hard_count_top10_overlap": 0.0,
    }
    counts = {
        "sa0_tp": 0.0,
        "sa0_fp": 0.0,
        "sa0_fn": 0.0,
        "sa1_tp": 0.0,
        "sa1_fp": 0.0,
        "sa1_fn": 0.0,
        "hard_correct": 0.0,
        "hard_total": 0.0,
    }
    hard_scores = [[], []]
    hard_logits_all = [[], []]
    hard_targets_all = [[], []]
    per_benchmark: dict[str, dict] = defaultdict(lambda: {"sa0_scores": [], "sa0_targets": [], "sa1_scores": [], "sa1_targets": [], "num_nodes": 0})
    per_bucket: dict[tuple, dict] = defaultdict(lambda: {"sa0_scores": [], "sa0_targets": [], "sa1_scores": [], "sa1_targets": [], "samples": 0})
    action_records: list[dict] = []
    steps = 0
    for idx in range(len(dataset)):
        sample = dataset[idx]
        x_pre = sample.x_pre.to(device)
        x_post = sample.x_post.to(device)
        relation = sample.relation_features.to(device)
        out = model(
            sample.graph,
            x_pre,
            x_post,
            sample.action_node_id,
            sample.action_type_id,
            relation,
        )
        scoap_target = x_post[:, SCOAP_START:SCOAP_END]
        delta_scoap_target = x_post[:, SCOAP_START:SCOAP_END] - x_pre[:, SCOAP_START:SCOAP_END]
        hard_target = sample.hard_targets_post.to(device)
        hard_count_target = sample.hard_count_post.to(device)
        hard_reduction_target = sample.hard_reduction_target.to(device)
        reward_target = coverage_scale * sample.delta_fault_coverage.to(device)

        totals["latent_smooth_l1"] += float(F.smooth_l1_loss(out["z_pred"], out["z_t1"]).cpu().item())
        totals["scoap_mae"] += float((out["scoap_pred"] - scoap_target).abs().mean().cpu().item())
        totals["delta_scoap_mae"] += float((out["delta_scoap_pred"] - delta_scoap_target).abs().mean().cpu().item())
        totals["hard_bce"] += float(F.binary_cross_entropy_with_logits(out["hard_logits"], hard_target).cpu().item())
        totals["hard_count_mae"] += float((out["hard_count_pred"].sigmoid() - hard_count_target).abs().mean().cpu().item())
        totals["hard_reduction_mae"] += float((out["hard_reduction_pred"] - hard_reduction_target).abs().mean().cpu().item())
        totals["reward_mae"] += float((out["reward_pred"] - reward_target).abs().cpu().item())
        totals["latent_cosine"] += float(
            ((F.cosine_similarity(out["z_pred"], out["z_t1"], dim=1).mean().clamp(-1.0, 1.0) + 1.0) * 0.5)
            .cpu()
            .item()
        )
        totals["scoap_acc_at_005"] += float(((out["scoap_pred"] - scoap_target).abs() <= 0.05).float().mean().cpu().item())
        totals["delta_scoap_acc_at_001"] += float(
            ((out["delta_scoap_pred"] - delta_scoap_target).abs() <= 0.01).float().mean().cpu().item()
        )
        totals["hard_count_acc_at_010"] += float(
            ((out["hard_count_pred"].sigmoid() - hard_count_target).abs() <= 0.10).float().mean().cpu().item()
        )
        totals["hard_reduction_acc_at_005"] += float(
            ((out["hard_reduction_pred"] - hard_reduction_target).abs() <= 0.05).float().mean().cpu().item()
        )
        totals["hard_reduction_sign_acc"] += float(
            ((out["hard_reduction_pred"] >= 0.0) == (hard_reduction_target >= 0.0)).float().mean().cpu().item()
        )
        totals["reward_sign_acc"] += float(
            ((out["reward_pred"] >= 0.0) == (reward_target >= 0.0)).float().cpu().item()
        )

        prob = out["hard_logits"].sigmoid()
        pred = (prob >= 0.5).float()
        target = hard_target
        prob_cpu = prob.detach().cpu()
        target_cpu = target.detach().cpu()
        hard_count_pred_cpu = out["hard_count_pred"].sigmoid().detach().cpu()
        hard_count_target_cpu = hard_count_target.detach().cpu()
        hard_score = prob.max(dim=1).values
        hard_any = target.max(dim=1).values
        totals["hard_recall_at_top_1pct"] += top_recall(hard_score, hard_any, 0.01)
        totals["hard_recall_at_top_5pct"] += top_recall(hard_score, hard_any, 0.05)
        totals["hard_recall_at_top_10pct"] += top_recall(hard_score, hard_any, 0.10)
        hard_count_pred = out["hard_count_pred"].sigmoid()
        totals["hard_count_spearman"] += spearman(hard_count_pred, hard_count_target)
        totals["hard_count_top10_overlap"] += top_overlap(
            hard_count_pred,
            hard_count_target,
            0.10,
        )
        for col, prefix in [(0, "sa0"), (1, "sa1")]:
            p = pred[:, col]
            t = target[:, col]
            hard_scores[col].extend(prob[:, col].detach().cpu().tolist())
            hard_logits_all[col].extend(out["hard_logits"][:, col].detach().cpu().tolist())
            hard_targets_all[col].extend(t.detach().cpu().tolist())
            counts[f"{prefix}_tp"] += float(((p == 1) & (t == 1)).sum().cpu().item())
            counts[f"{prefix}_fp"] += float(((p == 1) & (t == 0)).sum().cpu().item())
            counts[f"{prefix}_fn"] += float(((p == 0) & (t == 1)).sum().cpu().item())
        counts["hard_correct"] += float((pred == target).sum().cpu().item())
        counts["hard_total"] += float(target.numel())

        if diagnostics:
            bench = str(sample.benchmark_id)
            per_benchmark[bench]["num_nodes"] = sample.graph.num_nodes
            per_benchmark[bench]["sa0_scores"].extend(prob_cpu[:, 0].tolist())
            per_benchmark[bench]["sa0_targets"].extend(target_cpu[:, 0].tolist())
            per_benchmark[bench]["sa1_scores"].extend(prob_cpu[:, 1].tolist())
            per_benchmark[bench]["sa1_targets"].extend(target_cpu[:, 1].tolist())

            hard_pos_rate = float(target_cpu.max(dim=1).values.float().mean().item())
            bucket_key = (
                node_count_bucket(sample.graph.num_nodes),
                positive_rate_bucket(hard_pos_rate),
                str(getattr(sample, "action_type", "")),
            )
            per_bucket[bucket_key]["samples"] += 1
            per_bucket[bucket_key]["sa0_scores"].extend(prob_cpu[:, 0].tolist())
            per_bucket[bucket_key]["sa0_targets"].extend(target_cpu[:, 0].tolist())
            per_bucket[bucket_key]["sa1_scores"].extend(prob_cpu[:, 1].tolist())
            per_bucket[bucket_key]["sa1_targets"].extend(target_cpu[:, 1].tolist())

            action_records.append(
                {
                    "checkpoint": str(path),
                    "epoch": checkpoint_epoch(path),
                    "benchmark_id": bench,
                    "state_key": getattr(sample, "state_key", f"{bench}|{idx}"),
                    "sequence_id": getattr(sample, "sequence_id", ""),
                    "step": getattr(sample, "step", ""),
                    "pre_action_count": getattr(sample, "pre_action_count", ""),
                    "action_node_name": getattr(sample, "action_node_name", ""),
                    "action_type": getattr(sample, "action_type", ""),
                    "hard_reduction_pred_total": float(out["hard_reduction_pred"][0].detach().cpu().item()),
                    "hard_reduction_target_total": float(hard_reduction_target[0].detach().cpu().item()),
                    "hard_reduction_pred_sa0": float(out["hard_reduction_pred"][1].detach().cpu().item()),
                    "hard_reduction_target_sa0": float(hard_reduction_target[1].detach().cpu().item()),
                    "hard_reduction_pred_sa1": float(out["hard_reduction_pred"][2].detach().cpu().item()),
                    "hard_reduction_target_sa1": float(hard_reduction_target[2].detach().cpu().item()),
                    "reward_pred": float(out["reward_pred"].detach().cpu().item()),
                    "delta_fault_coverage": float(sample.delta_fault_coverage.detach().cpu().item()),
                    "num_nodes": sample.graph.num_nodes,
                    "hard_positive_rate": hard_pos_rate,
                    "hard_count_pred_mean": float(hard_count_pred_cpu.mean().item()),
                    "hard_count_target_mean": float(hard_count_target_cpu.mean().item()),
                }
            )

        steps += 1
        if max_steps is not None and steps >= max_steps:
            break

    sa0_p, sa0_r, sa0_f1 = f1_from_counts(counts["sa0_tp"], counts["sa0_fp"], counts["sa0_fn"])
    sa1_p, sa1_r, sa1_f1 = f1_from_counts(counts["sa1_tp"], counts["sa1_fp"], counts["sa1_fn"])
    tuned_sa0 = tuned_binary_f1(hard_scores[0], hard_targets_all[0])
    tuned_sa1 = tuned_binary_f1(hard_scores[1], hard_targets_all[1])
    brier_sa0, ece_sa0, bins_sa0 = calibration_stats(hard_scores[0], hard_targets_all[0], calibration_bins)
    brier_sa1, ece_sa1, bins_sa1 = calibration_stats(hard_scores[1], hard_targets_all[1], calibration_bins)
    temperature_rows: list[dict] = []
    temperature_summary = {
        "temperature_sa0": 1.0,
        "temperature_sa1": 1.0,
        "temperature_scaled_brier_sa0": brier_sa0,
        "temperature_scaled_brier_sa1": brier_sa1,
        "temperature_scaled_ece_sa0": ece_sa0,
        "temperature_scaled_ece_sa1": ece_sa1,
        "delta_ece_sa0_after_temperature": 0.0,
        "delta_ece_sa1_after_temperature": 0.0,
    }
    if temperature_scale_hard:
        temperature_rows, temperature_summary = hard_calibration_temperature_rows(
            checkpoint=str(path),
            epoch=checkpoint_epoch(path),
            hard_logits=hard_logits_all,
            hard_targets_all=hard_targets_all,
            temperature_grid=temperature_grid or [1.0],
            calibration_bins=calibration_bins,
        )
    hard_macro_f1_tuned = (tuned_sa0[2] + tuned_sa1[2]) * 0.5
    averaged = {key: value / max(1, steps) for key, value in totals.items()}
    hard_reduction_score = max(0.0, 1.0 - float(averaged["hard_reduction_mae"]))
    predictive_score = (
        0.45 * hard_macro_f1_tuned
        + 0.25 * float(averaged.get("hard_recall_at_top_10pct", 0.0))
        + 0.15 * hard_reduction_score
        + 0.10 * float(averaged.get("hard_count_top10_overlap", 0.0))
        + 0.05 * float(averaged["scoap_acc_at_005"])
    )
    row = {
        "checkpoint": str(path),
        "epoch": checkpoint_epoch(path),
        "samples": steps,
        **averaged,
        "hard_sa0_precision": sa0_p,
        "hard_sa0_recall": sa0_r,
        "hard_sa0_f1": sa0_f1,
        "hard_sa0_pr_auc": binary_pr_auc(hard_scores[0], hard_targets_all[0]),
        "hard_sa1_precision": sa1_p,
        "hard_sa1_recall": sa1_r,
        "hard_sa1_f1": sa1_f1,
        "hard_sa1_pr_auc": binary_pr_auc(hard_scores[1], hard_targets_all[1]),
        "hard_macro_f1": (sa0_f1 + sa1_f1) * 0.5,
        "hard_macro_f1_at_0p5": (sa0_f1 + sa1_f1) * 0.5,
        "hard_sa0_f1_at_0p5": sa0_f1,
        "hard_sa1_f1_at_0p5": sa1_f1,
        "hard_macro_f1_tuned": hard_macro_f1_tuned,
        "hard_threshold_sa0": tuned_sa0[3],
        "hard_threshold_sa1": tuned_sa1[3],
        "brier_sa0": brier_sa0,
        "brier_sa1": brier_sa1,
        "ece_sa0": ece_sa0,
        "ece_sa1": ece_sa1,
        **temperature_summary,
        "positive_rate_sa0": float(np.mean(hard_targets_all[0])) if hard_targets_all[0] else 0.0,
        "positive_rate_sa1": float(np.mean(hard_targets_all[1])) if hard_targets_all[1] else 0.0,
        "hard_micro_accuracy": safe_div(counts["hard_correct"], counts["hard_total"]),
        "hard_reduction_score": hard_reduction_score,
        "predictive_score": predictive_score,
    }
    if not diagnostics:
        return row

    checkpoint = str(path)
    epoch = checkpoint_epoch(path)
    threshold_rows = []
    threshold_rows.extend(threshold_sweep(hard_scores[0], hard_targets_all[0], "sa0", checkpoint, epoch))
    threshold_rows.extend(threshold_sweep(hard_scores[1], hard_targets_all[1], "sa1", checkpoint, epoch))
    thresholds_by_class = []
    for class_name, tuned, brier, ece, scores, targets in [
        ("sa0", tuned_sa0, brier_sa0, ece_sa0, hard_scores[0], hard_targets_all[0]),
        ("sa1", tuned_sa1, brier_sa1, ece_sa1, hard_scores[1], hard_targets_all[1]),
    ]:
        metrics = threshold_metrics(scores, targets, tuned[3])
        total = metrics["tp"] + metrics["fp"] + metrics["fn"] + metrics["tn"]
        thresholds_by_class.append(
            {
                "checkpoint": checkpoint,
                "epoch": epoch,
                "class": class_name,
                "threshold": tuned[3],
                "precision": tuned[0],
                "recall": tuned[1],
                "f1": tuned[2],
                "brier": brier,
                "ece": ece,
                "positive_rate": float(np.mean(targets)) if targets else 0.0,
                "fp_rate": safe_div(metrics["fp"], total),
                "fn_rate": safe_div(metrics["fn"], total),
                **metrics,
            }
        )
    calibration_bin_rows = []
    for class_name, brier, ece, bin_rows in [("sa0", brier_sa0, ece_sa0, bins_sa0), ("sa1", brier_sa1, ece_sa1, bins_sa1)]:
        for bin_row in bin_rows:
            calibration_bin_rows.append({"checkpoint": checkpoint, "epoch": epoch, "class": class_name, "brier": brier, "ece": ece, **bin_row})

    per_benchmark_rows = []
    for bench, values in sorted(per_benchmark.items()):
        sa0 = tuned_binary_f1(values["sa0_scores"], values["sa0_targets"])
        sa1 = tuned_binary_f1(values["sa1_scores"], values["sa1_targets"])
        b0, e0, _ = calibration_stats(values["sa0_scores"], values["sa0_targets"], calibration_bins)
        b1, e1, _ = calibration_stats(values["sa1_scores"], values["sa1_targets"], calibration_bins)
        per_benchmark_rows.append(
            {
                "checkpoint": checkpoint,
                "epoch": epoch,
                "benchmark_id": bench,
                "num_nodes": values["num_nodes"],
                "nodes_evaluated": len(values["sa0_scores"]),
                "hard_macro_f1_tuned": (sa0[2] + sa1[2]) * 0.5,
                "hard_sa0_f1_tuned": sa0[2],
                "hard_sa1_f1_tuned": sa1[2],
                "hard_threshold_sa0": sa0[3],
                "hard_threshold_sa1": sa1[3],
                "positive_rate_sa0": float(np.mean(values["sa0_targets"])) if values["sa0_targets"] else 0.0,
                "positive_rate_sa1": float(np.mean(values["sa1_targets"])) if values["sa1_targets"] else 0.0,
                "brier_sa0": b0,
                "brier_sa1": b1,
                "ece_sa0": e0,
                "ece_sa1": e1,
            }
        )

    bucket_rows = []
    for (node_bucket, pos_bucket, action_type), values in sorted(per_bucket.items()):
        sa0 = tuned_binary_f1(values["sa0_scores"], values["sa0_targets"])
        sa1 = tuned_binary_f1(values["sa1_scores"], values["sa1_targets"])
        bucket_rows.append(
            {
                "checkpoint": checkpoint,
                "epoch": epoch,
                "node_count_bucket": node_bucket,
                "hard_positive_rate_bucket": pos_bucket,
                "action_type": action_type,
                "samples": values["samples"],
                "nodes_evaluated": len(values["sa0_scores"]),
                "hard_macro_f1_tuned": (sa0[2] + sa1[2]) * 0.5,
                "hard_sa0_f1_tuned": sa0[2],
                "hard_sa1_f1_tuned": sa1[2],
                "positive_rate_sa0": float(np.mean(values["sa0_targets"])) if values["sa0_targets"] else 0.0,
                "positive_rate_sa1": float(np.mean(values["sa1_targets"])) if values["sa1_targets"] else 0.0,
            }
        )

    action_metric_rows, action_example_rows = action_ranking_diagnostics(action_records, action_score_field, min_action_group_size)
    calibration_mode_rows, per_bench_calibrated_rows, threshold_policy_rows = calibration_policy_diagnostics(
        checkpoint=checkpoint,
        epoch=epoch,
        hard_scores=hard_scores,
        hard_targets_all=hard_targets_all,
        per_benchmark=per_benchmark,
        tuned_sa0=tuned_sa0,
        tuned_sa1=tuned_sa1,
        brier_sa0=brier_sa0,
        brier_sa1=brier_sa1,
        ece_sa0=ece_sa0,
        ece_sa1=ece_sa1,
        calibration_policy=calibration_policy,
        shrinkages=benchmark_threshold_shrinkage or [0.25, 0.5, 0.75],
    )
    diag = {
        "thresholds_by_class": thresholds_by_class,
        "threshold_sweep": threshold_rows,
        "calibration_bins": calibration_bin_rows,
        "per_benchmark_metrics": per_benchmark_rows,
        "bucket_metrics": bucket_rows,
        "action_records": action_records,
        "action_ranking_metrics": action_metric_rows,
        "action_group_examples": action_example_rows,
        "calibration_mode_metrics": calibration_mode_rows,
        "per_benchmark_calibrated_metrics": per_bench_calibrated_rows,
        "threshold_policy_comparison": threshold_policy_rows,
        "hard_calibration_temperature": temperature_rows,
    }
    return row, diag


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def plot_metrics(path: Path, rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epoch_rows = [row for row in rows if int(row["epoch"]) > 0]
    epoch_rows.sort(key=lambda row: int(row["epoch"]))
    xs = [int(row["epoch"]) for row in epoch_rows]
    metrics = [
        ("latent_cosine", "latent cosine"),
        ("scoap_acc_at_005", "SCOAP acc@0.05"),
        ("delta_scoap_acc_at_001", "delta-SCOAP acc@0.01"),
        ("hard_macro_f1_tuned", "hard tuned macro F1"),
        ("hard_recall_at_top_10pct", "hard recall@top10%"),
        ("hard_count_top10_overlap", "hard-count top10 overlap"),
        ("hard_reduction_score", "hard-reduction score"),
        ("predictive_score", "predictive score"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    for ax, (field, title) in zip(axes.flat, metrics):
        ys = [float(row[field]) for row in epoch_rows]
        ax.plot(xs, ys, marker="o")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
    axes.flat[-1].axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)


DIAGNOSTIC_FIELDS = {
    "thresholds_by_class.tsv": [
        "checkpoint",
        "epoch",
        "class",
        "threshold",
        "precision",
        "recall",
        "f1",
        "brier",
        "ece",
        "positive_rate",
        "fp_rate",
        "fn_rate",
        "tp",
        "fp",
        "fn",
        "tn",
    ],
    "threshold_sweep.tsv": ["checkpoint", "epoch", "class", "threshold", "precision", "recall", "f1", "tp", "fp", "fn", "tn"],
    "calibration_bins.tsv": [
        "checkpoint",
        "epoch",
        "class",
        "bin",
        "low",
        "high",
        "count",
        "mean_confidence",
        "positive_rate",
        "abs_gap",
        "brier",
        "ece",
    ],
    "per_benchmark_metrics.tsv": [
        "checkpoint",
        "epoch",
        "benchmark_id",
        "num_nodes",
        "nodes_evaluated",
        "hard_macro_f1_tuned",
        "hard_sa0_f1_tuned",
        "hard_sa1_f1_tuned",
        "hard_threshold_sa0",
        "hard_threshold_sa1",
        "positive_rate_sa0",
        "positive_rate_sa1",
        "brier_sa0",
        "brier_sa1",
        "ece_sa0",
        "ece_sa1",
    ],
    "bucket_metrics.tsv": [
        "checkpoint",
        "epoch",
        "node_count_bucket",
        "hard_positive_rate_bucket",
        "action_type",
        "samples",
        "nodes_evaluated",
        "hard_macro_f1_tuned",
        "hard_sa0_f1_tuned",
        "hard_sa1_f1_tuned",
        "positive_rate_sa0",
        "positive_rate_sa1",
    ],
    "action_ranking_metrics.tsv": [
        "checkpoint",
        "epoch",
        "benchmark_id",
        "state_key",
        "group_size",
        "action_score_field",
        "action_pairwise_acc",
        "pair_count",
        "action_spearman",
        "action_ndcg_at_5",
        "action_ndcg_at_10",
        "action_top1_hit",
        "target_gain_spread",
        "best_target_gain",
        "pred_top_action",
        "target_top_action",
    ],
    "action_group_examples.tsv": [
        "checkpoint",
        "epoch",
        "benchmark_id",
        "state_key",
        "rank_by_score",
        "action_node_name",
        "action_type",
        "action_score",
        "hard_reduction_target_total",
        "reward_pred",
        "delta_fault_coverage",
        "group_pairwise_acc",
        "group_top1_hit",
    ],
    "calibration_mode_metrics.tsv": [
        "checkpoint",
        "epoch",
        "policy",
        "shrinkage",
        "benchmarks",
        "nodes_evaluated",
        "threshold_sa0",
        "threshold_sa1",
        "hard_macro_f1",
        "hard_sa0_precision",
        "hard_sa0_recall",
        "hard_sa0_f1",
        "hard_sa1_precision",
        "hard_sa1_recall",
        "hard_sa1_f1",
        "mean_benchmark_f1",
        "worst_benchmark_f1",
        "best_benchmark_f1",
        "brier_sa0",
        "brier_sa1",
        "ece_sa0",
        "ece_sa1",
        "fp_rate",
        "fn_rate",
    ],
    "per_benchmark_calibrated_metrics.tsv": [
        "checkpoint",
        "epoch",
        "policy",
        "shrinkage",
        "benchmark_id",
        "num_nodes",
        "nodes_evaluated",
        "positive_rate_sa0",
        "positive_rate_sa1",
        "threshold_sa0",
        "threshold_sa1",
        "hard_macro_f1",
        "hard_sa0_precision",
        "hard_sa0_recall",
        "hard_sa0_f1",
        "hard_sa1_precision",
        "hard_sa1_recall",
        "hard_sa1_f1",
        "fp_rate",
        "fn_rate",
    ],
    "threshold_policy_comparison.tsv": [
        "checkpoint",
        "epoch",
        "policy",
        "shrinkage",
        "hard_macro_f1",
        "delta_vs_class_tuned",
        "delta_vs_global_0p5",
        "mean_benchmark_f1",
        "worst_benchmark_f1",
        "best_benchmark_f1",
        "ece_sa0",
        "ece_sa1",
        "decision",
    ],
    "hard_calibration_temperature.tsv": [
        "checkpoint",
        "epoch",
        "class",
        "temperature",
        "raw_brier",
        "scaled_brier",
        "raw_ece",
        "scaled_ece",
        "delta_ece",
        "raw_f1_at_0p5",
        "scaled_f1_at_0p5",
        "delta_f1_at_0p5",
    ],
}


def mean_field(rows: list[dict], field: str) -> float:
    vals = [float(row[field]) for row in rows if row.get(field) not in (None, "")]
    return sum(vals) / len(vals) if vals else 0.0


def write_diagnostics_report(diagnostics_dir: Path, rows_out: list[dict], diag_rows: dict[str, list[dict]]) -> None:
    best = max(rows_out, key=lambda row: float(row.get("hard_macro_f1_tuned", 0.0))) if rows_out else {}
    thresholds = diag_rows.get("thresholds_by_class.tsv", [])
    bench_rows = diag_rows.get("per_benchmark_metrics.tsv", [])
    bucket_rows = diag_rows.get("bucket_metrics.tsv", [])
    action_rows = diag_rows.get("action_ranking_metrics.tsv", [])

    worst_bench = sorted(bench_rows, key=lambda row: float(row.get("hard_macro_f1_tuned", 0.0)))[:5]
    worst_bucket = sorted(bucket_rows, key=lambda row: float(row.get("hard_macro_f1_tuned", 0.0)))[:5]
    action_pairwise = mean_field(action_rows, "action_pairwise_acc")
    action_ndcg10 = mean_field(action_rows, "action_ndcg_at_10")
    action_top1 = mean_field(action_rows, "action_top1_hit")
    mean_ece_sa0 = mean_field([row for row in thresholds if row.get("class") == "sa0"], "ece")
    mean_ece_sa1 = mean_field([row for row in thresholds if row.get("class") == "sa1"], "ece")

    recommendation = []
    if action_rows and (action_pairwise >= 0.60 or action_ndcg10 >= 0.65 or action_top1 >= 0.50):
        recommendation.append("Action-ranking signal is present enough to justify implementing an action-level ranking loss.")
    elif action_rows:
        recommendation.append("Action-ranking signal is weak or inconclusive; inspect action groups before adding ranking loss.")
    else:
        recommendation.append("No comparable action groups were found; action-ranking diagnostics need richer candidate grouping/data.")
    if max(mean_ece_sa0, mean_ece_sa1) >= 0.10:
        recommendation.append("Calibration error is high; prioritize threshold/calibration work before claiming new hard-F1 gains.")
    else:
        recommendation.append("Calibration error is not the dominant issue by ECE; inspect benchmark buckets and labels.")

    lines = [
        "# Calibration / Action-Ranking Diagnostics",
        "",
        "## Evaluated Checkpoints",
        "",
        f"- checkpoint_count: `{len(rows_out)}`",
        f"- best_epoch: `{best.get('epoch', '')}`",
        f"- best_hard_macro_f1_tuned: `{best.get('hard_macro_f1_tuned', '')}`",
        f"- best_predictive_score: `{best.get('predictive_score', '')}`",
        "",
        "## Calibration Summary",
        "",
        f"- mean_ece_sa0: `{mean_ece_sa0:.6f}`",
        f"- mean_ece_sa1: `{mean_ece_sa1:.6f}`",
        f"- thresholds_by_class: `thresholds_by_class.tsv`",
        f"- threshold_sweep: `threshold_sweep.tsv`",
        f"- calibration_bins: `calibration_bins.tsv`",
        "",
        "## Worst Benchmarks",
        "",
        "| benchmark | epoch | hard F1 | SA0 F1 | SA1 F1 | nodes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in worst_bench:
        lines.append(
            f"| `{row.get('benchmark_id', '')}` | `{row.get('epoch', '')}` | "
            f"`{float(row.get('hard_macro_f1_tuned', 0.0)):.4f}` | "
            f"`{float(row.get('hard_sa0_f1_tuned', 0.0)):.4f}` | "
            f"`{float(row.get('hard_sa1_f1_tuned', 0.0)):.4f}` | `{row.get('num_nodes', '')}` |"
        )
    lines.extend(
        [
            "",
            "## Worst Buckets",
            "",
            "| node bucket | positive-rate bucket | action type | samples | hard F1 |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in worst_bucket:
        lines.append(
            f"| `{row.get('node_count_bucket', '')}` | `{row.get('hard_positive_rate_bucket', '')}` | "
            f"`{row.get('action_type', '')}` | `{row.get('samples', '')}` | "
            f"`{float(row.get('hard_macro_f1_tuned', 0.0)):.4f}` |"
        )
    lines.extend(
        [
            "",
            "## Action-Ranking Summary",
            "",
            f"- comparable_action_groups: `{len(action_rows)}`",
            f"- mean_action_pairwise_acc: `{action_pairwise:.6f}`",
            f"- mean_action_ndcg_at_10: `{action_ndcg10:.6f}`",
            f"- mean_action_top1_hit: `{action_top1:.6f}`",
            f"- metrics_file: `action_ranking_metrics.tsv`",
            f"- examples_file: `action_group_examples.tsv`",
            "",
            "## Recommendation",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in recommendation)
    (diagnostics_dir / "calibration_action_diagnostics.md").write_text("\n".join(lines) + "\n")


def write_calibration_mode_report(diagnostics_dir: Path, diag_rows: dict[str, list[dict]]) -> None:
    mode_rows = diag_rows.get("calibration_mode_metrics.tsv", [])
    comparison_rows = diag_rows.get("threshold_policy_comparison.tsv", [])
    per_bench_rows = diag_rows.get("per_benchmark_calibrated_metrics.tsv", [])
    if not mode_rows:
        return

    best = max(mode_rows, key=lambda row: float(row.get("hard_macro_f1", 0.0)))
    class_rows = [row for row in mode_rows if row.get("policy") == "class_tuned"]
    class_best = max(class_rows, key=lambda row: float(row.get("hard_macro_f1", 0.0))) if class_rows else {}
    global_rows = [row for row in mode_rows if row.get("policy") == "global_0p5"]
    global_best = max(global_rows, key=lambda row: float(row.get("hard_macro_f1", 0.0))) if global_rows else {}
    promoted = [row for row in comparison_rows if row.get("decision") == "promote"]
    worst = sorted(per_bench_rows, key=lambda row: float(row.get("hard_macro_f1", 0.0)))[:8]

    best_delta_class = float(best.get("hard_macro_f1", 0.0)) - float(class_best.get("hard_macro_f1", 0.0) or 0.0)
    if promoted:
        recommendation = "Promote the best promoted threshold policy for the next full validation run."
    elif best_delta_class >= 0.01:
        recommendation = "Keep this as a diagnostic-only candidate; improvement is below the current promote threshold."
    else:
        recommendation = "Do not change training/evaluation policy yet; calibration policy did not produce a reliable F1 gain."

    lines = [
        "# Calibration Policy Comparison",
        "",
        "## Scope",
        "",
        "- No model training was run.",
        "- This report compares post-hoc hard-node threshold policies on the same evaluated validation samples.",
        "- ECE/Brier are probability calibration diagnostics; threshold policies change F1/FP/FN tradeoffs, not raw probability calibration.",
        "",
        "## Headline",
        "",
        f"- best_policy: `{best.get('policy', '')}`",
        f"- best_shrinkage: `{best.get('shrinkage', '')}`",
        f"- best_hard_macro_f1: `{float(best.get('hard_macro_f1', 0.0)):.6f}`",
        f"- class_tuned_hard_macro_f1: `{float(class_best.get('hard_macro_f1', 0.0) or 0.0):.6f}`",
        f"- global_0p5_hard_macro_f1: `{float(global_best.get('hard_macro_f1', 0.0) or 0.0):.6f}`",
        f"- delta_vs_class_tuned: `{best_delta_class:.6f}`",
        f"- best_worst_benchmark_f1: `{float(best.get('worst_benchmark_f1', 0.0)):.6f}`",
        "",
        "## Policy Table",
        "",
        "| policy | shrinkage | hard F1 | delta vs class | worst benchmark F1 | decision |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in sorted(comparison_rows, key=lambda item: float(item.get("hard_macro_f1", 0.0)), reverse=True):
        lines.append(
            f"| `{row.get('policy', '')}` | `{row.get('shrinkage', '')}` | "
            f"`{float(row.get('hard_macro_f1', 0.0)):.4f}` | "
            f"`{float(row.get('delta_vs_class_tuned', 0.0)):.4f}` | "
            f"`{float(row.get('worst_benchmark_f1', 0.0)):.4f}` | `{row.get('decision', '')}` |"
        )
    lines.extend(
        [
            "",
            "## Worst Per-Benchmark Cases",
            "",
            "| policy | benchmark | hard F1 | SA0 F1 | SA1 F1 | thresholds |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in worst:
        lines.append(
            f"| `{row.get('policy', '')}` | `{row.get('benchmark_id', '')}` | "
            f"`{float(row.get('hard_macro_f1', 0.0)):.4f}` | "
            f"`{float(row.get('hard_sa0_f1', 0.0)):.4f}` | "
            f"`{float(row.get('hard_sa1_f1', 0.0)):.4f}` | "
            f"`{float(row.get('threshold_sa0', 0.0)):.3f}/{float(row.get('threshold_sa1', 0.0)):.3f}` |"
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"- {recommendation}",
            "- Promotion rule used here: hard F1 improves by at least `0.03`, or worst benchmark F1 improves by at least `0.10` versus class-tuned.",
            "- Output files: `calibration_mode_metrics.tsv`, `per_benchmark_calibrated_metrics.tsv`, `threshold_policy_comparison.tsv`.",
        ]
    )
    (diagnostics_dir / "calibration_mode_report.md").write_text("\n".join(lines) + "\n")


def write_hard_calibration_report(diagnostics_dir: Path, rows_out: list[dict], diag_rows: dict[str, list[dict]]) -> None:
    temp_rows = diag_rows.get("hard_calibration_temperature.tsv", [])
    if not temp_rows:
        return
    best = max(rows_out, key=lambda row: float(row.get("hard_macro_f1_tuned", 0.0))) if rows_out else {}
    sa0_rows = [row for row in temp_rows if row.get("class") == "sa0"]
    sa1_rows = [row for row in temp_rows if row.get("class") == "sa1"]
    mean_raw_ece_sa0 = mean_field(sa0_rows, "raw_ece")
    mean_scaled_ece_sa0 = mean_field(sa0_rows, "scaled_ece")
    mean_raw_ece_sa1 = mean_field(sa1_rows, "raw_ece")
    mean_scaled_ece_sa1 = mean_field(sa1_rows, "scaled_ece")
    mean_raw_f1 = mean_field(temp_rows, "raw_f1_at_0p5")
    mean_scaled_f1 = mean_field(temp_rows, "scaled_f1_at_0p5")
    ece_drop_sa0 = safe_div(mean_raw_ece_sa0 - mean_scaled_ece_sa0, mean_raw_ece_sa0)
    ece_drop_sa1 = safe_div(mean_raw_ece_sa1 - mean_scaled_ece_sa1, mean_raw_ece_sa1)
    if max(ece_drop_sa0, ece_drop_sa1) >= 0.20:
        recommendation = "Temperature scaling shows meaningful calibration headroom; train-time calibration loss is justified."
    else:
        recommendation = "Temperature scaling does not show a large ECE reduction in this smoke run; treat train-time calibration as experimental."

    lines = [
        "# Hard Calibration Temperature Report",
        "",
        "## Scope",
        "",
        "- No model weights were changed.",
        "- Temperatures are fit independently for SA0 and SA1 on validation logits.",
        "- This report measures calibration headroom before training-time Brier/soft-F1 changes.",
        "",
        "## Headline",
        "",
        f"- checkpoint_count: `{len(rows_out)}`",
        f"- best_epoch: `{best.get('epoch', '')}`",
        f"- best_hard_macro_f1_tuned: `{best.get('hard_macro_f1_tuned', '')}`",
        f"- mean_raw_ece_sa0: `{mean_raw_ece_sa0:.6f}`",
        f"- mean_scaled_ece_sa0: `{mean_scaled_ece_sa0:.6f}`",
        f"- mean_ece_drop_sa0: `{ece_drop_sa0:.6f}`",
        f"- mean_raw_ece_sa1: `{mean_raw_ece_sa1:.6f}`",
        f"- mean_scaled_ece_sa1: `{mean_scaled_ece_sa1:.6f}`",
        f"- mean_ece_drop_sa1: `{ece_drop_sa1:.6f}`",
        f"- mean_raw_f1_at_0p5: `{mean_raw_f1:.6f}`",
        f"- mean_scaled_f1_at_0p5: `{mean_scaled_f1:.6f}`",
        "",
        "## Per-Checkpoint Temperature",
        "",
        "| epoch | class | temperature | raw ECE | scaled ECE | raw F1@0.5 | scaled F1@0.5 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in temp_rows:
        lines.append(
            f"| `{row.get('epoch', '')}` | `{row.get('class', '')}` | "
            f"`{float(row.get('temperature', 0.0)):.3f}` | "
            f"`{float(row.get('raw_ece', 0.0)):.4f}` | "
            f"`{float(row.get('scaled_ece', 0.0)):.4f}` | "
            f"`{float(row.get('raw_f1_at_0p5', 0.0)):.4f}` | "
            f"`{float(row.get('scaled_f1_at_0p5', 0.0)):.4f}` |"
        )
    lines.extend(["", "## Recommendation", "", f"- {recommendation}"])
    (diagnostics_dir / "hard_calibration_report.md").write_text("\n".join(lines) + "\n")


def write_diagnostic_outputs(diagnostics_dir: Path, rows_out: list[dict], diagnostics: list[dict]) -> None:
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    combined: dict[str, list[dict]] = {name: [] for name in DIAGNOSTIC_FIELDS}
    key_map = {
        "thresholds_by_class.tsv": "thresholds_by_class",
        "threshold_sweep.tsv": "threshold_sweep",
        "calibration_bins.tsv": "calibration_bins",
        "per_benchmark_metrics.tsv": "per_benchmark_metrics",
        "bucket_metrics.tsv": "bucket_metrics",
        "action_ranking_metrics.tsv": "action_ranking_metrics",
        "action_group_examples.tsv": "action_group_examples",
        "calibration_mode_metrics.tsv": "calibration_mode_metrics",
        "per_benchmark_calibrated_metrics.tsv": "per_benchmark_calibrated_metrics",
        "threshold_policy_comparison.tsv": "threshold_policy_comparison",
        "hard_calibration_temperature.tsv": "hard_calibration_temperature",
    }
    for diag in diagnostics:
        for filename, key in key_map.items():
            combined[filename].extend(diag.get(key, []))
    for filename, rows in combined.items():
        write_tsv(diagnostics_dir / filename, rows, DIAGNOSTIC_FIELDS[filename])
    write_diagnostics_report(diagnostics_dir, rows_out, combined)
    write_calibration_mode_report(diagnostics_dir, combined)
    write_hard_calibration_report(diagnostics_dir, rows_out, combined)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/aig_lowtc_100k_hard_pretrain.json")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--out-png", default=None)
    parser.add_argument("--max-val-samples", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--diagnostics-dir", default=None)
    parser.add_argument("--write-calibration-diagnostics", action="store_true")
    parser.add_argument("--write-action-ranking-diagnostics", action="store_true")
    parser.add_argument("--write-calibration-policy-report", action="store_true")
    parser.add_argument("--write-hard-calibration-report", action="store_true")
    parser.add_argument("--temperature-scale-hard", action="store_true")
    parser.add_argument("--temperature-grid", default="0.5,0.75,1.0,1.25,1.5,2.0,3.0,4.0")
    parser.add_argument(
        "--calibration-policy",
        default="all",
        choices=["all", "global_0p5", "class_tuned", "benchmark_tuned", "benchmark_shrinkage"],
    )
    parser.add_argument("--benchmark-threshold-shrinkage", default="0.25,0.5,0.75")
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument(
        "--action-score-field",
        default="hard_reduction_total",
        choices=["hard_reduction_total", "hard_reduction_sa0", "hard_reduction_sa1", "reward"],
    )
    parser.add_argument("--min-action-group-size", type=int, default=2)
    args = parser.parse_args()
    shrinkages = parse_float_list(args.benchmark_threshold_shrinkage)
    temperature_grid = parse_float_list(args.temperature_grid)

    config = load_config(args.config)
    run_dir = Path(args.run_dir or config["run_dir"])
    all_rows = load_labels(config["labels"])
    rows = filter_rows_by_excluded_benchmarks(all_rows, excluded_benchmarks_from_config(config))
    _, val_rows, _ = split_by_benchmark(
        rows,
        int(config["seed"]),
        train_frac=float(config.get("train_frac", 0.70)),
        val_frac=float(config.get("val_frac", 0.15)),
    )
    dataset = TPIDataset(
        val_rows,
        max_specs=args.max_val_samples,
        max_nodes=int(config.get("max_nodes", 0)) or None,
        feature_mode=str(config.get("feature_mode", "basic")),
        relation_mode=str(config.get("relation_mode", "basic")),
        relation_depth=int(config.get("relation_depth", 8)),
        real_fault_prior_path=config.get("real_fault_priors") or config.get("real_fault_prior_path"),
        activation_prior_path=config.get("activation_priors") or config.get("activation_prior_path"),
        cache_samples=bool(config.get("cache_eval_samples", config.get("cache_samples", False))),
        sample_cache_max_entries=int(config.get("sample_cache_max_entries", 0)) or None,
    )
    checkpoints = sorted(run_dir.glob("epoch_*.pt"), key=checkpoint_epoch)
    for name in ["best.pt", "latest.pt"]:
        path = run_dir / name
        if path.exists():
            checkpoints.append(path)
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found in {run_dir}")

    device = torch.device(args.device)
    diagnostics_enabled = bool(
        args.write_calibration_diagnostics
        or args.write_action_ranking_diagnostics
        or args.write_calibration_policy_report
        or args.write_hard_calibration_report
        or args.diagnostics_dir
    )
    rows_out = []
    diagnostics_out = []
    for path in checkpoints:
        result = evaluate_checkpoint(
            path,
            dataset,
            device,
            args.max_steps,
            diagnostics=diagnostics_enabled,
            calibration_bins=args.calibration_bins,
            action_score_field=args.action_score_field,
            min_action_group_size=args.min_action_group_size,
            calibration_policy=args.calibration_policy,
            benchmark_threshold_shrinkage=shrinkages,
            temperature_scale_hard=bool(args.temperature_scale_hard or args.write_hard_calibration_report),
            temperature_grid=temperature_grid,
        )
        if diagnostics_enabled:
            row, diag = result  # type: ignore[misc]
            if not args.write_calibration_diagnostics:
                for key in ["thresholds_by_class", "threshold_sweep", "calibration_bins", "per_benchmark_metrics", "bucket_metrics"]:
                    diag[key] = []
            if not args.write_action_ranking_diagnostics:
                for key in ["action_ranking_metrics", "action_group_examples"]:
                    diag[key] = []
            if not args.write_calibration_policy_report:
                for key in ["calibration_mode_metrics", "per_benchmark_calibrated_metrics", "threshold_policy_comparison"]:
                    diag[key] = []
            if not args.write_hard_calibration_report:
                diag["hard_calibration_temperature"] = []
            rows_out.append(row)
            diagnostics_out.append(diag)
        else:
            rows_out.append(result)  # type: ignore[arg-type]
    out_csv = Path(args.out_csv or run_dir / "target_metrics.csv")
    out_png = Path(args.out_png or run_dir / "target_metrics.png")
    write_csv(out_csv, rows_out)
    try:
        plot_metrics(out_png, rows_out)
    except Exception as exc:
        print(f"[eval] warning: failed to plot metrics: {exc}")
    if diagnostics_enabled:
        diagnostics_dir = Path(args.diagnostics_dir or run_dir / "diagnostics")
        write_diagnostic_outputs(diagnostics_dir, rows_out, diagnostics_out)
        print(f"wrote_diagnostics={diagnostics_dir}")
    print(f"wrote_csv={out_csv}")
    print(f"wrote_png={out_png}")
    for row in rows_out:
        print(
            f"[eval] epoch={row['epoch']} latent={row['latent_smooth_l1']:.6f} "
            f"scoap_mae={row['scoap_mae']:.6f} hard_bce={row['hard_bce']:.6f} "
            f"hard_macro_f1_tuned={row['hard_macro_f1_tuned']:.6f} "
            f"hard_recall_top10={row['hard_recall_at_top_10pct']:.6f} "
            f"predictive_score={row['predictive_score']:.6f}"
        )


if __name__ == "__main__":
    main()
