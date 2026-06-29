"""Train a fixed-checkpoint action-value ranker from rescored oracle actions."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oracle_action_value_probe import (  # noqa: E402
    kendall_tau,
    linear_fit,
    mean,
    pearson,
    safe_float,
    sign,
    spearman,
)


DEFAULT_NUMERIC_FEATURES = [
    "reward_pred",
    "fc_pred",
    "guarded_reward",
    "return_pred",
    "hard_reduction_total_pred",
    "hard_reduction_sa0_pred",
    "hard_reduction_sa1_pred",
    "hybrid_pred",
    "bounded_residual_hybrid_pred",
    "derived_hard_reduction_total_pred",
    "derived_hard_reduction_hybrid_pred",
    "derived_hard_reduction_sa0_pred",
    "derived_hard_reduction_sa1_pred",
    "derived_hard_count_pre_total_pred",
    "derived_hard_count_post_total_pred",
    "candidate_rank",
]
DEFAULT_CATEGORICAL_FEATURES = ["type", "candidate_strategy", "benchmark_id"]
FORBIDDEN_FEATURES = {
    "oracle_delta_tc",
    "oracle_delta_fault_coverage",
    "oracle_delta_pattern_count",
    "oracle_test_coverage",
    "oracle_fault_coverage",
    "oracle_hard_fault_count",
    "oracle_undetected_fault_count",
}
BASELINE_FIELDS = [
    "hybrid_pred",
    "hard_reduction_total_pred",
    "reward_pred",
    "bounded_residual_hybrid_pred",
    "derived_hard_reduction_hybrid_pred",
]
METRIC_FIELDS = [
    "split",
    "variant",
    "score_field",
    "kind",
    "groups",
    "actions",
    "mean_spearman",
    "mean_kendall_tau",
    "mean_pearson",
    "mean_top1_real_delta_tc",
    "mean_top1_regret",
    "negative_top1_rate",
    "mean_sign_accuracy",
    "pairwise_accuracy",
    "pairwise_count",
    "ndcg_at_5",
    "ndcg_at_10",
]


def parse_csv_values(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


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


def finite_target_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("status", "ok") == "ok" and math.isfinite(safe_float(row.get("oracle_delta_tc")))
    ]


def group_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("checkpoint_name", "")),
        str(row.get("benchmark_id", "")),
        str(row.get("state_id", "")),
        str(row.get("candidate_strategy", "")),
    )


def grouped_indices(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], list[int]]:
    out: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        out[group_key(row)].append(idx)
    return dict(out)


def rank_percentiles(values: list[float]) -> list[float]:
    finite = [(idx, value) for idx, value in enumerate(values) if math.isfinite(value)]
    out = [0.5 for _ in values]
    if len(finite) <= 1:
        return out
    finite_sorted = sorted(finite, key=lambda item: item[1])
    pos = 0
    while pos < len(finite_sorted):
        end = pos + 1
        while end < len(finite_sorted) and finite_sorted[end][1] == finite_sorted[pos][1]:
            end += 1
        percentile = ((pos + end - 1) / 2.0) / float(len(finite_sorted) - 1)
        for idx, _ in finite_sorted[pos:end]:
            out[idx] = percentile
        pos = end
    return out


@dataclass
class FeatureSpec:
    numeric_features: list[str]
    categorical_features: list[str]
    category_values: dict[str, list[str]]
    feature_names: list[str]
    mean: list[float]
    std: list[float]
    type_to_index: dict[str, int]


def available_numeric_features(fieldnames: list[str], requested: list[str]) -> list[str]:
    return [field for field in requested if field in fieldnames]


def fit_feature_spec(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
) -> FeatureSpec:
    for field in [*numeric_features, *categorical_features]:
        if field in FORBIDDEN_FEATURES:
            raise ValueError(f"forbidden feature requested: {field}")
    numeric = available_numeric_features(fieldnames, numeric_features)
    if not numeric:
        raise ValueError("no requested numeric features are present in input TSV")
    cats = [field for field in categorical_features if field in fieldnames]
    category_values: dict[str, list[str]] = {}
    for field in cats:
        values = sorted({str(row.get(field, "")) for row in rows if str(row.get(field, ""))})
        category_values[field] = [*values, "__UNK__"]

    raw_features = build_raw_features(rows, numeric, cats, category_values)
    mean_values = raw_features.mean(dim=0)
    std_values = raw_features.std(dim=0, unbiased=False).clamp_min(1e-6)

    feature_names: list[str] = []
    for field in numeric:
        feature_names.extend(
            [
                field,
                f"{field}__minus_group_mean",
                f"{field}__z_group",
                f"{field}__rank_pct_group",
            ]
        )
    for field in cats:
        for value in category_values[field]:
            feature_names.append(f"{field}={value}")

    type_values = category_values.get("type", ["__UNK__"])
    type_to_index = {value: idx for idx, value in enumerate(type_values)}
    return FeatureSpec(
        numeric_features=numeric,
        categorical_features=cats,
        category_values=category_values,
        feature_names=feature_names,
        mean=mean_values.tolist(),
        std=std_values.tolist(),
        type_to_index=type_to_index,
    )


def build_raw_features(
    rows: list[dict[str, str]],
    numeric_features: list[str],
    categorical_features: list[str],
    category_values: dict[str, list[str]],
) -> torch.Tensor:
    groups = grouped_indices(rows)
    per_field_values: dict[str, list[float]] = {}
    for field in numeric_features:
        per_field_values[field] = [safe_float(row.get(field), 0.0) for row in rows]

    group_stats: dict[str, dict[int, tuple[float, float, float]]] = {field: {} for field in numeric_features}
    for field in numeric_features:
        values = per_field_values[field]
        for _, indices in groups.items():
            group_vals = [values[idx] for idx in indices if math.isfinite(values[idx])]
            group_mean = sum(group_vals) / len(group_vals) if group_vals else 0.0
            group_std = (
                math.sqrt(sum((value - group_mean) ** 2 for value in group_vals) / len(group_vals))
                if group_vals
                else 0.0
            )
            ranks = rank_percentiles([values[idx] for idx in indices])
            for local_pos, idx in enumerate(indices):
                group_stats[field][idx] = (group_mean, group_std, ranks[local_pos])

    matrix: list[list[float]] = []
    for idx, row in enumerate(rows):
        features: list[float] = []
        for field in numeric_features:
            value = per_field_values[field][idx]
            if not math.isfinite(value):
                value = 0.0
            group_mean, group_std, rank_pct = group_stats[field][idx]
            features.extend(
                [
                    value,
                    value - group_mean,
                    (value - group_mean) / group_std if group_std > 1e-12 else 0.0,
                    rank_pct,
                ]
            )
        for field in categorical_features:
            value = str(row.get(field, ""))
            values = category_values[field]
            if value not in values:
                value = "__UNK__"
            features.extend(1.0 if value == item else 0.0 for item in values)
        matrix.append(features)
    return torch.tensor(matrix, dtype=torch.float32)


def build_features(rows: list[dict[str, str]], spec: FeatureSpec) -> torch.Tensor:
    raw = build_raw_features(rows, spec.numeric_features, spec.categorical_features, spec.category_values)
    mean_tensor = torch.tensor(spec.mean, dtype=torch.float32)
    std_tensor = torch.tensor(spec.std, dtype=torch.float32).clamp_min(1e-6)
    return (raw - mean_tensor) / std_tensor


def target_tensor(rows: list[dict[str, str]]) -> torch.Tensor:
    return torch.tensor([safe_float(row.get("oracle_delta_tc"), 0.0) for row in rows], dtype=torch.float32)


def type_indices(rows: list[dict[str, str]], spec: FeatureSpec) -> torch.Tensor:
    unk = spec.type_to_index.get("__UNK__", 0)
    return torch.tensor([spec.type_to_index.get(str(row.get("type", "")), unk) for row in rows], dtype=torch.long)


def pair_indices(
    rows: list[dict[str, str]],
    targets: torch.Tensor,
    pairwise_min_delta: float,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    out = []
    for _, indices in grouped_indices(rows).items():
        left: list[int] = []
        right: list[int] = []
        direction: list[float] = []
        for pos, i in enumerate(indices):
            for j in indices[pos + 1 :]:
                delta = float(targets[i] - targets[j])
                if abs(delta) < pairwise_min_delta:
                    continue
                if delta > 0.0:
                    left.append(i)
                    right.append(j)
                    direction.append(1.0)
                else:
                    left.append(j)
                    right.append(i)
                    direction.append(1.0)
        if left:
            out.append(
                (
                    torch.tensor(left, dtype=torch.long),
                    torch.tensor(right, dtype=torch.long),
                    torch.tensor(direction, dtype=torch.float32),
                )
            )
    return out


class Ranker(nn.Module):
    def __init__(self, input_dim: int, variant: str, type_count: int) -> None:
        super().__init__()
        self.variant = variant
        if variant in {"linear", "linear_l2", "action_type_linear"}:
            self.net = nn.Linear(input_dim, 1)
        elif variant == "mlp_small":
            self.net = nn.Sequential(nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, 1))
        else:
            raise ValueError(f"unsupported variant: {variant}")
        self.type_bias = nn.Embedding(type_count, 1) if variant == "action_type_linear" else None

    def forward(self, features: torch.Tensor, type_ids: torch.Tensor | None = None) -> torch.Tensor:
        scores = self.net(features).squeeze(-1)
        if self.type_bias is not None:
            if type_ids is None:
                raise ValueError("type_ids are required for action_type_linear")
            scores = scores + self.type_bias(type_ids).squeeze(-1)
        return scores


def pairwise_loss(
    scores: torch.Tensor,
    pairs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    temperature: float,
) -> torch.Tensor:
    losses = []
    for left, right, _ in pairs:
        losses.append(F.softplus(-((scores[left] - scores[right]) / max(temperature, 1e-6))).mean())
    if not losses:
        return scores.sum() * 0.0
    return torch.stack(losses).mean()


def l2_penalty(model: nn.Module) -> torch.Tensor:
    penalties = [param.pow(2).mean() for name, param in model.named_parameters() if "weight" in name]
    return torch.stack(penalties).mean() if penalties else torch.tensor(0.0)


def pairwise_accuracy_for(scores: list[float], targets: list[float], min_delta: float) -> tuple[float, int]:
    correct = 0
    total = 0
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            delta = targets[i] - targets[j]
            if abs(delta) < min_delta:
                continue
            pred_delta = scores[i] - scores[j]
            if pred_delta == 0.0:
                continue
            total += 1
            if pred_delta * delta > 0.0:
                correct += 1
    return (correct / total if total else float("nan")), total


def dcg(relevances: list[float]) -> float:
    return sum((rel / math.log2(idx + 2.0)) for idx, rel in enumerate(relevances))


def ndcg_at(scores: list[float], targets: list[float], k: int) -> float:
    if not scores:
        return float("nan")
    min_target = min(targets)
    rel = [max(0.0, value - min_target) for value in targets]
    order = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:k]
    ideal = sorted(range(len(targets)), key=lambda idx: targets[idx], reverse=True)[:k]
    ideal_dcg = dcg([rel[idx] for idx in ideal])
    if ideal_dcg <= 0.0:
        return float("nan")
    return dcg([rel[idx] for idx in order]) / ideal_dcg


def evaluate_score(
    rows: list[dict[str, str]],
    scores: list[float],
    *,
    split: str,
    variant: str,
    score_field: str,
    kind: str,
    pairwise_min_delta: float,
) -> dict[str, Any]:
    group_values: dict[tuple[str, str, str, str], list[tuple[float, float]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        target = safe_float(row.get("oracle_delta_tc"))
        if math.isfinite(score) and math.isfinite(target):
            group_values[group_key(row)].append((score, target))

    spearmans = []
    kendalls = []
    pearsons = []
    top1_deltas = []
    top1_regrets = []
    neg_top1 = []
    sign_accs = []
    pair_accs = []
    pair_counts = []
    ndcg5s = []
    ndcg10s = []
    actions = 0
    for pairs in group_values.values():
        if not pairs:
            continue
        group_scores = [item[0] for item in pairs]
        group_targets = [item[1] for item in pairs]
        actions += len(pairs)
        best_target = max(group_targets)
        top_idx = max(range(len(group_scores)), key=lambda idx: group_scores[idx])
        top_delta = group_targets[top_idx]
        top1_deltas.append(top_delta)
        top1_regrets.append(best_target - top_delta)
        neg_top1.append(1.0 if top_delta < 0.0 else 0.0)
        spearmans.append(spearman(group_scores, group_targets))
        kendalls.append(kendall_tau(group_scores, group_targets))
        pearsons.append(pearson(group_scores, group_targets))
        sign_accs.append(
            sum(1 for pred, target in zip(group_scores, group_targets) if sign(pred) == sign(target))
            / max(1, len(group_scores))
        )
        pair_acc, pair_count = pairwise_accuracy_for(group_scores, group_targets, pairwise_min_delta)
        pair_accs.append(pair_acc)
        pair_counts.append(pair_count)
        ndcg5s.append(ndcg_at(group_scores, group_targets, 5))
        ndcg10s.append(ndcg_at(group_scores, group_targets, 10))
    return {
        "split": split,
        "variant": variant,
        "score_field": score_field,
        "kind": kind,
        "groups": len(group_values),
        "actions": actions,
        "mean_spearman": mean(spearmans),
        "mean_kendall_tau": mean(kendalls),
        "mean_pearson": mean(pearsons),
        "mean_top1_real_delta_tc": mean(top1_deltas),
        "mean_top1_regret": mean(top1_regrets),
        "negative_top1_rate": mean(neg_top1),
        "mean_sign_accuracy": mean(sign_accs),
        "pairwise_accuracy": mean(pair_accs),
        "pairwise_count": sum(pair_counts),
        "ndcg_at_5": mean(ndcg5s),
        "ndcg_at_10": mean(ndcg10s),
    }


def baseline_metrics(
    rows_by_split: dict[str, list[dict[str, str]]],
    baseline_fields: list[str],
    pairwise_min_delta: float,
) -> list[dict[str, Any]]:
    out = []
    for split, rows in rows_by_split.items():
        for field in baseline_fields:
            scores = [safe_float(row.get(field)) for row in rows]
            if not any(math.isfinite(score) for score in scores):
                continue
            out.append(
                evaluate_score(
                    rows,
                    scores,
                    split=split,
                    variant="baseline",
                    score_field=field,
                    kind="baseline",
                    pairwise_min_delta=pairwise_min_delta,
                )
            )
    return out


def val_key(row: dict[str, Any]) -> tuple[float, float, float]:
    neg = safe_float(row.get("negative_top1_rate"), float("inf"))
    spearman_value = safe_float(row.get("mean_spearman"), float("-inf"))
    regret = safe_float(row.get("mean_top1_regret"), float("inf"))
    return (-neg, spearman_value, -regret)


def regret_first_val_key(row: dict[str, Any]) -> tuple[float, float, float]:
    regret = safe_float(row.get("mean_top1_regret"), float("inf"))
    spearman_value = safe_float(row.get("mean_spearman"), float("-inf"))
    neg = safe_float(row.get("negative_top1_rate"), float("inf"))
    return (-regret, spearman_value, -neg)


def train_variant(
    *,
    variant: str,
    train_features: torch.Tensor,
    train_types: torch.Tensor,
    train_targets: torch.Tensor,
    train_pairs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    val_rows: list[dict[str, str]],
    val_features: torch.Tensor,
    val_types: torch.Tensor,
    epochs: int,
    patience: int,
    lr: float,
    temperature: float,
    pairwise_min_delta: float,
    seed: int,
) -> tuple[Ranker, list[dict[str, Any]]]:
    torch.manual_seed(seed)
    model = Ranker(train_features.shape[1], variant, int(train_types.max().item()) + 1 if len(train_types) else 1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_key = (float("-inf"), float("-inf"), float("-inf"))
    stale = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        train_scores = model(train_features, train_types)
        loss = pairwise_loss(train_scores, train_pairs, temperature)
        if variant == "linear_l2":
            loss = loss + 1e-2 * l2_penalty(model)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_scores = model(val_features, val_types).tolist()
        val_metric = evaluate_score(
            val_rows,
            val_scores,
            split="expanded_val",
            variant=variant,
            score_field="ranker_score",
            kind="ranker",
            pairwise_min_delta=pairwise_min_delta,
        )
        current_key = val_key(val_metric)
        history.append(
            {
                "variant": variant,
                "epoch": epoch,
                "train_pairwise_loss": float(loss.detach().item()),
                **{f"val_{key}": value for key, value in val_metric.items() if key in METRIC_FIELDS},
            }
        )
        if current_key > best_key:
            best_key = current_key
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model, history


def model_scores(model: Ranker, features: torch.Tensor, type_ids: torch.Tensor) -> list[float]:
    model.eval()
    with torch.no_grad():
        return model(features, type_ids).tolist()


def feature_weight_rows(model: Ranker, spec: FeatureSpec, variant: str) -> list[dict[str, Any]]:
    rows = []
    if isinstance(model.net, nn.Linear):
        weights = model.net.weight.detach().cpu().view(-1).tolist()
        for name, weight in sorted(zip(spec.feature_names, weights), key=lambda item: abs(item[1]), reverse=True):
            rows.append({"variant": variant, "feature": name, "weight": weight})
    if model.type_bias is not None:
        weights = model.type_bias.weight.detach().cpu().view(-1).tolist()
        inv = {idx: value for value, idx in spec.type_to_index.items()}
        for idx, weight in enumerate(weights):
            rows.append({"variant": variant, "feature": f"type_bias={inv.get(idx, idx)}", "weight": weight})
    return rows


def format_float(value: Any) -> str:
    number = safe_float(value)
    return "nan" if not math.isfinite(number) else f"{number:.6f}"


def choose_best_ranker(metrics: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        row
        for row in metrics
        if row.get("split") == "expanded_val" and row.get("kind") == "ranker"
    ]
    return max(candidates, key=val_key, default=None)


def metric_lookup(metrics: list[dict[str, Any]], split: str, variant: str, score_field: str) -> dict[str, Any] | None:
    for row in metrics:
        if row.get("split") == split and row.get("variant") == variant and row.get("score_field") == score_field:
            return row
    return None


def verdict(metrics: list[dict[str, Any]], baseline_score_field: str) -> tuple[str, list[str], str | None]:
    best = choose_best_ranker(metrics)
    if not best:
        return "REJECT", ["no ranker metrics were produced"], None
    best_variant = str(best["variant"])
    val_base = metric_lookup(metrics, "expanded_val", "baseline", baseline_score_field)
    transfer_base = metric_lookup(metrics, "transfer", "baseline", baseline_score_field)
    transfer_ranker = metric_lookup(metrics, "transfer", best_variant, "ranker_score")
    if val_base is None or transfer_base is None or transfer_ranker is None:
        return "INCONCLUSIVE", ["missing baseline or transfer metrics"], best_variant

    reasons = []
    val_s = safe_float(best.get("mean_spearman"))
    val_base_s = safe_float(val_base.get("mean_spearman"))
    val_neg = safe_float(best.get("negative_top1_rate"))
    val_base_neg = safe_float(val_base.get("negative_top1_rate"))
    val_regret = safe_float(best.get("mean_top1_regret"))
    val_base_regret = safe_float(val_base.get("mean_top1_regret"))
    tr_neg = safe_float(transfer_ranker.get("negative_top1_rate"))
    tr_base_neg = safe_float(transfer_base.get("negative_top1_rate"))
    tr_top1 = safe_float(transfer_ranker.get("mean_top1_real_delta_tc"))
    tr_base_top1 = safe_float(transfer_base.get("mean_top1_real_delta_tc"))
    tr_regret = safe_float(transfer_ranker.get("mean_top1_regret"))
    tr_base_regret = safe_float(transfer_base.get("mean_top1_regret"))

    if val_neg > val_base_neg:
        return "REJECT", ["expanded negative_top1_rate worsened"], best_variant
    if tr_top1 < 0.0 and tr_base_top1 > 0.0:
        return "REJECT", ["transfer mean_top1_real_delta_tc became negative while baseline is positive"], best_variant

    checks = [
        (val_s >= val_base_s + 0.05, "expanded Spearman did not improve by >= 0.05"),
        (val_neg <= val_base_neg, "expanded negative_top1_rate worsened"),
        (val_regret <= val_base_regret, "expanded top1 regret worsened"),
        (tr_neg <= tr_base_neg + 0.10, "transfer negative_top1_rate safety failed"),
        (tr_top1 >= 0.0, "transfer top1 real delta is negative"),
        (tr_regret <= tr_base_regret + 0.01, "transfer top1 regret safety failed"),
    ]
    for ok, reason in checks:
        if not ok:
            reasons.append(reason)
    return ("PROMOTE" if not reasons else "INCONCLUSIVE"), reasons, best_variant


def group_top1_rows(
    rows: list[dict[str, str]],
    baseline_scores: list[float],
    ranker_scores: list[float],
    split: str,
    variant: str,
) -> list[dict[str, Any]]:
    out = []
    for key, indices in grouped_indices(rows).items():
        targets = [safe_float(rows[idx].get("oracle_delta_tc")) for idx in indices]
        if not targets:
            continue
        best_target = max(targets)
        base_idx = max(indices, key=lambda idx: baseline_scores[idx])
        ranker_idx = max(indices, key=lambda idx: ranker_scores[idx])
        base_delta = safe_float(rows[base_idx].get("oracle_delta_tc"))
        ranker_delta = safe_float(rows[ranker_idx].get("oracle_delta_tc"))
        out.append(
            {
                "split": split,
                "variant": variant,
                "checkpoint_name": key[0],
                "benchmark_id": key[1],
                "state_id": key[2],
                "candidate_strategy": key[3],
                "oracle_best_delta_tc": best_target,
                "baseline_top1_action": rows[base_idx].get("action_key"),
                "baseline_top1_delta_tc": base_delta,
                "ranker_top1_action": rows[ranker_idx].get("action_key"),
                "ranker_top1_delta_tc": ranker_delta,
                "delta_vs_baseline": ranker_delta - base_delta,
                "ranker_top1_regret": best_target - ranker_delta,
            }
        )
    return out


def markdown_report(
    metrics: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    top1_rows: list[dict[str, Any]],
    final_verdict: str,
    reasons: list[str],
    best_variant: str | None,
    baseline_score_field: str,
) -> str:
    lines = [
        "# Fixed-Checkpoint Action-Value Ranker",
        "",
        f"generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        f"verdict: `{final_verdict}`",
        f"best_variant: `{best_variant or 'none'}`",
        f"baseline_score_field: `{baseline_score_field}`",
        "",
    ]
    if reasons:
        lines.extend(["## Gate Reasons", ""])
        for reason in reasons:
            lines.append(f"- {reason}")
        lines.append("")

    lines.extend(
        [
            "## Metrics",
            "",
            "| split | variant | score | kind | Spearman | negative top1 | top1 real delta | top1 regret | pairwise acc | ndcg@10 |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in metrics:
        if row.get("split") not in {"expanded_val", "transfer"}:
            continue
        lines.append(
            "| {split} | `{variant}` | `{score}` | `{kind}` | {s} | {neg} | {top1} | {regret} | {pair} | {ndcg} |".format(
                split=row["split"],
                variant=row["variant"],
                score=row["score_field"],
                kind=row["kind"],
                s=format_float(row.get("mean_spearman")),
                neg=format_float(row.get("negative_top1_rate")),
                top1=format_float(row.get("mean_top1_real_delta_tc")),
                regret=format_float(row.get("mean_top1_regret")),
                pair=format_float(row.get("pairwise_accuracy")),
                ndcg=format_float(row.get("ndcg_at_10")),
            )
        )

    lines.extend(["", "## Linear Feature Weights", ""])
    linear_rows = [row for row in feature_rows if row["variant"] in {"linear", "linear_l2", "action_type_linear"}]
    for variant in sorted({row["variant"] for row in linear_rows}):
        lines.extend([f"### {variant}", "", "| feature | weight |", "|---|---:|"])
        for row in [item for item in linear_rows if item["variant"] == variant][:20]:
            lines.append(f"| `{row['feature']}` | {format_float(row['weight'])} |")
        lines.append("")

    if top1_rows:
        sorted_groups = sorted(top1_rows, key=lambda row: safe_float(row.get("delta_vs_baseline")), reverse=True)
        lines.extend(["## Top Improved Groups", "", "| split | benchmark | strategy | baseline top1 | ranker top1 | gain |", "|---|---|---|---:|---:|---:|"])
        for row in sorted_groups[:10]:
            lines.append(
                "| {split} | `{bench}` | `{strategy}` | {base} | {ranker} | {gain} |".format(
                    split=row["split"],
                    bench=row["benchmark_id"],
                    strategy=row["candidate_strategy"],
                    base=format_float(row["baseline_top1_delta_tc"]),
                    ranker=format_float(row["ranker_top1_delta_tc"]),
                    gain=format_float(row["delta_vs_baseline"]),
                )
            )
        lines.extend(["", "## Top Worsened Groups", "", "| split | benchmark | strategy | baseline top1 | ranker top1 | gain |", "|---|---|---|---:|---:|---:|"])
        for row in sorted_groups[-10:]:
            lines.append(
                "| {split} | `{bench}` | `{strategy}` | {base} | {ranker} | {gain} |".format(
                    split=row["split"],
                    bench=row["benchmark_id"],
                    strategy=row["candidate_strategy"],
                    base=format_float(row["baseline_top1_delta_tc"]),
                    ranker=format_float(row["ranker_top1_delta_tc"]),
                    gain=format_float(row["delta_vs_baseline"]),
                )
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-rescored", required=True)
    parser.add_argument("--val-rescored", required=True)
    parser.add_argument("--transfer-rescored", required=True)
    parser.add_argument("--variants", default="linear,linear_l2,mlp_small,action_type_linear")
    parser.add_argument("--baseline-score-field", default="hybrid_pred")
    parser.add_argument("--baseline-score-fields", default=",".join(BASELINE_FIELDS))
    parser.add_argument("--numeric-features", default=",".join(DEFAULT_NUMERIC_FEATURES))
    parser.add_argument("--categorical-features", default=",".join(DEFAULT_CATEGORICAL_FEATURES))
    parser.add_argument("--pairwise-min-delta", type=float, default=0.001)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    ranker_dir = out_dir / "rankers"
    ranker_dir.mkdir(parents=True, exist_ok=True)

    train_fields, train_rows_raw = read_tsv(Path(args.train_rescored))
    val_fields, val_rows_raw = read_tsv(Path(args.val_rescored))
    transfer_fields, transfer_rows_raw = read_tsv(Path(args.transfer_rescored))
    train_rows = finite_target_rows(train_rows_raw)
    val_rows = finite_target_rows(val_rows_raw)
    transfer_rows = finite_target_rows(transfer_rows_raw)
    if not train_rows or not val_rows:
        raise ValueError("train and val rescored TSVs must contain finite oracle_delta_tc rows")

    numeric_features = parse_csv_values(args.numeric_features)
    categorical_features = parse_csv_values(args.categorical_features)
    spec = fit_feature_spec(train_rows, train_fields, numeric_features, categorical_features)

    train_features = build_features(train_rows, spec)
    val_features = build_features(val_rows, spec)
    transfer_features = build_features(transfer_rows, spec)
    train_targets = target_tensor(train_rows)
    train_types = type_indices(train_rows, spec)
    val_types = type_indices(val_rows, spec)
    transfer_types = type_indices(transfer_rows, spec)
    train_pairs = pair_indices(train_rows, train_targets, args.pairwise_min_delta)
    if not train_pairs:
        raise ValueError("no train pairs pass --pairwise-min-delta")

    rows_by_split = {"train": train_rows, "expanded_val": val_rows, "transfer": transfer_rows}
    baseline_fields = [field for field in parse_csv_values(args.baseline_score_fields) if field in set(train_fields + val_fields + transfer_fields)]
    metrics: list[dict[str, Any]] = baseline_metrics(rows_by_split, baseline_fields, args.pairwise_min_delta)
    feature_weight_rows_all: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []

    prediction_rows = {
        "train": [dict(row) for row in train_rows],
        "expanded_val": [dict(row) for row in val_rows],
        "transfer": [dict(row) for row in transfer_rows],
    }
    variant_scores_by_split: dict[tuple[str, str], list[float]] = {}

    for variant in parse_csv_values(args.variants):
        model, history = train_variant(
            variant=variant,
            train_features=train_features,
            train_types=train_types,
            train_targets=train_targets,
            train_pairs=train_pairs,
            val_rows=val_rows,
            val_features=val_features,
            val_types=val_types,
            epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
            temperature=args.temperature,
            pairwise_min_delta=args.pairwise_min_delta,
            seed=args.seed,
        )
        history_rows.extend(history)
        torch.save(
            {
                "variant": variant,
                "state_dict": model.state_dict(),
                "feature_spec": {
                    "numeric_features": spec.numeric_features,
                    "categorical_features": spec.categorical_features,
                    "category_values": spec.category_values,
                    "feature_names": spec.feature_names,
                    "mean": spec.mean,
                    "std": spec.std,
                    "type_to_index": spec.type_to_index,
                },
                "args": vars(args),
            },
            ranker_dir / f"{variant}.pt",
        )
        feature_weight_rows_all.extend(feature_weight_rows(model, spec, variant))

        split_features = {
            "train": (train_rows, train_features, train_types),
            "expanded_val": (val_rows, val_features, val_types),
            "transfer": (transfer_rows, transfer_features, transfer_types),
        }
        for split, (rows, features, type_ids) in split_features.items():
            scores = model_scores(model, features, type_ids)
            variant_scores_by_split[(variant, split)] = scores
            for row, score in zip(prediction_rows[split], scores):
                row[f"{variant}_ranker_score"] = score
            metrics.append(
                evaluate_score(
                    rows,
                    scores,
                    split=split,
                    variant=variant,
                    score_field="ranker_score",
                    kind="ranker",
                    pairwise_min_delta=args.pairwise_min_delta,
                )
            )

    final_verdict, reasons, best_variant = verdict(metrics, args.baseline_score_field)
    top1_rows: list[dict[str, Any]] = []
    if best_variant:
        for split, rows in rows_by_split.items():
            baseline_scores = [safe_float(row.get(args.baseline_score_field)) for row in rows]
            ranker_scores = variant_scores_by_split.get((best_variant, split), [])
            if ranker_scores:
                top1_rows.extend(group_top1_rows(rows, baseline_scores, ranker_scores, split, best_variant))

    metric_rows = [{field: row.get(field, "") for field in METRIC_FIELDS} for row in metrics]
    write_tsv(out_dir / "ranker_metrics.tsv", metric_rows, METRIC_FIELDS)
    write_tsv(out_dir / "feature_weights.tsv", feature_weight_rows_all, ["variant", "feature", "weight"])
    write_tsv(out_dir / "train_history.tsv", history_rows, sorted({key for row in history_rows for key in row.keys()}))
    prediction_fields = sorted({key for rows in prediction_rows.values() for row in rows for key in row.keys()})
    write_tsv(out_dir / "ranker_predictions_train.tsv", prediction_rows["train"], prediction_fields)
    write_tsv(out_dir / "ranker_predictions_val.tsv", prediction_rows["expanded_val"], prediction_fields)
    write_tsv(out_dir / "ranker_predictions_transfer.tsv", prediction_rows["transfer"], prediction_fields)
    write_tsv(
        out_dir / "top1_group_deltas.tsv",
        top1_rows,
        [
            "split",
            "variant",
            "checkpoint_name",
            "benchmark_id",
            "state_id",
            "candidate_strategy",
            "oracle_best_delta_tc",
            "baseline_top1_action",
            "baseline_top1_delta_tc",
            "ranker_top1_action",
            "ranker_top1_delta_tc",
            "delta_vs_baseline",
            "ranker_top1_regret",
        ],
    )
    (out_dir / "ranker_report.md").write_text(
        markdown_report(
            metrics,
            feature_weight_rows_all,
            top1_rows,
            final_verdict,
            reasons,
            best_variant,
            args.baseline_score_field,
        )
    )
    handoff = {
        "mode": "fix",
        "objective": "fixed-checkpoint action-value ranker",
        "status": "completed",
        "out_dir": str(out_dir),
        "verdict": final_verdict,
        "best_variant": best_variant,
        "reasons": reasons,
        "inputs": {
            "train_rescored": args.train_rescored,
            "val_rescored": args.val_rescored,
            "transfer_rescored": args.transfer_rescored,
        },
        "outputs": {
            "ranker_metrics": str(out_dir / "ranker_metrics.tsv"),
            "ranker_report": str(out_dir / "ranker_report.md"),
            "handoff": str(out_dir / "handoff.json"),
        },
        "records": {
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "transfer_rows": len(transfer_rows),
            "train_pair_groups": len(train_pairs),
        },
    }
    write_json(out_dir / "handoff.json", handoff)
    print(json.dumps(handoff, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
