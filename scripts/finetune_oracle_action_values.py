"""Finetune action-value heads on backend-labeled oracle action groups."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oracle_action_value_probe import (  # noqa: E402
    checkpoint_id,
    parse_csv_values,
    read_tsv,
    safe_float,
    write_json,
)
from tpi_jepa.bench import parse_bench  # noqa: E402
from tpi_jepa.features import (  # noqa: E402
    action_type_to_id,
    make_action_relation_features,
    make_base_node_features,
    make_state_features,
)
from tpi_jepa.graph import build_graph  # noqa: E402
from tpi_jepa.labels import find_bench_path  # noqa: E402
from tpi_jepa.plan import load_checkpoint, set_real_fault_context  # noqa: E402


HISTORY_FIELDS = [
    "epoch",
    "bounded_residual_alpha",
    "train_loss",
    "train_value_loss",
    "train_rank_loss",
    "train_groups",
    "train_pairs",
    "val_rank_loss",
    "val_spearman",
    "val_negative_top1_rate",
    "val_top1_regret",
    "val_groups",
    "val_pairs",
    "is_best",
]


def group_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("benchmark_id", "")),
        str(row.get("state_id", "")),
        str(row.get("candidate_strategy", "")),
    )


def load_oracle_groups(path: Path, max_actions_per_group: int | None = None) -> list[list[dict[str, Any]]]:
    rows = read_tsv(path)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("state_id") != "initial":
            raise ValueError("finetune_oracle_action_values.py v1 supports only state_id=initial")
        delta = safe_float(row.get("oracle_delta_tc"))
        if not math.isfinite(delta):
            continue
        grouped[group_key(row)].append(row)
    groups = []
    for _, group in sorted(grouped.items()):
        group = sorted(group, key=lambda row: int(float(row.get("candidate_rank") or 0)))
        if max_actions_per_group is not None and max_actions_per_group > 0:
            group = group[:max_actions_per_group]
        if group:
            groups.append(group)
    if not groups:
        raise ValueError(f"no oracle groups with finite actions in {path}")
    return groups


def set_trainable_parts(model: torch.nn.Module, train_heads: str, train_scope: str = "heads") -> list[str]:
    train_scope = str(train_scope or "heads").lower()
    if train_scope == "bounded_residual":
        for param in model.parameters():
            param.requires_grad = False
        return ["bounded_residual_alpha"]
    if train_scope == "planner_joint":
        prefixes = [
            "action_encoder",
            "dynamics",
            "reward_head",
            "return_head",
            "hard_reduction_head",
        ]
        for name, param in model.named_parameters():
            param.requires_grad = any(name.startswith(prefix) for prefix in prefixes)
        for param in model.online_encoder.parameters():
            param.requires_grad = False
        for param in model.target_encoder.parameters():
            param.requires_grad = False
        return prefixes
    if train_scope == "planner_joint_frozen_hard":
        prefixes = [
            "action_encoder",
            "dynamics",
            "reward_head",
            "return_head",
        ]
        for name, param in model.named_parameters():
            param.requires_grad = any(name.startswith(prefix) for prefix in prefixes)
        for param in model.hard_reduction_head.parameters():
            param.requires_grad = False
        for param in model.online_encoder.parameters():
            param.requires_grad = False
        for param in model.target_encoder.parameters():
            param.requires_grad = False
        return prefixes
    if train_scope != "heads":
        raise ValueError(
            f"unsupported --train-scope {train_scope!r}; "
            "expected heads, bounded_residual, planner_joint, or planner_joint_frozen_hard"
        )

    requested = {item.lower() for item in parse_csv_values(train_heads)}
    prefix_map = {
        "reward": "reward_head",
        "return": "return_head",
        "hard_reduction": "hard_reduction_head",
        "all": "",
    }
    if "all" in requested:
        for param in model.parameters():
            param.requires_grad = True
        return ["all"]
    prefixes = [prefix_map[item] for item in requested if item in prefix_map]
    if not prefixes:
        raise ValueError(f"no supported --train-heads entries in {train_heads!r}")
    for name, param in model.named_parameters():
        param.requires_grad = any(name.startswith(prefix) for prefix in prefixes)
    for param in model.target_encoder.parameters():
        param.requires_grad = False
    return prefixes


def graph_and_base_for(
    benchmark_id: str,
    graph_cache: dict[str, Any],
    base_cache: dict[str, torch.Tensor],
    config: dict[str, Any],
) -> tuple[Any, torch.Tensor]:
    if benchmark_id not in graph_cache:
        set_real_fault_context(benchmark_id, config.get("real_fault_priors"), config.get("activation_priors"))
        graph_cache[benchmark_id] = build_graph(parse_bench(find_bench_path(benchmark_id)))
    if benchmark_id not in base_cache:
        graph = graph_cache[benchmark_id]
        base_cache[benchmark_id] = make_base_node_features(
            graph,
            str(config.get("feature_mode", "basic")),
            benchmark_id=benchmark_id,
            real_fault_prior_path=config.get("real_fault_priors") or config.get("real_fault_prior_path"),
            activation_prior_path=config.get("activation_priors") or config.get("activation_prior_path"),
        )
    return graph_cache[benchmark_id], base_cache[benchmark_id]


def predict_group_scores(
    model,
    config: dict[str, Any],
    group: list[dict[str, Any]],
    graph_cache: dict[str, Any],
    base_cache: dict[str, torch.Tensor],
    device: torch.device,
    bounded_residual_alpha: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    benchmark_id = str(group[0]["benchmark_id"])
    graph, base_features = graph_and_base_for(benchmark_id, graph_cache, base_cache, config)
    selected: list[tuple[str, str]] = []
    x_state = make_state_features(graph, selected, base_features).to(device)
    z_state = model.online_encoder(
        x_state,
        graph.edge_src.to(device),
        graph.edge_dst.to(device),
        graph.gate_type_ids.to(device),
    )
    relation_mode = str(config.get("relation_mode", "basic"))
    relation_depth = int(config.get("relation_depth", 8))
    node_ids = {name: idx for idx, name in enumerate(graph.node_names)}
    coverage_scale = float(getattr(model, "coverage_scale", config.get("coverage_scale", 100.0)))
    if bounded_residual_alpha is None:
        alpha_value = float(config.get("bounded_residual_alpha", getattr(model, "bounded_residual_alpha", 1.0)))
        alpha_bound = float(config.get("bounded_residual_alpha_bound", getattr(model, "bounded_residual_alpha_bound", 0.25)))
        bounded_residual_alpha = torch.tensor(
            max(-alpha_bound, min(alpha_value, alpha_bound)),
            dtype=base_features.dtype,
            device=device,
        )
    score_lists: dict[str, list[torch.Tensor]] = {
        "reward_pred": [],
        "return_pred": [],
        "guarded_reward": [],
        "hard_reduction_total_pred": [],
        "hybrid_pred": [],
        "bounded_residual_hybrid_pred": [],
        "derived_hard_reduction_total_pred": [],
        "derived_hard_reduction_hybrid_pred": [],
    }
    for row in group:
        node = row["node"]
        action_type = row["type"]
        if node not in node_ids:
            raise ValueError(f"node {node!r} from oracle TSV not found in {benchmark_id}")
        action_node_id = node_ids[node]
        relation = make_action_relation_features(graph, action_node_id, relation_mode, relation_depth).to(device)
        pred = model.predict_from_latent(
            z_state,
            action_node_id,
            action_type_to_id(action_type),
            relation,
            include_aux_heads=False,
        )
        reward_pred = pred["reward_pred"]
        return_pred = pred["return_pred"]
        hard_reduction_pred = pred["hard_reduction_pred"].view(-1)
        hard_reduction_total = hard_reduction_pred[0] if hard_reduction_pred.numel() > 0 else reward_pred.new_zeros(())
        derived_reduction = pred.get("derived_hard_reduction_pred")
        derived_reduction = (
            derived_reduction.view(-1)
            if derived_reduction is not None
            else hard_reduction_pred.new_zeros(3)
        )
        derived_hard_reduction_total = (
            derived_reduction[0] if derived_reduction.numel() > 0 else reward_pred.new_zeros(())
        )
        score_lists["reward_pred"].append(reward_pred)
        score_lists["return_pred"].append(return_pred)
        score_lists["guarded_reward"].append(torch.minimum(reward_pred, return_pred))
        score_lists["hard_reduction_total_pred"].append(hard_reduction_total)
        score_lists["hybrid_pred"].append(return_pred + reward_pred + hard_reduction_total * coverage_scale)
        score_lists["bounded_residual_hybrid_pred"].append(
            hard_reduction_total * coverage_scale + bounded_residual_alpha * (reward_pred + return_pred)
        )
        score_lists["derived_hard_reduction_total_pred"].append(derived_hard_reduction_total)
        score_lists["derived_hard_reduction_hybrid_pred"].append(derived_hard_reduction_total * coverage_scale)
    return {field: torch.stack(values) for field, values in score_lists.items()}


def require_score(scores: dict[str, torch.Tensor], field: str) -> torch.Tensor:
    if field not in scores:
        valid = ", ".join(sorted(scores))
        raise ValueError(f"unsupported score field {field!r}; valid fields: {valid}")
    return scores[field]


def pairwise_rank_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    min_delta: float,
    temperature: float,
) -> tuple[torch.Tensor, int]:
    losses = []
    count = 0
    temp = max(1e-6, float(temperature))
    for i in range(targets.numel()):
        for j in range(i + 1, targets.numel()):
            diff = targets[i] - targets[j]
            if diff.abs().item() < float(min_delta):
                continue
            order = diff.sign().detach()
            losses.append(F.softplus(-order * (preds[i] - preds[j]) / temp))
            count += 1
    if not losses:
        return torch.zeros((), dtype=preds.dtype, device=preds.device), 0
    return torch.stack(losses).mean(), count


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    out = [0.0 for _ in values]
    pos = 0
    while pos < len(order):
        end = pos + 1
        while end < len(order) and values[order[end]] == values[order[pos]]:
            end += 1
        avg_rank = (pos + 1 + end) / 2.0
        for idx in order[pos:end]:
            out[idx] = avg_rank
        pos = end
    return out


def pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return float("nan")
    left, right = zip(*pairs)
    mx = sum(left) / len(left)
    my = sum(right) / len(right)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    dx = math.sqrt(sum((x - mx) ** 2 for x in left))
    dy = math.sqrt(sum((y - my) ** 2 for y in right))
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def spearman(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return float("nan")
    left, right = zip(*pairs)
    return pearson(ranks(list(left)), ranks(list(right)))


def mean_finite(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else float("nan")


def validation_metrics(
    model,
    config: dict[str, Any],
    groups: list[list[dict[str, Any]]],
    graph_cache: dict[str, Any],
    base_cache: dict[str, torch.Tensor],
    device: torch.device,
    score_field: str,
    coverage_scale: float,
    pairwise_min_delta: float,
    pairwise_temperature: float,
    bounded_residual_alpha: torch.Tensor | None = None,
) -> dict[str, Any]:
    model.eval()
    rank_losses: list[float] = []
    pair_total = 0
    spearmans: list[float] = []
    negative_top1: list[float] = []
    regrets: list[float] = []
    with torch.no_grad():
        for group in groups:
            scores = predict_group_scores(
                model,
                config,
                group,
                graph_cache,
                base_cache,
                device,
                bounded_residual_alpha=bounded_residual_alpha,
            )
            preds = require_score(scores, score_field)
            targets = torch.tensor(
                [coverage_scale * safe_float(row.get("oracle_delta_tc")) for row in group],
                dtype=preds.dtype,
                device=device,
            )
            rank_loss, pair_count = pairwise_rank_loss(
                preds,
                targets,
                pairwise_min_delta * coverage_scale,
                pairwise_temperature,
            )
            rank_losses.append(float(rank_loss.detach().cpu().item()))
            pair_total += int(pair_count)
            pred_values = [float(value) for value in preds.detach().cpu().tolist()]
            target_values = [safe_float(row.get("oracle_delta_tc")) for row in group]
            spearmans.append(spearman(pred_values, target_values))
            finite = [
                (pred, target)
                for pred, target in zip(pred_values, target_values)
                if math.isfinite(pred) and math.isfinite(target)
            ]
            if finite:
                top1_target = max(finite, key=lambda item: item[0])[1]
                best_target = max(target for _, target in finite)
                negative_top1.append(float(top1_target < 0.0))
                regrets.append(best_target - top1_target)
    return {
        "val_rank_loss": mean_finite(rank_losses),
        "val_spearman": mean_finite(spearmans),
        "val_negative_top1_rate": mean_finite(negative_top1),
        "val_top1_regret": mean_finite(regrets),
        "val_groups": len(groups),
        "val_pairs": pair_total,
    }


def better_validation(candidate: dict[str, Any], incumbent: dict[str, Any] | None) -> bool:
    if incumbent is None:
        return True
    cand_key = (
        safe_float(candidate.get("val_spearman"), float("-inf")),
        -safe_float(candidate.get("val_negative_top1_rate"), float("inf")),
        -safe_float(candidate.get("val_top1_regret"), float("inf")),
        -safe_float(candidate.get("val_rank_loss"), float("inf")),
    )
    inc_key = (
        safe_float(incumbent.get("val_spearman"), float("-inf")),
        -safe_float(incumbent.get("val_negative_top1_rate"), float("inf")),
        -safe_float(incumbent.get("val_top1_regret"), float("inf")),
        -safe_float(incumbent.get("val_rank_loss"), float("inf")),
    )
    return cand_key > inc_key


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(
    path: Path,
    model,
    config: dict[str, Any],
    source_checkpoint: str,
    feature_dim: int,
    relation_dim: int,
    extra_config: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out_config = dict(config)
    out_config["oracle_action_value_finetune"] = True
    out_config["oracle_action_value_source_checkpoint"] = source_checkpoint
    if extra_config:
        out_config.update(extra_config)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": out_config,
            "feature_dim": feature_dim,
            "relation_dim": relation_dim,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--oracle-actions", required=True)
    parser.add_argument("--val-oracle-actions", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-oracle-value", type=float, default=1.0)
    parser.add_argument("--lambda-oracle-rank", type=float, default=0.5)
    parser.add_argument("--pairwise-min-delta", type=float, default=0.001)
    parser.add_argument("--pairwise-temperature", type=float, default=1.0)
    parser.add_argument("--train-heads", default="reward,return")
    parser.add_argument("--train-scope", default="heads", choices=["heads", "bounded_residual", "planner_joint", "planner_joint_frozen_hard"])
    parser.add_argument("--ranking-score-field", default="reward_pred")
    parser.add_argument("--value-score-field", default="reward_pred")
    parser.add_argument("--bounded-residual-alpha-init", type=float, default=0.0)
    parser.add_argument("--bounded-residual-alpha-bound", type=float, default=0.25)
    parser.add_argument("--max-actions-per-group", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--plan-device", default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device_name = args.plan_device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    model, config = load_checkpoint(args.checkpoint, device)
    trainable_prefixes = set_trainable_parts(model, args.train_heads, args.train_scope)
    residual_alpha_raw = None
    if args.train_scope == "bounded_residual":
        alpha_bound = max(1e-6, float(args.bounded_residual_alpha_bound))
        alpha_init = max(-alpha_bound + 1e-6, min(float(args.bounded_residual_alpha_init), alpha_bound - 1e-6))
        residual_alpha_raw = torch.nn.Parameter(torch.tensor(math.atanh(alpha_init / alpha_bound), device=device))
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if residual_alpha_raw is not None:
        trainable_params.append(residual_alpha_raw)
    if not trainable_params:
        raise RuntimeError("no trainable parameters selected")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    groups = load_oracle_groups(
        Path(args.oracle_actions),
        max_actions_per_group=int(args.max_actions_per_group) or None,
    )
    val_groups = (
        load_oracle_groups(Path(args.val_oracle_actions), max_actions_per_group=int(args.max_actions_per_group) or None)
        if args.val_oracle_actions
        else []
    )
    coverage_scale = float(config.get("coverage_scale", 100.0))
    out_dir = Path(args.out_dir)
    graph_cache: dict[str, Any] = {}
    base_cache: dict[str, torch.Tensor] = {}
    first_graph, first_base = graph_and_base_for(str(groups[0][0]["benchmark_id"]), graph_cache, base_cache, config)
    feature_dim = first_base.shape[1] + 3
    relation_dim = make_action_relation_features(first_graph, 0, str(config.get("relation_mode", "basic")), int(config.get("relation_depth", 8))).shape[1]

    history = []
    best_row: dict[str, Any] | None = None
    for epoch in range(1, int(args.epochs) + 1):
        random.shuffle(groups)
        model.train()
        totals = {"loss": 0.0, "value": 0.0, "rank": 0.0, "pairs": 0.0}
        for group in groups:
            bounded_residual_alpha = None
            if residual_alpha_raw is not None:
                bounded_residual_alpha = float(args.bounded_residual_alpha_bound) * torch.tanh(residual_alpha_raw)
            scores = predict_group_scores(
                model,
                config,
                group,
                graph_cache,
                base_cache,
                device,
                bounded_residual_alpha=bounded_residual_alpha,
            )
            value_preds = require_score(scores, args.value_score_field)
            rank_preds = require_score(scores, args.ranking_score_field)
            targets = torch.tensor(
                [coverage_scale * safe_float(row.get("oracle_delta_tc")) for row in group],
                dtype=rank_preds.dtype,
                device=device,
            )
            value_loss = F.smooth_l1_loss(value_preds, targets)
            if "return_head" in trainable_prefixes or "all" in trainable_prefixes:
                value_loss = 0.5 * value_loss + 0.5 * F.smooth_l1_loss(scores["return_pred"], targets)
            rank_loss, pair_count = pairwise_rank_loss(
                rank_preds,
                targets,
                args.pairwise_min_delta * coverage_scale,
                args.pairwise_temperature,
            )
            loss = float(args.lambda_oracle_value) * value_loss + float(args.lambda_oracle_rank) * rank_loss
            if not loss.requires_grad:
                totals["loss"] += float(loss.detach().cpu().item())
                totals["value"] += float(value_loss.detach().cpu().item())
                totals["rank"] += float(rank_loss.detach().cpu().item())
                totals["pairs"] += float(pair_count)
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            totals["loss"] += float(loss.detach().cpu().item())
            totals["value"] += float(value_loss.detach().cpu().item())
            totals["rank"] += float(rank_loss.detach().cpu().item())
            totals["pairs"] += float(pair_count)
        denom = max(1, len(groups))
        row = {
            "epoch": epoch,
            "bounded_residual_alpha": (
                float((float(args.bounded_residual_alpha_bound) * torch.tanh(residual_alpha_raw)).detach().cpu().item())
                if residual_alpha_raw is not None
                else ""
            ),
            "train_loss": totals["loss"] / denom,
            "train_value_loss": totals["value"] / denom,
            "train_rank_loss": totals["rank"] / denom,
            "train_groups": len(groups),
            "train_pairs": int(totals["pairs"]),
        }
        if val_groups:
            row.update(
                validation_metrics(
                    model,
                    config,
                    val_groups,
                    graph_cache,
                    base_cache,
                    device,
                    args.ranking_score_field,
                    coverage_scale,
                    args.pairwise_min_delta,
                    args.pairwise_temperature,
                    bounded_residual_alpha=(
                        float(args.bounded_residual_alpha_bound) * torch.tanh(residual_alpha_raw)
                        if residual_alpha_raw is not None
                        else None
                    ),
                )
            )
            row["is_best"] = int(better_validation(row, best_row))
            if row["is_best"]:
                best_row = dict(row)
                save_checkpoint(
                    out_dir / "best.pt",
                    model,
                    config,
                    args.checkpoint,
                    feature_dim,
                    relation_dim,
                    extra_config={
                        "bounded_residual_alpha": row["bounded_residual_alpha"],
                        "bounded_residual_alpha_bound": float(args.bounded_residual_alpha_bound),
                    },
                )
        else:
            row.update(
                {
                    "val_rank_loss": "",
                    "val_spearman": "",
                    "val_negative_top1_rate": "",
                    "val_top1_regret": "",
                    "val_groups": "",
                    "val_pairs": "",
                    "is_best": "",
                }
            )
        history.append(row)
        print(json.dumps(row, sort_keys=True))

    final_alpha = (
        float((float(args.bounded_residual_alpha_bound) * torch.tanh(residual_alpha_raw)).detach().cpu().item())
        if residual_alpha_raw is not None
        else None
    )
    save_checkpoint(
        out_dir / "candidate.pt",
        model,
        config,
        args.checkpoint,
        feature_dim,
        relation_dim,
        extra_config=(
            {
                "bounded_residual_alpha": final_alpha,
                "bounded_residual_alpha_bound": float(args.bounded_residual_alpha_bound),
            }
            if final_alpha is not None
            else None
        ),
    )
    write_history(out_dir / "history.tsv", history)
    handoff = {
        "mode": "fix",
        "status": "progressed",
        "objective": "oracle action-value finetune",
        "source_checkpoint": args.checkpoint,
        "source_checkpoint_id": checkpoint_id(args.checkpoint),
        "oracle_actions": args.oracle_actions,
        "val_oracle_actions": args.val_oracle_actions,
        "out_dir": str(out_dir),
        "train_scope": args.train_scope,
        "trainable_prefixes": trainable_prefixes,
        "ranking_score_field": args.ranking_score_field,
        "value_score_field": args.value_score_field,
        "bounded_residual_alpha_bound": float(args.bounded_residual_alpha_bound),
        "groups": len(groups),
        "epochs": int(args.epochs),
        "outputs": {
            "candidate": str(out_dir / "candidate.pt"),
            "best": str(out_dir / "best.pt") if val_groups else "",
            "history": str(out_dir / "history.tsv"),
        },
        "final": history[-1] if history else {},
        "best": best_row or {},
    }
    write_json(out_dir / "handoff.json", handoff)
    print(json.dumps(handoff, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
