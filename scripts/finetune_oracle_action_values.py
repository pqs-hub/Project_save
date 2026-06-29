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
    "train_loss",
    "train_value_loss",
    "train_rank_loss",
    "train_groups",
    "train_pairs",
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


def set_trainable_parts(model: torch.nn.Module, train_heads: str) -> list[str]:
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
) -> tuple[torch.Tensor, torch.Tensor]:
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
    reward_preds = []
    return_preds = []
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
        reward_preds.append(pred["reward_pred"])
        return_preds.append(pred["return_pred"])
    return torch.stack(reward_preds), torch.stack(return_preds)


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


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(path: Path, model, config: dict[str, Any], source_checkpoint: str, feature_dim: int, relation_dim: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out_config = dict(config)
    out_config["oracle_action_value_finetune"] = True
    out_config["oracle_action_value_source_checkpoint"] = source_checkpoint
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
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-oracle-value", type=float, default=1.0)
    parser.add_argument("--lambda-oracle-rank", type=float, default=0.5)
    parser.add_argument("--pairwise-min-delta", type=float, default=0.001)
    parser.add_argument("--pairwise-temperature", type=float, default=1.0)
    parser.add_argument("--train-heads", default="reward,return")
    parser.add_argument("--max-actions-per-group", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--plan-device", default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device_name = args.plan_device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    model, config = load_checkpoint(args.checkpoint, device)
    trainable_prefixes = set_trainable_parts(model, args.train_heads)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("no trainable parameters selected")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    groups = load_oracle_groups(
        Path(args.oracle_actions),
        max_actions_per_group=int(args.max_actions_per_group) or None,
    )
    coverage_scale = float(config.get("coverage_scale", 100.0))
    out_dir = Path(args.out_dir)
    graph_cache: dict[str, Any] = {}
    base_cache: dict[str, torch.Tensor] = {}
    first_graph, first_base = graph_and_base_for(str(groups[0][0]["benchmark_id"]), graph_cache, base_cache, config)
    feature_dim = first_base.shape[1] + 3
    relation_dim = make_action_relation_features(first_graph, 0, str(config.get("relation_mode", "basic")), int(config.get("relation_depth", 8))).shape[1]

    history = []
    for epoch in range(1, int(args.epochs) + 1):
        random.shuffle(groups)
        model.train()
        totals = {"loss": 0.0, "value": 0.0, "rank": 0.0, "pairs": 0.0}
        for group in groups:
            reward_preds, return_preds = predict_group_scores(model, config, group, graph_cache, base_cache, device)
            targets = torch.tensor(
                [coverage_scale * safe_float(row.get("oracle_delta_tc")) for row in group],
                dtype=reward_preds.dtype,
                device=device,
            )
            value_loss = F.smooth_l1_loss(reward_preds, targets)
            if "return_head" in trainable_prefixes or "all" in trainable_prefixes:
                value_loss = 0.5 * value_loss + 0.5 * F.smooth_l1_loss(return_preds, targets)
            rank_loss, pair_count = pairwise_rank_loss(
                reward_preds,
                targets,
                args.pairwise_min_delta * coverage_scale,
                args.pairwise_temperature,
            )
            loss = float(args.lambda_oracle_value) * value_loss + float(args.lambda_oracle_rank) * rank_loss
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
            "train_loss": totals["loss"] / denom,
            "train_value_loss": totals["value"] / denom,
            "train_rank_loss": totals["rank"] / denom,
            "train_groups": len(groups),
            "train_pairs": int(totals["pairs"]),
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True))

    save_checkpoint(out_dir / "candidate.pt", model, config, args.checkpoint, feature_dim, relation_dim)
    write_history(out_dir / "history.tsv", history)
    handoff = {
        "mode": "fix",
        "status": "progressed",
        "objective": "oracle action-value finetune",
        "source_checkpoint": args.checkpoint,
        "source_checkpoint_id": checkpoint_id(args.checkpoint),
        "oracle_actions": args.oracle_actions,
        "out_dir": str(out_dir),
        "trainable_prefixes": trainable_prefixes,
        "groups": len(groups),
        "epochs": int(args.epochs),
        "outputs": {
            "candidate": str(out_dir / "candidate.pt"),
            "history": str(out_dir / "history.tsv"),
        },
        "final": history[-1] if history else {},
    }
    write_json(out_dir / "handoff.json", handoff)
    print(json.dumps(handoff, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
