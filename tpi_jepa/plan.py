"""Greedy planning with the trained minimal TPI-JEPA model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import torch

from .bench import parse_bench
from .features import (
    ACTION_TYPES,
    SCOAP_END,
    SCOAP_START,
    action_type_to_id,
    make_action_relation_features,
    make_base_node_features,
    make_state_features,
)
from .graph import GATE_TYPES, GraphData, build_graph
from .labels import find_bench_path
from .model import TPIWorldModel


LUT_TEMP_MARKER = "__lut_"
PLAN_FIELDNAMES = [
    "step",
    "node",
    "type",
    "q_pred",
    "q_pred_mean",
    "q_pred_std",
    "q_pred_lcb",
    "q_pred_context",
    "score_pred",
    "score_pred_mean",
    "score_pred_std",
    "score_pred_lcb",
    "reward_pred",
    "reward_pred_mean",
    "reward_pred_std",
    "reward_pred_lcb",
    "reward_pred_context",
    "fc_pred",
    "pattern_pred",
    "return_pred",
    "return_pred_mean",
    "return_pred_std",
    "return_pred_lcb",
    "guarded_reward",
    "guarded_reward_mean",
    "guarded_reward_std",
    "guarded_reward_lcb",
    "guarded_reward_context",
    "hard_reduction_total_pred",
    "hard_reduction_total_pred_mean",
    "hard_reduction_total_pred_std",
    "hard_reduction_total_pred_lcb",
    "hard_reduction_sa0_pred",
    "hard_reduction_sa1_pred",
    "derived_hard_count_pre_total_pred",
    "derived_hard_count_pre_sa0_pred",
    "derived_hard_count_pre_sa1_pred",
    "derived_hard_count_post_total_pred",
    "derived_hard_count_post_sa0_pred",
    "derived_hard_count_post_sa1_pred",
    "derived_hard_reduction_total_pred",
    "derived_hard_reduction_sa0_pred",
    "derived_hard_reduction_sa1_pred",
    "derived_hard_reduction_hybrid_pred",
    "derived_hard_reduction_hybrid_pred_mean",
    "derived_hard_reduction_hybrid_pred_std",
    "derived_hard_reduction_hybrid_pred_lcb",
    "hybrid_pred",
    "hybrid_pred_mean",
    "hybrid_pred_std",
    "hybrid_pred_lcb",
    "hybrid_pred_context",
    "bounded_residual_hybrid_pred",
    "bounded_residual_hybrid_pred_mean",
    "bounded_residual_hybrid_pred_std",
    "bounded_residual_hybrid_pred_lcb",
    "bounded_residual_hybrid_pred_context",
    "step_value",
    "sequence_score",
    "lookahead_score",
    "objective_score",
    "objective",
    "planner",
    "score_adjusted",
    "diversity_penalty",
    "candidate_strategy",
]

STRUCTURAL_START = len(GATE_TYPES)
REGION_START = SCOAP_END
_REAL_FAULT_BENCHMARK_ID: str | None = None
_REAL_FAULT_PRIOR_PATH: str | None = None
_ACTIVATION_PRIOR_PATH: str | None = None
_CANDIDATE_CACHE: dict[
    tuple[int, str, str | None, str | None, str | None],
    list[tuple[str, str, float]],
] = {}
_NODE_ID_CACHE: dict[int, dict[str, int]] = {}
_RELATION_CACHE: dict[tuple[int, int, str, int], torch.Tensor] = {}
_HARD_CONE_CACHE: dict[tuple[int, str | None, str | None, str | None, int], dict[str, torch.Tensor]] = {}
_HARD_CLUSTER_CACHE: dict[
    tuple[int, str | None, str | None, str | None, int, int, int],
    dict,
] = {}
_HARD_CLUSTER_MANAGER_CACHE: dict[
    tuple[int, str | None, str | None, str | None],
    "HardClusterCandidateManager",
] = {}
_NEIGHBORHOOD_CACHE: dict[tuple[int, int, int], set[int]] = {}
HEURISTIC_RECALL_POOL = [
    ("hard_fault_cone", 0.50),
    ("ffr", 0.20),
    ("reconvergence", 0.15),
    ("testability", 0.10),
    ("diversity", 0.05),
]


def clear_planner_caches() -> None:
    """Clear per-graph planner caches before long multi-circuit batches."""

    _NODE_ID_CACHE.clear()
    _RELATION_CACHE.clear()
    _HARD_CONE_CACHE.clear()
    _HARD_CLUSTER_CACHE.clear()
    _HARD_CLUSTER_MANAGER_CACHE.clear()
    _NEIGHBORHOOD_CACHE.clear()


@dataclass
class BeamPath:
    """One candidate action suffix explored from the current latent state."""

    selected: list[tuple[str, str]]
    z_state: torch.Tensor
    rows: list[dict]
    values: list[float]
    sequence_score: float
    objective_score: float


def set_real_fault_context(
    benchmark_id: str | None,
    prior_path: str | Path | None,
    activation_prior_path: str | Path | None = None,
) -> None:
    """Set optional real-fault priors used by planner feature builders and candidate ranking."""

    global _REAL_FAULT_BENCHMARK_ID, _REAL_FAULT_PRIOR_PATH, _ACTIVATION_PRIOR_PATH
    _REAL_FAULT_BENCHMARK_ID = benchmark_id
    _REAL_FAULT_PRIOR_PATH = str(prior_path) if prior_path else None
    _ACTIVATION_PRIOR_PATH = str(activation_prior_path) if activation_prior_path else None


def sequence_objective(values: list[float], objective: str, discount_gamma: float = 1.0) -> float:
    """Score one imagined action sequence."""

    if not values:
        return float("-inf")
    if objective == "terminal":
        return values[-1]
    if objective == "mean":
        return sum(values) / len(values)
    if objective == "discounted":
        return sum((discount_gamma**idx) * value for idx, value in enumerate(values))
    return sum(values)


def is_internal_lut_node(name: str) -> bool:
    """Return true for parser-synthesized LUT helper nodes."""

    return LUT_TEMP_MARKER in name


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[TPIWorldModel, dict]:
    """Load a trained checkpoint and return model plus config."""

    ckpt = torch.load(Path(path), map_location=device)
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
        q_head_type=str(config.get("q_head_type", "summary")),
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.coverage_scale = float(config.get("coverage_scale", 100.0))
    model.bounded_residual_alpha = float(config.get("bounded_residual_alpha", 1.0))
    model.bounded_residual_alpha_bound = float(config.get("bounded_residual_alpha_bound", 0.25))
    model.eval()
    return model, config


def enumerate_candidates(
    graph: GraphData,
    inserted_actions: list[tuple[str, str]],
    max_candidates: int | None = 8,
    strategy: str = "testability",
    real_fault_benchmark_id: str | None = None,
    real_fault_prior_path: str | Path | None = None,
    activation_prior_path: str | Path | None = None,
    candidate_cache_dir: str | Path | None = None,
    candidate_sample_seed: int = 0,
) -> list[tuple[str, str]]:
    """List candidate `(node, action_type)` pairs not already selected."""

    used = set(inserted_actions)
    strategy = (strategy or "testability").lower()
    if strategy.startswith("cached_"):
        return _cached_candidate_slice(
            graph,
            used,
            max_candidates,
            strategy,
            real_fault_benchmark_id or _REAL_FAULT_BENCHMARK_ID,
            candidate_cache_dir,
            candidate_sample_seed,
            len(inserted_actions),
        )
    if strategy in {"recall_pool", "heuristic_recall_pool"}:
        return _recall_pool_candidate_slice(
            graph,
            used,
            max_candidates,
            real_fault_benchmark_id,
            real_fault_prior_path,
            activation_prior_path,
        )
    if strategy == "hard_fault_recall_union":
        return _hard_fault_recall_union_candidate_slice(
            graph,
            used,
            max_candidates,
            real_fault_benchmark_id,
            real_fault_prior_path,
            activation_prior_path,
        )
    if strategy.endswith("_ranked"):
        base_strategy = strategy[: -len("_ranked")] or "testability"
        ranked = _ranked_candidates(
            graph,
            base_strategy,
            real_fault_benchmark_id,
            real_fault_prior_path,
            activation_prior_path,
        )
        available = [(node, action_type) for node, action_type, _ in ranked if (node, action_type) not in used]
        return available if max_candidates is None else available[: int(max_candidates)]
    if strategy != "netlist":
        ranked = _ranked_candidates(
            graph,
            strategy,
            real_fault_benchmark_id,
            real_fault_prior_path,
            activation_prior_path,
        )
        available = [(node, action_type, score) for node, action_type, score in ranked if (node, action_type) not in used]
        if max_candidates is None:
            return [(node, action_type) for node, action_type, _ in available]
        if strategy == "hard_fault_cluster":
            return _hard_fault_cluster_candidate_slice(
                graph,
                available,
                used,
                max_candidates,
                real_fault_benchmark_id,
                real_fault_prior_path,
                activation_prior_path,
            )
        if strategy == "ffr_hier":
            return _ffr_hier_candidate_slice(graph, available, max_candidates)
        return _balanced_candidate_slice(graph, available, max_candidates)

    candidates: list[tuple[str, str]] = []
    for idx, name in enumerate(graph.node_names):
        if bool(graph.input_mask[idx].item()) or is_internal_lut_node(name):
            continue
        for action_type in ACTION_TYPES:
            candidate = (name, action_type)
            if candidate in used:
                continue
            candidates.append(candidate)
            if max_candidates is not None and len(candidates) >= max_candidates:
                return candidates
    return candidates


def _recall_pool_candidate_slice(
    graph: GraphData,
    used: set[tuple[str, str]],
    max_candidates: int | None,
    real_fault_benchmark_id: str | None = None,
    real_fault_prior_path: str | Path | None = None,
    activation_prior_path: str | Path | None = None,
) -> list[tuple[str, str]]:
    """Build the fixed 50/20/15/10/5 heuristic recall pool."""

    limit = None if max_candidates is None else int(max_candidates)
    strategies = [name for name, _ in HEURISTIC_RECALL_POOL]
    ranked_lists: dict[str, list[tuple[str, str]]] = {}
    for base_strategy in strategies:
        if base_strategy == "diversity":
            ranked_lists[base_strategy] = _diversity_candidate_list(
                graph,
                used,
                limit,
                real_fault_benchmark_id,
                real_fault_prior_path,
                activation_prior_path,
            )
            continue
        ranked = _ranked_candidates(
            graph,
            base_strategy,
            real_fault_benchmark_id,
            real_fault_prior_path,
            activation_prior_path,
        )
        ranked_lists[base_strategy] = [
            (node, action_type) for node, action_type, _ in ranked if (node, action_type) not in used
        ]

    selected: list[tuple[str, str]] = []
    selected_set: set[tuple[str, str]] = set()
    if limit is not None:
        counts = _weighted_bucket_counts(limit, HEURISTIC_RECALL_POOL)
        for base_strategy, _ in HEURISTIC_RECALL_POOL:
            for candidate in ranked_lists[base_strategy][: counts[base_strategy]]:
                if candidate in selected_set:
                    continue
                selected.append(candidate)
                selected_set.add(candidate)
                if len(selected) >= limit:
                    return selected

    cursor = 0
    while limit is None or len(selected) < limit:
        progressed = False
        for base_strategy, _ in HEURISTIC_RECALL_POOL:
            ranked = ranked_lists[base_strategy]
            if cursor >= len(ranked):
                continue
            candidate = ranked[cursor]
            if candidate in selected_set:
                continue
            selected.append(candidate)
            selected_set.add(candidate)
            progressed = True
            if limit is not None and len(selected) >= limit:
                break
        if not progressed and all(cursor >= len(ranked) for ranked in ranked_lists.values()):
            break
        cursor += 1
    return selected


def _weighted_bucket_counts(limit: int, weights: list[tuple[str, float]]) -> dict[str, int]:
    """Allocate an exact candidate budget across weighted heuristic sources."""

    raw = [(name, max(0.0, float(weight)) * limit) for name, weight in weights]
    counts = {name: int(value) for name, value in raw}
    remaining = max(0, limit - sum(counts.values()))
    for name, _ in sorted(raw, key=lambda item: item[1] - int(item[1]), reverse=True)[:remaining]:
        counts[name] += 1
    return counts


def _diversity_candidate_list(
    graph: GraphData,
    used: set[tuple[str, str]],
    max_candidates: int | None,
    real_fault_benchmark_id: str | None,
    real_fault_prior_path: str | Path | None,
    activation_prior_path: str | Path | None,
) -> list[tuple[str, str]]:
    """Return a stable spread-out fallback list from the mixed heuristic ranking."""

    ranked = _ranked_candidates(
        graph,
        "mixed",
        real_fault_benchmark_id,
        real_fault_prior_path,
        activation_prior_path,
    )
    node_to_id = _node_id_map(graph)
    selected: list[tuple[str, str]] = []
    selected_set: set[tuple[str, str]] = set()
    selected_nodes: list[int] = []
    limit = None if max_candidates is None else max(1, int(max_candidates))
    for min_distance in (8, 5, 3, 0):
        for node, action_type, _ in ranked:
            candidate = (node, action_type)
            if candidate in used or candidate in selected_set:
                continue
            node_id = node_to_id.get(node)
            if node_id is None:
                continue
            if not _is_far_enough(graph, node_id, selected_nodes, min_distance):
                continue
            selected.append(candidate)
            selected_set.add(candidate)
            selected_nodes.append(node_id)
            if limit is not None and len(selected) >= limit:
                return selected
        if len(selected) >= len(ranked):
            break
    return selected


def _hard_fault_recall_union_candidate_slice(
    graph: GraphData,
    used: set[tuple[str, str]],
    max_candidates: int | None,
    real_fault_benchmark_id: str | None = None,
    real_fault_prior_path: str | Path | None = None,
    activation_prior_path: str | Path | None = None,
) -> list[tuple[str, str]]:
    """Merge Top-N hard-fault ranking with Top-N recall-pool actions before model scoring."""

    hard_ranked = _ranked_candidates(
        graph,
        "hard_fault",
        real_fault_benchmark_id,
        real_fault_prior_path,
        activation_prior_path,
    )
    branch_limit = None if max_candidates is None else int(max_candidates)
    hard_candidates = [(node, action_type) for node, action_type, _ in hard_ranked if (node, action_type) not in used]
    if branch_limit is not None:
        hard_candidates = hard_candidates[:branch_limit]
    recall_candidates = _recall_pool_candidate_slice(
        graph,
        used,
        branch_limit,
        real_fault_benchmark_id,
        real_fault_prior_path,
        activation_prior_path,
    )
    selected: list[tuple[str, str]] = []
    selected_set: set[tuple[str, str]] = set()
    cursor = 0
    while True:
        progressed = False
        for candidates in (hard_candidates, recall_candidates):
            if cursor >= len(candidates):
                continue
            candidate = candidates[cursor]
            if candidate in selected_set:
                continue
            selected.append(candidate)
            selected_set.add(candidate)
            progressed = True
        if not progressed and all(cursor >= len(candidates) for candidates in (hard_candidates, recall_candidates)):
            break
        cursor += 1
    return selected


def _load_candidate_cache(
    benchmark_id: str | None,
    candidate_cache_dir: str | Path | None,
) -> list[str]:
    """Load cached backend-valid insertable TP net names for one benchmark."""

    return [str(item["net"]) for item in _load_candidate_cache_items(benchmark_id, candidate_cache_dir) if item.get("net")]


def _load_candidate_cache_items(
    benchmark_id: str | None,
    candidate_cache_dir: str | Path | None,
) -> list[dict]:
    """Load cached backend-valid insertable TP candidate records for one benchmark."""

    if not benchmark_id:
        raise ValueError("cached candidate strategies require benchmark_id context")
    if not candidate_cache_dir:
        raise ValueError("cached candidate strategies require --candidate-cache-dir")
    path = Path(candidate_cache_dir) / f"{benchmark_id}.json"
    payload = json.loads(path.read_text())
    items = [item for item in payload.get("candidates", []) if item.get("net")]
    if not items:
        raise ValueError(f"candidate cache is empty: {path}")
    return items


def _stable_seed(*parts: object) -> int:
    """Return a reproducible integer seed from structured parts."""

    text = "::".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def _stride_sample(items: list[str], count: int) -> list[str]:
    """Select `count` items spread across the full input order."""

    if count >= len(items):
        return list(items)
    if count <= 0:
        return []
    if count == 1:
        return [items[0]]
    last = len(items) - 1
    indices = sorted({round(index * last / (count - 1)) for index in range(count)})
    selected = [items[index] for index in indices]
    cursor = 0
    while len(selected) < count and cursor < len(items):
        item = items[cursor]
        if item not in selected:
            selected.append(item)
        cursor += 1
    return selected[:count]


def _cached_candidate_slice(
    graph: GraphData,
    used: set[tuple[str, str]],
    max_candidates: int | None,
    strategy: str,
    benchmark_id: str | None,
    candidate_cache_dir: str | Path | None,
    candidate_sample_seed: int,
    selection_step: int,
) -> list[tuple[str, str]]:
    """Sample candidate actions from cached backend-valid insertable TP nets."""

    node_to_id = _node_id_map(graph)
    if strategy == "cached_hard_cone":
        return _cached_hard_cone_candidate_slice(
            graph,
            used,
            max_candidates,
            benchmark_id,
            candidate_cache_dir,
        )
    cache_nets = _load_candidate_cache(benchmark_id, candidate_cache_dir)
    valid_nets = [net for net in cache_nets if net in node_to_id and not is_internal_lut_node(net)]
    if not valid_nets:
        raise ValueError(f"no cached candidate nets are present in graph for benchmark_id={benchmark_id!r}")
    if max_candidates is None:
        net_count = len(valid_nets)
    else:
        net_count = max(1, (int(max_candidates) + len(ACTION_TYPES) - 1) // len(ACTION_TYPES))

    if strategy == "cached_stride":
        selected_nets = _stride_sample(valid_nets, net_count)
    elif strategy == "cached_random":
        import random

        seed = _stable_seed(benchmark_id, candidate_sample_seed, selection_step, len(valid_nets))
        generator = random.Random(seed)
        if net_count >= len(valid_nets):
            selected_nets = list(valid_nets)
            generator.shuffle(selected_nets)
        else:
            selected_nets = generator.sample(valid_nets, net_count)
    else:
        selected_nets = valid_nets[:net_count]

    candidates: list[tuple[str, str]] = []
    for net in selected_nets:
        for action_type in ACTION_TYPES:
            candidate = (net, action_type)
            if candidate in used:
                continue
            candidates.append(candidate)
            if max_candidates is not None and len(candidates) >= int(max_candidates):
                return candidates
    return candidates


def _cached_hard_cone_candidate_slice(
    graph: GraphData,
    used: set[tuple[str, str]],
    max_candidates: int | None,
    benchmark_id: str | None,
    candidate_cache_dir: str | Path | None,
) -> list[tuple[str, str]]:
    """Select actions from a cache pre-ranked by current hard-fault cone evidence."""

    node_to_id = _node_id_map(graph)
    items = _load_candidate_cache_items(benchmark_id, candidate_cache_dir)
    ranked: list[tuple[str, str, float]] = []
    for item in items:
        net = str(item.get("net") or "")
        if net not in node_to_id or is_internal_lut_node(net):
            continue
        action_scores = item.get("action_scores")
        if not isinstance(action_scores, dict):
            base_score = float(item.get("score") or item.get("priority") or 0.0)
            action_scores = {action_type: base_score for action_type in ACTION_TYPES}
        for action_type in ACTION_TYPES:
            candidate = (net, action_type)
            if candidate in used:
                continue
            ranked.append((net, action_type, float(action_scores.get(action_type, 0.0))))
    ranked.sort(key=lambda item: item[2], reverse=True)
    if max_candidates is None:
        return [(node, action_type) for node, action_type, _ in ranked]
    return _balanced_candidate_slice(graph, ranked, int(max_candidates))


def _node_id_map(graph: GraphData) -> dict[str, int]:
    """Return a cached mapping from node name to integer id."""

    key = id(graph)
    mapping = _NODE_ID_CACHE.get(key)
    if mapping is None:
        mapping = {name: idx for idx, name in enumerate(graph.node_names)}
        _NODE_ID_CACHE[key] = mapping
    return mapping


def _cached_relation_features(
    graph: GraphData,
    action_node_id: int,
    relation_mode: str,
    relation_depth: int,
    device: torch.device,
) -> torch.Tensor:
    """Reuse action relation features across beam branches."""

    mode = (relation_mode or "basic").lower()
    key = (id(graph), action_node_id, mode, int(relation_depth))
    relation = _RELATION_CACHE.get(key)
    if relation is None:
        relation = make_action_relation_features(graph, action_node_id, mode, relation_depth)
        _RELATION_CACHE[key] = relation
    return relation.to(device)


def _ranked_candidates(
    graph: GraphData,
    strategy: str,
    real_fault_benchmark_id: str | None = None,
    real_fault_prior_path: str | Path | None = None,
    activation_prior_path: str | Path | None = None,
) -> list[tuple[str, str, float]]:
    """Return all candidate actions sorted by a cheap testability heuristic."""

    benchmark_id = real_fault_benchmark_id or _REAL_FAULT_BENCHMARK_ID
    prior_path = str(real_fault_prior_path) if real_fault_prior_path else _REAL_FAULT_PRIOR_PATH
    act_path = str(activation_prior_path) if activation_prior_path else _ACTIVATION_PRIOR_PATH
    key = (id(graph), strategy, benchmark_id, prior_path, act_path)
    if key in _CANDIDATE_CACHE:
        return _CANDIDATE_CACHE[key]
    x = make_base_node_features(
        graph,
        "full",
        benchmark_id=benchmark_id,
        real_fault_prior_path=prior_path,
        activation_prior_path=act_path,
    )
    structural = x[:, STRUCTURAL_START:SCOAP_START]
    scoap = x[:, SCOAP_START:SCOAP_END]
    region = x[:, REGION_START:] if x.shape[1] > REGION_START else torch.zeros((graph.num_nodes, 8))
    cc0 = scoap[:, 0]
    cc1 = scoap[:, 1]
    co = scoap[:, 2]
    fanin = structural[:, 0]
    fanout = structural[:, 1]
    hard_control = torch.maximum(cc0, cc1)
    hard_fault = region[:, 3] if region.shape[1] > 3 else (hard_control + co) * 0.5
    extra_start = 8
    if prior_path and region.shape[1] >= extra_start + 3:
        hard_fault = torch.maximum(hard_fault, region[:, extra_start])
        extra_start += 3
    if act_path and region.shape[1] >= extra_start + 3:
        activation_min = region[:, extra_start + 1]
        activation_hardness = 1.0 - activation_min
        hard_fault = torch.maximum(hard_fault, activation_hardness)
    reconvergence = region[:, 4] if region.shape[1] > 4 else fanout
    ffr_span = region[:, 5] if region.shape[1] > 5 else torch.zeros_like(co)
    transparent = region[:, 6] if region.shape[1] > 6 else torch.zeros_like(co)
    cone = region[:, 7] if region.shape[1] > 7 else torch.zeros_like(co)
    cone_scores = None
    if strategy in {"hard_fault_cone", "hard_fault_cluster"}:
        cone_scores = _hard_fault_cone_scores(
            graph,
            x,
            benchmark_id=benchmark_id,
            real_fault_prior_path=prior_path,
            activation_prior_path=act_path,
        )

    ranked: list[tuple[str, str, float]] = []
    for idx, name in enumerate(graph.node_names):
        if bool(graph.input_mask[idx].item()) or is_internal_lut_node(name):
            continue
        base_bonus = 0.08 * float(fanout[idx].item()) + 0.05 * float(fanin[idx].item())
        transparent_penalty = 0.10 * float(transparent[idx].item())
        for action_type in ACTION_TYPES:
            if strategy in {"hard_fault_cone", "hard_fault_cluster"}:
                assert cone_scores is not None
                action_score = float(cone_scores[action_type][idx].item())
                if action_type == "control0":
                    action_score += 0.25 * float(cc0[idx].item())
                elif action_type == "control1":
                    action_score += 0.25 * float(cc1[idx].item())
                else:
                    action_score += 0.25 * float(co[idx].item())
                score = action_score + base_bonus - transparent_penalty
            elif strategy == "hard_fault":
                if action_type == "control0":
                    action_score = 0.75 * float(cc0[idx].item()) + 0.35 * float(hard_fault[idx].item())
                elif action_type == "control1":
                    action_score = 0.75 * float(cc1[idx].item()) + 0.35 * float(hard_fault[idx].item())
                else:
                    action_score = 0.80 * float(co[idx].item()) + 0.45 * float(hard_fault[idx].item())
                score = action_score + 0.20 * float(cone[idx].item()) + base_bonus - transparent_penalty
            elif strategy == "reconvergence":
                if action_type == "observe":
                    action_score = 1.05 * float(reconvergence[idx].item()) + 0.75 * float(co[idx].item())
                elif action_type == "control0":
                    action_score = 0.65 * float(reconvergence[idx].item()) + 0.45 * float(cc0[idx].item())
                else:
                    action_score = 0.65 * float(reconvergence[idx].item()) + 0.45 * float(cc1[idx].item())
                score = action_score + 0.25 * float(hard_fault[idx].item()) + base_bonus - transparent_penalty
            elif strategy in {"ffr", "ffr_hier"}:
                if action_type == "observe":
                    action_score = 0.85 * float(ffr_span[idx].item()) + 0.65 * float(co[idx].item())
                elif action_type == "control0":
                    action_score = 0.80 * float(ffr_span[idx].item()) + 0.55 * float(cc0[idx].item())
                else:
                    action_score = 0.80 * float(ffr_span[idx].item()) + 0.55 * float(cc1[idx].item())
                score = action_score + 0.25 * float(hard_fault[idx].item()) + base_bonus - transparent_penalty
            else:
                if action_type == "control0":
                    action_score = 1.15 * float(cc0[idx].item()) + 0.35 * float(hard_fault[idx].item())
                elif action_type == "control1":
                    action_score = 1.15 * float(cc1[idx].item()) + 0.35 * float(hard_fault[idx].item())
                else:
                    action_score = 1.15 * float(co[idx].item()) + 0.35 * float(reconvergence[idx].item())
                if strategy == "mixed":
                    action_score += 0.25 * float(ffr_span[idx].item()) + 0.25 * float(cone[idx].item())
                score = action_score + base_bonus - transparent_penalty
            ranked.append((name, action_type, float(score)))
    ranked.sort(key=lambda item: item[2], reverse=True)
    _CANDIDATE_CACHE[key] = ranked
    return ranked


def _hard_fault_cone_scores(
    graph: GraphData,
    x: torch.Tensor,
    *,
    benchmark_id: str | None,
    real_fault_prior_path: str | None,
    activation_prior_path: str | None,
    max_depth: int = 10,
    max_hard_nodes: int = 256,
) -> dict[str, torch.Tensor]:
    """Score candidates by explicit hard-fault cones, paths, bottlenecks, and shared coverage."""

    key = (id(graph), benchmark_id, real_fault_prior_path, activation_prior_path, int(max_depth), int(max_hard_nodes))
    cached = _HARD_CONE_CACHE.get(key)
    if cached is not None:
        return cached

    structural = x[:, STRUCTURAL_START:SCOAP_START]
    scoap = x[:, SCOAP_START:SCOAP_END]
    region = x[:, REGION_START:] if x.shape[1] > REGION_START else torch.zeros((graph.num_nodes, 8))
    cc0 = scoap[:, 0]
    cc1 = scoap[:, 1]
    co = scoap[:, 2]
    fanout = structural[:, 1]
    hard_proxy = region[:, 3] if region.shape[1] > 3 else (torch.maximum(cc0, cc1) + co) * 0.5

    comp = _hard_feature_components(graph, x, real_fault_prior_path, activation_prior_path)
    activation_hard = comp["activation_hard"]
    hard_weight = _hard_weight_from_features(graph, x, real_fault_prior_path, activation_prior_path)
    hard_nodes = _select_hard_seed_nodes(
        hard_weight,
        graph_nodes=graph.num_nodes,
        max_hard_nodes=max_hard_nodes,
    )

    upstream_sum = torch.zeros(graph.num_nodes, dtype=torch.float32)
    downstream_sum = torch.zeros(graph.num_nodes, dtype=torch.float32)
    upstream_count = torch.zeros(graph.num_nodes, dtype=torch.float32)
    downstream_count = torch.zeros(graph.num_nodes, dtype=torch.float32)
    bottleneck_sum = torch.zeros(graph.num_nodes, dtype=torch.float32)
    activation_sum = torch.zeros(graph.num_nodes, dtype=torch.float32)
    propagation_sum = torch.zeros(graph.num_nodes, dtype=torch.float32)

    output_nodes = [idx for idx, is_output in enumerate(graph.output_mask.tolist()) if is_output]
    output_set = set(output_nodes)

    for hard in hard_nodes:
        weight = float(hard_weight[hard].item())
        if weight <= 0.0:
            continue
        hard_observe = float(co[hard].item())
        hard_activation = float(activation_hard[hard].item())

        fanin_dist = _distance_map(hard, graph.fanin_lists, max_depth)
        for node, dist in fanin_dist.items():
            if node == hard:
                continue
            decay = 1.0 / float(1 + dist)
            value = weight * decay
            downstream_sum[node] += value
            downstream_count[node] += 1.0
            activation_sum[node] += value * (0.5 + hard_activation)

        fanout_dist = _distance_map(hard, graph.fanout_lists, max_depth)
        reaches_output = bool(output_set & set(fanout_dist))
        for node, dist in fanout_dist.items():
            if node == hard:
                continue
            decay = 1.0 / float(1 + dist)
            value = weight * decay
            upstream_sum[node] += value
            upstream_count[node] += 1.0
            bottleneck_sum[node] += value * (0.5 * hard_observe + 0.5 * float(co[node].item()))
            if reaches_output:
                propagation_sum[node] += value

    def normalize(v: torch.Tensor) -> torch.Tensor:
        return v / v.max().clamp_min(1.0)

    upstream_sum = normalize(upstream_sum)
    downstream_sum = normalize(downstream_sum)
    upstream_count = normalize(torch.log1p(upstream_count))
    downstream_count = normalize(torch.log1p(downstream_count))
    bottleneck_sum = normalize(bottleneck_sum)
    activation_sum = normalize(activation_sum)
    propagation_sum = normalize(propagation_sum)
    fanout = normalize(fanout)

    observe = (
        1.15 * upstream_sum
        + 0.80 * bottleneck_sum
        + 0.55 * upstream_count
        + 0.35 * propagation_sum
        + 0.20 * co
    )
    control0 = (
        1.05 * downstream_sum
        + 0.80 * activation_sum
        + 0.45 * downstream_count
        + 0.25 * hard_weight
        + 0.20 * cc0
        + 0.10 * fanout
    )
    control1 = (
        1.05 * downstream_sum
        + 0.80 * activation_sum
        + 0.45 * downstream_count
        + 0.25 * hard_weight
        + 0.20 * cc1
        + 0.10 * fanout
    )
    scores = {
        "control0": normalize(control0),
        "control1": normalize(control1),
        "observe": normalize(observe),
    }
    _HARD_CONE_CACHE[key] = scores
    return scores


def _distance_map(start: int, adjacency: list[list[int]], max_depth: int) -> dict[int, int]:
    """Return local graph distances from one source along one direction."""

    dist = {start: 0}
    queue: deque[int] = deque([start])
    while queue:
        node = queue.popleft()
        if dist[node] >= max_depth:
            continue
        for nxt in adjacency[node]:
            if nxt in dist:
                continue
            dist[nxt] = dist[node] + 1
            queue.append(nxt)
    return dist


def _undirected_distance_map(graph: GraphData, start: int, max_depth: int) -> dict[int, int]:
    """Return local undirected graph distances from one source."""

    dist = {start: 0}
    queue: deque[int] = deque([start])
    while queue:
        node = queue.popleft()
        if dist[node] >= max_depth:
            continue
        for nxt in graph.fanin_lists[node] + graph.fanout_lists[node]:
            if nxt in dist:
                continue
            dist[nxt] = dist[node] + 1
            queue.append(nxt)
    return dist


def _local_distance_between(graph: GraphData, lhs: int, rhs: int, max_depth: int = 4) -> int | None:
    """Return an undirected local distance between two nodes if it is small."""

    if lhs == rhs:
        return 0
    queue: deque[tuple[int, int]] = deque([(lhs, 0)])
    seen = {lhs}
    while queue:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for nxt in graph.fanin_lists[node] + graph.fanout_lists[node]:
            if nxt in seen:
                continue
            if nxt == rhs:
                return depth + 1
            seen.add(nxt)
            queue.append((nxt, depth + 1))
    return None


def _hard_feature_components(
    graph: GraphData,
    x: torch.Tensor,
    real_fault_prior_path: str | None,
    activation_prior_path: str | None,
) -> dict[str, torch.Tensor]:
    """Return normalized hard-fault proxy components aligned to graph nodes."""

    scoap = x[:, SCOAP_START:SCOAP_END]
    region = x[:, REGION_START:] if x.shape[1] > REGION_START else torch.zeros((graph.num_nodes, 8))
    cc0 = scoap[:, 0]
    cc1 = scoap[:, 1]
    co = scoap[:, 2]
    hard_control = torch.maximum(cc0, cc1)
    hard_proxy = region[:, 3] if region.shape[1] > 3 else (hard_control + co) * 0.5
    reconvergence = region[:, 4] if region.shape[1] > 4 else torch.zeros_like(co)
    cone_pressure = region[:, 7] if region.shape[1] > 7 else torch.zeros_like(co)

    extra_start = 8
    real_hard = torch.zeros(graph.num_nodes, dtype=torch.float32)
    if real_fault_prior_path and region.shape[1] >= extra_start + 3:
        real_hard = torch.maximum(region[:, extra_start], region[:, extra_start + 1])
        extra_start += 3

    activation_hard = hard_control
    if activation_prior_path and region.shape[1] >= extra_start + 3:
        activation_hard = torch.maximum(activation_hard, 1.0 - region[:, extra_start + 1])

    return {
        "cc0": cc0,
        "cc1": cc1,
        "co": co,
        "hard_proxy": hard_proxy,
        "reconvergence": reconvergence,
        "cone_pressure": cone_pressure,
        "real_hard": real_hard,
        "activation_hard": activation_hard,
    }


def _normalize_tensor(values: torch.Tensor) -> torch.Tensor:
    """Normalize a non-negative tensor without changing all-zero tensors."""

    return values / values.max().clamp_min(1.0)


def _select_hard_seed_nodes(
    hard_weight: torch.Tensor,
    *,
    graph_nodes: int,
    max_hard_nodes: int,
    min_hard_nodes: int = 64,
) -> list[int]:
    """Select hard seeds by topK/percentile instead of a fixed absolute threshold."""

    positive = torch.nonzero(hard_weight > 0.0, as_tuple=False).flatten()
    if positive.numel() == 0:
        return torch.topk(hard_weight, k=min(16, graph_nodes)).indices.tolist()
    target = min(max_hard_nodes, max(min_hard_nodes, (graph_nodes + 99) // 100))
    target = min(int(target), int(positive.numel()))
    scores = hard_weight[positive]
    top_local = torch.topk(scores, k=max(1, target)).indices
    return positive[top_local].tolist()


def _hard_weight_from_features(
    graph: GraphData,
    x: torch.Tensor,
    real_fault_prior_path: str | None,
    activation_prior_path: str | None,
) -> torch.Tensor:
    """Return the normalized per-node hard-fault weight used for cone/cluster heuristics."""

    comp = _hard_feature_components(graph, x, real_fault_prior_path, activation_prior_path)
    composite_proxy = (
        0.45 * comp["hard_proxy"]
        + 0.30 * comp["activation_hard"]
        + 0.15 * comp["cone_pressure"]
        + 0.10 * comp["reconvergence"]
    )
    hard_weight = torch.maximum(comp["real_hard"], composite_proxy)
    return _normalize_tensor(hard_weight)


def _hard_fault_cluster_data(
    graph: GraphData,
    *,
    benchmark_id: str | None,
    real_fault_prior_path: str | None,
    activation_prior_path: str | None,
    max_depth: int = 10,
    cluster_depth: int = 4,
    max_hard_nodes: int = 512,
) -> dict:
    """Cluster hard-fault seeds and build per-cluster cone coverage scores."""

    key = (
        id(graph),
        benchmark_id,
        real_fault_prior_path,
        activation_prior_path,
        int(max_depth),
        int(cluster_depth),
        int(max_hard_nodes),
    )
    cached = _HARD_CLUSTER_CACHE.get(key)
    if cached is not None:
        return cached

    x = make_base_node_features(
        graph,
        "full",
        benchmark_id=benchmark_id,
        real_fault_prior_path=real_fault_prior_path,
        activation_prior_path=activation_prior_path,
    )
    hard_weight = _hard_weight_from_features(graph, x, real_fault_prior_path, activation_prior_path)
    comp = _hard_feature_components(graph, x, real_fault_prior_path, activation_prior_path)
    hard_nodes = _select_hard_seed_nodes(
        hard_weight,
        graph_nodes=graph.num_nodes,
        max_hard_nodes=max_hard_nodes,
    )

    hard_set = set(int(node) for node in hard_nodes)
    remaining = set(hard_set)
    clusters: list[list[int]] = []
    while remaining:
        seed = max(remaining, key=lambda node: float(hard_weight[node].item()))
        remaining.remove(seed)
        cluster = [seed]
        queue: deque[int] = deque([seed])
        while queue:
            node = queue.popleft()
            nearby = _undirected_distance_map(graph, node, cluster_depth)
            for other in list(remaining & set(nearby)):
                remaining.remove(other)
                cluster.append(other)
                queue.append(other)
        clusters.append(cluster)

    cluster_rows = []
    control_depth = min(int(max_depth), 6)
    observe_depth = min(int(max_depth), 8)
    for cluster_id, seeds in enumerate(clusters):
        score_by_type = {action_type: torch.zeros(graph.num_nodes, dtype=torch.float32) for action_type in ACTION_TYPES}
        count_by_type = {action_type: torch.zeros(graph.num_nodes, dtype=torch.float32) for action_type in ACTION_TYPES}
        type_mass = {action_type: 0.0 for action_type in ACTION_TYPES}
        mass = 0.0
        for hard in seeds:
            weight = float(hard_weight[hard].item())
            mass += weight
            control0_weight = weight * (0.55 + 0.45 * float(comp["cc0"][hard].item()))
            control1_weight = weight * (0.55 + 0.45 * float(comp["cc1"][hard].item()))
            observe_weight = weight * (
                0.50
                + 0.35 * float(comp["co"][hard].item())
                + 0.15 * float(comp["reconvergence"][hard].item())
            )
            seed_weights = {
                "control0": control0_weight,
                "control1": control1_weight,
                "observe": observe_weight,
            }
            for action_type, typed_weight in seed_weights.items():
                type_mass[action_type] += typed_weight
                score_by_type[action_type][hard] += typed_weight
                count_by_type[action_type][hard] += 1.0
            for node, dist in _distance_map(hard, graph.fanin_lists, control_depth).items():
                decay = 1.0 / float(1 + dist)
                for action_type in ("control0", "control1"):
                    value = seed_weights[action_type] * decay
                    score_by_type[action_type][node] += value
                    count_by_type[action_type][node] += 1.0
            for node, dist in _distance_map(hard, graph.fanout_lists, observe_depth).items():
                decay = 1.0 / float(1 + dist)
                value = seed_weights["observe"] * decay
                score_by_type["observe"][node] += value
                count_by_type["observe"][node] += 1.0
        for action_type in ACTION_TYPES:
            score = score_by_type[action_type]
            count = count_by_type[action_type]
            score = score / torch.sqrt(1.0 + count)
            score_by_type[action_type] = _normalize_tensor(score)
        combined_score = torch.maximum(
            score_by_type["observe"],
            torch.maximum(score_by_type["control0"], score_by_type["control1"]),
        )
        cluster_rows.append(
            {
                "id": cluster_id,
                "seeds": seeds,
                "mass": float(mass),
                "type_mass": type_mass,
                "score_by_type": score_by_type,
                "score": combined_score,
            }
        )
    cluster_rows.sort(key=lambda row: row["mass"], reverse=True)
    for rank, row in enumerate(cluster_rows):
        row["rank"] = rank

    payload = {"clusters": cluster_rows, "hard_weight": hard_weight, "components": comp}
    _HARD_CLUSTER_CACHE[key] = payload
    return payload


def _cluster_for_node(cluster_data: dict, node_id: int, action_type: str | None = None) -> tuple[int, float]:
    """Return the best hard-fault cluster assignment for a candidate node."""

    best_cluster = -1
    best_score = 0.0
    for cluster in cluster_data["clusters"]:
        if action_type and "score_by_type" in cluster:
            value = float(cluster["score_by_type"].get(action_type, cluster["score"])[node_id].item())
        else:
            value = float(cluster["score"][node_id].item())
        if value > best_score:
            best_cluster = int(cluster["id"])
            best_score = value
    return best_cluster, best_score


def _cluster_action_type_quota(
    max_candidates: int,
    clusters: list[dict],
    used_cluster_counts: dict[int, int],
    used_action_type_counts: dict[str, int],
) -> dict[str, int]:
    """Allocate action types from typed cluster mass with bounded diversity."""

    raw = {action_type: 0.0 for action_type in ACTION_TYPES}
    for cluster in clusters:
        cluster_id = int(cluster["id"])
        decay = float(1 + used_cluster_counts.get(cluster_id, 0)) ** 0.75
        type_mass = cluster.get("type_mass", {})
        for action_type in ACTION_TYPES:
            raw[action_type] += float(type_mass.get(action_type, 0.0)) / decay
    for action_type in ACTION_TYPES:
        raw[action_type] /= float(1 + used_action_type_counts.get(action_type, 0)) ** 0.75
    total = sum(raw.values())
    if total <= 0.0:
        weights = {action_type: 1.0 / len(ACTION_TYPES) for action_type in ACTION_TYPES}
    else:
        weights = {action_type: raw[action_type] / total for action_type in ACTION_TYPES}
    bounded = {action_type: min(0.50, max(0.20, weights[action_type])) for action_type in ACTION_TYPES}
    bounded_total = sum(bounded.values())
    if bounded_total > 0.0:
        bounded = {action_type: value / bounded_total for action_type, value in bounded.items()}
    return _weighted_bucket_counts(max_candidates, list(bounded.items()))


def _cached_local_neighborhood(graph: GraphData, node_id: int, max_depth: int) -> set[int]:
    """Return cached undirected neighborhood nodes for cheap near-duplicate checks."""

    key = (id(graph), int(node_id), int(max_depth))
    cached = _NEIGHBORHOOD_CACHE.get(key)
    if cached is not None:
        return cached
    nodes = set(_undirected_distance_map(graph, int(node_id), int(max_depth)).keys())
    nodes.discard(int(node_id))
    _NEIGHBORHOOD_CACHE[key] = nodes
    return nodes


class HardClusterCandidateManager:
    """Cached hard-cluster candidate manager with per-step lazy penalties."""

    def __init__(
        self,
        graph: GraphData,
        benchmark_id: str | None,
        real_fault_prior_path: str | None,
        activation_prior_path: str | None,
    ) -> None:
        self.graph = graph
        self.node_to_id = _node_id_map(graph)
        self.cluster_data = _hard_fault_cluster_data(
            graph,
            benchmark_id=benchmark_id,
            real_fault_prior_path=real_fault_prior_path,
            activation_prior_path=activation_prior_path,
        )
        self.clusters = self.cluster_data["clusters"]
        self.components = self.cluster_data.get("components", {})
        self.entries_by_type: dict[str, list[dict]] = {action_type: [] for action_type in ACTION_TYPES}

        x = make_base_node_features(
            graph,
            "full",
            benchmark_id=benchmark_id,
            real_fault_prior_path=real_fault_prior_path,
            activation_prior_path=activation_prior_path,
        )
        structural = x[:, STRUCTURAL_START:SCOAP_START]
        scoap = x[:, SCOAP_START:SCOAP_END]
        region = x[:, REGION_START:] if x.shape[1] > REGION_START else torch.zeros((graph.num_nodes, 8))
        fanin = structural[:, 0]
        fanout = structural[:, 1]
        transparent = region[:, 6] if region.shape[1] > 6 else torch.zeros(graph.num_nodes)
        base_bonus = 0.08 * fanout + 0.05 * fanin
        transparent_penalty = 0.10 * transparent
        cone_scores = _hard_fault_cone_scores(
            graph,
            x,
            benchmark_id=benchmark_id,
            real_fault_prior_path=real_fault_prior_path,
            activation_prior_path=activation_prior_path,
        )
        typed_cluster_scores = {action_type: torch.zeros(graph.num_nodes) for action_type in ACTION_TYPES}
        for cluster in self.clusters:
            score_by_type = cluster.get("score_by_type", {})
            for action_type in ACTION_TYPES:
                typed_cluster_scores[action_type] = torch.maximum(
                    typed_cluster_scores[action_type],
                    score_by_type.get(action_type, cluster["score"]),
                )

        invalid_mask = graph.input_mask.bool().clone()
        for idx, name in enumerate(graph.node_names):
            if is_internal_lut_node(name):
                invalid_mask[idx] = True

        pool_limit = max(256, int(os.environ.get("TPI_HARD_CLUSTER_POOL", "8192")))
        raw_entries: list[dict] = []
        zeros = torch.zeros(graph.num_nodes)
        for action_type in ACTION_TYPES:
            if action_type == "control0":
                testability = scoap[:, 0]
                sharing = self.components.get("cone_pressure", zeros)
            elif action_type == "control1":
                testability = scoap[:, 1]
                sharing = self.components.get("cone_pressure", zeros)
            else:
                testability = scoap[:, 2]
                sharing = self.components.get("reconvergence", zeros)
            base_tensor = (
                0.55 * (cone_scores[action_type] + 0.25 * testability + base_bonus - transparent_penalty)
                + 0.30 * typed_cluster_scores[action_type]
                + 0.10 * testability
                + 0.05 * sharing
            )
            base_tensor = base_tensor.masked_fill(invalid_mask, float("-inf"))
            k = min(pool_limit, graph.num_nodes)
            top = torch.topk(base_tensor, k=k).indices.tolist()
            for node_id in top:
                score = float(base_tensor[node_id].item())
                if score == float("-inf"):
                    continue
                cluster_id, _ = _cluster_for_node(self.cluster_data, int(node_id), action_type)
                raw_entries.append(
                    {
                        "node": graph.node_names[int(node_id)],
                        "type": action_type,
                        "node_id": int(node_id),
                        "cluster_id": cluster_id,
                        "base_score": score,
                    }
                )

        best_score_by_node: dict[int, float] = {}
        for entry in raw_entries:
            node_id = int(entry["node_id"])
            best_score_by_node[node_id] = max(best_score_by_node.get(node_id, float("-inf")), float(entry["base_score"]))
        for entry in raw_entries:
            node_id = int(entry["node_id"])
            action_type = str(entry["type"])
            best_for_node = best_score_by_node.get(node_id)
            if best_for_node is not None and best_for_node > 0.0 and float(entry["base_score"]) < 0.90 * best_for_node:
                continue
            self.entries_by_type[action_type].append(entry)
        for action_type in ACTION_TYPES:
            self.entries_by_type[action_type].sort(key=lambda item: item["base_score"], reverse=True)

    def select(self, inserted_actions: list[tuple[str, str]], max_candidates: int) -> list[tuple[str, str]]:
        """Return a lazy-rescored candidate pool for the current inserted actions."""

        if max_candidates <= 0:
            return []
        used = set(inserted_actions)
        used_node_ids: set[int] = set()
        used_cluster_counts: dict[int, int] = {}
        used_action_type_counts: dict[str, int] = {action_type: 0 for action_type in ACTION_TYPES}
        near_by_type: dict[str, set[int]] = {action_type: set() for action_type in ACTION_TYPES}
        near_any: set[int] = set()
        for node, action_type in inserted_actions:
            node_id = self.node_to_id.get(node)
            if node_id is None:
                continue
            used_node_ids.add(node_id)
            used_action_type_counts[action_type] = used_action_type_counts.get(action_type, 0) + 1
            cluster_id, _ = _cluster_for_node(self.cluster_data, node_id, action_type)
            if cluster_id >= 0:
                used_cluster_counts[cluster_id] = used_cluster_counts.get(cluster_id, 0) + 1
            same_type_neighbors = _cached_local_neighborhood(self.graph, node_id, 3)
            near_by_type.setdefault(action_type, set()).update(same_type_neighbors)
            near_any.update(_cached_local_neighborhood(self.graph, node_id, 2))

        type_quota = _cluster_action_type_quota(
            max_candidates,
            self.clusters,
            used_cluster_counts,
            used_action_type_counts,
        )
        action_type_order = sorted(
            ACTION_TYPES,
            key=lambda action_type: (
                used_action_type_counts.get(action_type, 0),
                -type_quota.get(action_type, 0),
                action_type,
            ),
        )

        pool: list[dict] = []
        scan_limit = max(256, int(max_candidates) * 16)
        for action_type in action_type_order:
            taken = 0
            for entry in self.entries_by_type[action_type]:
                candidate = (entry["node"], action_type)
                node_id = int(entry["node_id"])
                if candidate in used or node_id in used_node_ids:
                    continue
                cluster_id = int(entry["cluster_id"])
                cluster_penalty = float(1 + used_cluster_counts.get(cluster_id, 0)) ** 0.75 if cluster_id >= 0 else 1.0
                near_penalty = 0.0
                if node_id in near_by_type.get(action_type, set()):
                    near_penalty += 0.30
                if node_id in near_any:
                    near_penalty += 0.12
                adjusted = float(entry["base_score"]) / cluster_penalty - near_penalty
                pool.append(
                    {
                        "node": entry["node"],
                        "type": action_type,
                        "node_id": node_id,
                        "score": adjusted,
                        "near": near_penalty > 0.0,
                    }
                )
                taken += 1
                if taken >= scan_limit:
                    break
        pool.sort(key=lambda item: item["score"], reverse=True)
        pool_by_type: dict[str, list[dict]] = {action_type: [] for action_type in ACTION_TYPES}
        for item in pool:
            pool_by_type[item["type"]].append(item)

        selected: list[tuple[str, str]] = []
        selected_set: set[tuple[str, str]] = set()
        selected_node_ids: set[int] = set()
        selected_type_counts = {action_type: 0 for action_type in ACTION_TYPES}

        def try_add(item: dict, *, allow_near: bool, enforce_quota: bool) -> bool:
            action_type = item["type"]
            candidate = (item["node"], action_type)
            node_id = int(item["node_id"])
            if candidate in used or candidate in selected_set:
                return False
            if node_id in used_node_ids or node_id in selected_node_ids:
                return False
            if enforce_quota and selected_type_counts[action_type] >= type_quota[action_type]:
                return False
            if item["near"] and not allow_near:
                return False
            selected.append(candidate)
            selected_set.add(candidate)
            selected_node_ids.add(node_id)
            selected_type_counts[action_type] += 1
            return True

        for allow_near, enforce_quota in ((False, True), (True, True)):
            progressed = True
            cursors = {action_type: 0 for action_type in ACTION_TYPES}
            while progressed and len(selected) < max_candidates:
                progressed = False
                for action_type in action_type_order:
                    items = pool_by_type[action_type]
                    while cursors[action_type] < len(items):
                        item = items[cursors[action_type]]
                        cursors[action_type] += 1
                        if try_add(item, allow_near=allow_near, enforce_quota=enforce_quota):
                            progressed = True
                            break
                    if len(selected) >= max_candidates:
                        return selected
        for item in pool:
            if try_add(item, allow_near=True, enforce_quota=False):
                if len(selected) >= max_candidates:
                    return selected
        return selected


def _hard_cluster_manager(
    graph: GraphData,
    benchmark_id: str | None,
    real_fault_prior_path: str | None,
    activation_prior_path: str | None,
) -> HardClusterCandidateManager:
    """Return a cached hard-cluster manager for one circuit/prior tuple."""

    key = (id(graph), benchmark_id, real_fault_prior_path, activation_prior_path)
    cached = _HARD_CLUSTER_MANAGER_CACHE.get(key)
    if cached is None:
        cached = HardClusterCandidateManager(graph, benchmark_id, real_fault_prior_path, activation_prior_path)
        _HARD_CLUSTER_MANAGER_CACHE[key] = cached
    return cached


def hard_fault_cluster_lazy_sequence(
    graph: GraphData,
    budget: int,
    max_candidates: int,
    *,
    benchmark_id: str | None = None,
    real_fault_prior_path: str | Path | None = None,
    activation_prior_path: str | Path | None = None,
) -> list[tuple[str, str]]:
    """Build a heuristic sequence with one cached hard-cluster manager."""

    manager = _hard_cluster_manager(
        graph,
        benchmark_id or _REAL_FAULT_BENCHMARK_ID,
        str(real_fault_prior_path) if real_fault_prior_path else _REAL_FAULT_PRIOR_PATH,
        str(activation_prior_path) if activation_prior_path else _ACTIVATION_PRIOR_PATH,
    )
    selected: list[tuple[str, str]] = []
    for _ in range(max(0, int(budget))):
        candidates = manager.select(selected, max_candidates)
        if not candidates:
            break
        selected.append(candidates[0])
    return selected


def _hard_fault_cluster_candidate_slice(
    graph: GraphData,
    ranked: list[tuple[str, str, float]],
    used: set[tuple[str, str]],
    max_candidates: int,
    real_fault_benchmark_id: str | None,
    real_fault_prior_path: str | Path | None,
    activation_prior_path: str | Path | None,
) -> list[tuple[str, str]]:
    """Select candidates by rotating through high-mass hard-fault clusters."""

    if max_candidates <= 0:
        return []
    benchmark_id = real_fault_benchmark_id or _REAL_FAULT_BENCHMARK_ID
    prior_path = str(real_fault_prior_path) if real_fault_prior_path else _REAL_FAULT_PRIOR_PATH
    act_path = str(activation_prior_path) if activation_prior_path else _ACTIVATION_PRIOR_PATH
    return _hard_cluster_manager(graph, benchmark_id, prior_path, act_path).select(list(used), max_candidates)

    cluster_data = _hard_fault_cluster_data(
        graph,
        benchmark_id=benchmark_id,
        real_fault_prior_path=prior_path,
        activation_prior_path=act_path,
    )
    clusters = cluster_data["clusters"]
    if not clusters:
        return _balanced_candidate_slice(graph, ranked, max_candidates)

    node_to_id = _node_id_map(graph)
    used_node_ids_by_type: dict[str, list[int]] = {action_type: [] for action_type in ACTION_TYPES}
    used_action_type_counts: dict[str, int] = {action_type: 0 for action_type in ACTION_TYPES}
    used_node_ids: set[int] = set()
    used_cluster_counts: dict[int, int] = {}
    for node, action_type in used:
        node_id = node_to_id.get(node)
        if node_id is None:
            continue
        used_action_type_counts[action_type] = used_action_type_counts.get(action_type, 0) + 1
        used_node_ids.add(node_id)
        used_node_ids_by_type.setdefault(action_type, []).append(node_id)
        cluster_id, _ = _cluster_for_node(cluster_data, node_id, action_type)
        if cluster_id >= 0:
            used_cluster_counts[cluster_id] = used_cluster_counts.get(cluster_id, 0) + 1

    cluster_priority = {
        int(cluster["id"]): float(cluster["mass"]) / float(1 + used_cluster_counts.get(int(cluster["id"]), 0)) ** 0.75
        for cluster in clusters
    }
    cluster_order = sorted(cluster_priority, key=lambda cid: cluster_priority[cid], reverse=True)

    grouped: dict[int, dict[str, list[tuple[str, str, float, int]]]] = {
        int(cluster["id"]): {action_type: [] for action_type in ACTION_TYPES}
        for cluster in clusters
    }
    fallback: dict[str, list[tuple[str, str, float, int]]] = {action_type: [] for action_type in ACTION_TYPES}
    best_score_by_node: dict[int, float] = {}
    for node, _, score in ranked:
        node_id = node_to_id.get(node)
        if node_id is not None:
            best_score_by_node[node_id] = max(best_score_by_node.get(node_id, float("-inf")), float(score))
    comp = cluster_data.get("components", {})
    for node, action_type, score in ranked:
        node_id = node_to_id.get(node)
        if node_id is None:
            continue
        best_for_node = best_score_by_node.get(node_id)
        if best_for_node is not None and best_for_node > 0.0 and float(score) < 0.90 * best_for_node:
            continue
        cluster_id, cluster_score = _cluster_for_node(cluster_data, node_id, action_type)
        near_penalty = 0.0
        for selected_node in used_node_ids_by_type.get(action_type, []):
            distance = _local_distance_between(graph, node_id, selected_node, max_depth=6)
            if distance is not None:
                near_penalty = max(near_penalty, 0.60 / float(distance + 1))
        history_penalty = 0.12 * float(used_cluster_counts.get(cluster_id, 0)) if cluster_id >= 0 else 0.0
        if action_type == "control0":
            testability_score = float(comp.get("cc0", torch.zeros(graph.num_nodes))[node_id].item())
            sharing_score = float(comp.get("cone_pressure", torch.zeros(graph.num_nodes))[node_id].item())
        elif action_type == "control1":
            testability_score = float(comp.get("cc1", torch.zeros(graph.num_nodes))[node_id].item())
            sharing_score = float(comp.get("cone_pressure", torch.zeros(graph.num_nodes))[node_id].item())
        else:
            testability_score = float(comp.get("co", torch.zeros(graph.num_nodes))[node_id].item())
            sharing_score = float(comp.get("reconvergence", torch.zeros(graph.num_nodes))[node_id].item())
        adjusted = (
            0.55 * float(score)
            + 0.30 * cluster_score
            + 0.10 * testability_score
            + 0.05 * sharing_score
            - near_penalty
            - history_penalty
        )
        item = (node, action_type, adjusted, node_id)
        if cluster_id >= 0:
            grouped.setdefault(cluster_id, {name: [] for name in ACTION_TYPES})[action_type].append(item)
        else:
            fallback[action_type].append(item)

    for per_type in grouped.values():
        for action_type in ACTION_TYPES:
            per_type[action_type].sort(key=lambda item: item[2], reverse=True)
    for action_type in ACTION_TYPES:
        fallback[action_type].sort(key=lambda item: item[2], reverse=True)

    selected: list[tuple[str, str]] = []
    selected_set: set[tuple[str, str]] = set()
    selected_nodes_by_type: dict[str, list[int]] = {action_type: list(used_node_ids_by_type.get(action_type, [])) for action_type in ACTION_TYPES}
    selected_node_ids: set[int] = set()
    selected_node_list: list[int] = list(used_node_ids)
    type_quota = _cluster_action_type_quota(max_candidates, clusters, used_cluster_counts, used_action_type_counts)
    action_type_order = sorted(
        ACTION_TYPES,
        key=lambda action_type: (
            used_action_type_counts.get(action_type, 0),
            -type_quota.get(action_type, 0),
            action_type,
        ),
    )
    selected_type_counts = {action_type: 0 for action_type in ACTION_TYPES}

    def try_take(cluster_id: int, action_type: str, min_distance: int) -> bool:
        items = grouped.get(cluster_id, {}).get(action_type, [])
        while items:
            node, candidate_type, _, node_id = items.pop(0)
            candidate = (node, candidate_type)
            if candidate in used or candidate in selected_set:
                continue
            if node_id in used_node_ids or node_id in selected_node_ids:
                continue
            if selected_type_counts[action_type] >= type_quota[action_type]:
                return False
            if not _is_far_enough(graph, node_id, selected_nodes_by_type[action_type], min_distance):
                continue
            if not _is_far_enough(graph, node_id, selected_node_list, max(2, min_distance // 2)):
                continue
            selected.append(candidate)
            selected_set.add(candidate)
            selected_node_ids.add(node_id)
            selected_node_list.append(node_id)
            selected_nodes_by_type[action_type].append(node_id)
            selected_type_counts[action_type] += 1
            return True
        return False

    for min_distance in (6, 4, 2, 0):
        progressed = True
        while progressed and len(selected) < max_candidates:
            progressed = False
            for action_type in action_type_order:
                if selected_type_counts[action_type] >= type_quota[action_type]:
                    continue
                for cluster_id in cluster_order:
                    if try_take(cluster_id, action_type, min_distance):
                        progressed = True
                        break
                if len(selected) >= max_candidates:
                    break

    if len(selected) < max_candidates:
        for action_type in action_type_order:
            if selected_type_counts[action_type] >= type_quota[action_type]:
                continue
            for node, candidate_type, _ in ranked:
                if candidate_type != action_type:
                    continue
                node_id = node_to_id.get(node)
                if node_id is None:
                    continue
                candidate = (node, candidate_type)
                if candidate in used or candidate in selected_set:
                    continue
                if node_id in used_node_ids or node_id in selected_node_ids:
                    continue
                if not _is_far_enough(graph, node_id, selected_nodes_by_type[action_type], min_distance=2):
                    continue
                selected.append(candidate)
                selected_set.add(candidate)
                selected_node_ids.add(node_id)
                selected_node_list.append(node_id)
                selected_nodes_by_type[action_type].append(node_id)
                selected_type_counts[action_type] += 1
                if selected_type_counts[action_type] >= type_quota[action_type] or len(selected) >= max_candidates:
                    break

    if len(selected) < max_candidates:
        flattened: list[tuple[str, str, float, int]] = []
        for cluster_id in cluster_order:
            for action_type in action_type_order:
                flattened.extend(grouped.get(cluster_id, {}).get(action_type, []))
        for action_type in action_type_order:
            flattened.extend(fallback[action_type])
        flattened.sort(key=lambda item: item[2], reverse=True)
        for node, action_type, _, node_id in flattened:
            candidate = (node, action_type)
            if candidate in used or candidate in selected_set:
                continue
            if node_id in used_node_ids or node_id in selected_node_ids:
                continue
            if not _is_far_enough(graph, node_id, selected_node_list, min_distance=2):
                continue
            selected.append(candidate)
            selected_set.add(candidate)
            selected_node_ids.add(node_id)
            selected_node_list.append(node_id)
            if len(selected) >= max_candidates:
                break

    if len(selected) < max_candidates:
        for node, action_type, _ in ranked:
            candidate = (node, action_type)
            if candidate in used or candidate in selected_set:
                continue
            node_id = node_to_id.get(node)
            if node_id is not None and (node_id in used_node_ids or node_id in selected_node_ids):
                continue
            selected.append(candidate)
            selected_set.add(candidate)
            if node_id is not None:
                selected_node_ids.add(node_id)
            if len(selected) >= max_candidates:
                break
    return selected[:max_candidates]


def _is_far_enough(graph: GraphData, node_id: int, selected_node_ids: list[int], min_distance: int) -> bool:
    """Avoid filling the candidate pool with many nodes from the same tiny cone."""

    if min_distance <= 0:
        return True
    for selected in selected_node_ids:
        distance = _local_distance_between(graph, node_id, selected, max_depth=min_distance)
        if distance is not None and distance < min_distance:
            return False
    return True


def _balanced_candidate_slice(
    graph: GraphData,
    ranked: list[tuple[str, str, float]],
    max_candidates: int,
) -> list[tuple[str, str]]:
    """Keep a small candidate pool while avoiding action and local-region collapse."""

    if max_candidates <= 0:
        return []
    selected: list[tuple[str, str]] = []
    selected_set: set[tuple[str, str]] = set()
    selected_nodes_by_type: dict[str, list[int]] = {action_type: [] for action_type in ACTION_TYPES}
    node_to_id = _node_id_map(graph)
    per_type = max(1, max_candidates // max(1, len(ACTION_TYPES)))
    for action_type in ACTION_TYPES:
        count = 0
        for node, candidate_type, _ in ranked:
            candidate = (node, candidate_type)
            if candidate_type != action_type or candidate in selected_set:
                continue
            node_id = node_to_id.get(node)
            if node_id is None:
                continue
            if not _is_far_enough(graph, node_id, selected_nodes_by_type[action_type], min_distance=3):
                continue
            selected.append(candidate)
            selected_set.add(candidate)
            selected_nodes_by_type[action_type].append(node_id)
            count += 1
            if count >= per_type or len(selected) >= max_candidates:
                break
    if len(selected) < max_candidates:
        selected_nodes = [node_to_id[node] for node, _ in selected if node in node_to_id]
        for node, action_type, _ in ranked:
            candidate = (node, action_type)
            if candidate in selected_set:
                continue
            node_id = node_to_id.get(node)
            if node_id is None:
                continue
            if not _is_far_enough(graph, node_id, selected_nodes, min_distance=2):
                continue
            selected.append(candidate)
            selected_set.add(candidate)
            selected_nodes.append(node_id)
            if len(selected) >= max_candidates:
                break
    if len(selected) < max_candidates:
        for node, action_type, _ in ranked:
            candidate = (node, action_type)
            if candidate in selected_set:
                continue
            selected.append(candidate)
            selected_set.add(candidate)
            if len(selected) >= max_candidates:
                break
    return selected[:max_candidates]


def _ffr_anchor(graph: GraphData, node_id: int, max_depth: int = 32) -> int:
    """Approximate the downstream endpoint of the node's local fanout-free region."""

    current = node_id
    seen: set[int] = set()
    depth = 0
    while depth < max_depth and current not in seen:
        seen.add(current)
        fanouts = graph.fanout_lists[current]
        if len(fanouts) != 1:
            break
        nxt = fanouts[0]
        if len(graph.fanin_lists[nxt]) > 1:
            break
        current = nxt
        depth += 1
    return current


def _ffr_hier_candidate_slice(
    graph: GraphData,
    ranked: list[tuple[str, str, float]],
    max_candidates: int,
) -> list[tuple[str, str]]:
    """Select top FFR-like regions first, then pick action-balanced gates inside them."""

    if max_candidates <= 0:
        return []
    node_to_id = _node_id_map(graph)
    groups: dict[int, list[tuple[str, str, float]]] = {}
    for node, action_type, score in ranked:
        anchor = _ffr_anchor(graph, node_to_id[node])
        groups.setdefault(anchor, []).append((node, action_type, score))
    for items in groups.values():
        items.sort(key=lambda item: item[2], reverse=True)

    region_order = sorted(
        groups,
        key=lambda anchor: (groups[anchor][0][2], len(groups[anchor])),
        reverse=True,
    )
    max_regions = max(1, min(len(region_order), max_candidates // max(1, len(ACTION_TYPES)) or 1))
    region_order = region_order[:max_regions]
    per_region = max(1, min(3, (max_candidates + max_regions - 1) // max_regions))
    selected: list[tuple[str, str]] = []
    selected_set: set[tuple[str, str]] = set()
    region_counts: dict[int, int] = {anchor: 0 for anchor in region_order}

    for preferred_type in ("observe", "control0", "control1"):
        for anchor in region_order:
            if len(selected) >= max_candidates:
                break
            if region_counts[anchor] >= per_region:
                continue
            for node, action_type, _ in groups[anchor]:
                candidate = (node, action_type)
                if action_type != preferred_type or candidate in selected_set:
                    continue
                selected.append(candidate)
                selected_set.add(candidate)
                region_counts[anchor] += 1
                break

    if len(selected) < max_candidates:
        for anchor in region_order:
            for node, action_type, _ in groups[anchor]:
                candidate = (node, action_type)
                if candidate in selected_set:
                    continue
                selected.append(candidate)
                selected_set.add(candidate)
                region_counts[anchor] += 1
                if len(selected) >= max_candidates or region_counts[anchor] >= per_region:
                    break
            if len(selected) >= max_candidates:
                break

    if len(selected) < max_candidates:
        for node, action_type, _ in ranked:
            candidate = (node, action_type)
            if candidate in selected_set:
                continue
            selected.append(candidate)
            selected_set.add(candidate)
            if len(selected) >= max_candidates:
                break
    return selected[:max_candidates]


def _undirected_distance_to_selected(
    graph: GraphData,
    node_id: int,
    selected_actions: list[tuple[str, str]],
    max_depth: int,
) -> int | None:
    """Return local undirected distance to any selected TP node."""

    if not selected_actions or max_depth <= 0:
        return None
    node_to_id = _node_id_map(graph)
    selected_node_ids = {node_to_id[node] for node, _ in selected_actions if node in node_to_id}
    if node_id in selected_node_ids:
        return 0
    queue: deque[tuple[int, int]] = deque([(node_id, 0)])
    seen = {node_id}
    while queue:
        current, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for nxt in graph.fanin_lists[current] + graph.fanout_lists[current]:
            if nxt in seen:
                continue
            if nxt in selected_node_ids:
                return depth + 1
            seen.add(nxt)
            queue.append((nxt, depth + 1))
    return None


@torch.no_grad()
def score_candidate_from_latent(
    model: TPIWorldModel,
    graph: GraphData,
    z_state: torch.Tensor,
    candidate: tuple[str, str],
    device: torch.device,
    relation_mode: str = "basic",
    relation_depth: int = 8,
    selected_actions: list[tuple[str, str]] | None = None,
    diversity_penalty_weight: float = 0.0,
    diversity_depth: int = 4,
) -> dict:
    """Score one candidate using the current rolled-out latent state."""

    node, action_type = candidate
    action_node_id = _node_id_map(graph)[node]
    relation = _cached_relation_features(graph, action_node_id, relation_mode, relation_depth, device)
    out = model.predict_from_latent(
        z_state,
        action_node_id,
        action_type_to_id(action_type),
        relation,
        include_aux_heads=False,
    )
    coverage_scale = float(getattr(model, "coverage_scale", 100.0))
    q_pred = float(out["q_pred"].detach().cpu().item())
    reward_pred = float(out["reward_pred"].detach().cpu().item())
    return_pred = float(out["return_pred"].detach().cpu().item())
    hard_reduction_pred = out["hard_reduction_pred"].detach().cpu().view(-1)
    hard_reduction_total = float(hard_reduction_pred[0].item()) if hard_reduction_pred.numel() > 0 else 0.0
    hard_reduction_sa0 = float(hard_reduction_pred[1].item()) if hard_reduction_pred.numel() > 1 else 0.0
    hard_reduction_sa1 = float(hard_reduction_pred[2].item()) if hard_reduction_pred.numel() > 2 else 0.0
    derived_pre = out.get("derived_hard_count_pre_pred")
    derived_post = out.get("derived_hard_count_post_pred")
    derived_reduction = out.get("derived_hard_reduction_pred")
    derived_pre = derived_pre.detach().cpu().view(-1) if derived_pre is not None else hard_reduction_pred.new_zeros(3)
    derived_post = derived_post.detach().cpu().view(-1) if derived_post is not None else hard_reduction_pred.new_zeros(3)
    derived_reduction = (
        derived_reduction.detach().cpu().view(-1)
        if derived_reduction is not None
        else hard_reduction_pred.new_zeros(3)
    )
    derived_hard_count_pre_total = float(derived_pre[0].item()) if derived_pre.numel() > 0 else 0.0
    derived_hard_count_pre_sa0 = float(derived_pre[1].item()) if derived_pre.numel() > 1 else 0.0
    derived_hard_count_pre_sa1 = float(derived_pre[2].item()) if derived_pre.numel() > 2 else 0.0
    derived_hard_count_post_total = float(derived_post[0].item()) if derived_post.numel() > 0 else 0.0
    derived_hard_count_post_sa0 = float(derived_post[1].item()) if derived_post.numel() > 1 else 0.0
    derived_hard_count_post_sa1 = float(derived_post[2].item()) if derived_post.numel() > 2 else 0.0
    derived_hard_reduction_total = float(derived_reduction[0].item()) if derived_reduction.numel() > 0 else 0.0
    derived_hard_reduction_sa0 = float(derived_reduction[1].item()) if derived_reduction.numel() > 1 else 0.0
    derived_hard_reduction_sa1 = float(derived_reduction[2].item()) if derived_reduction.numel() > 2 else 0.0
    distance = _undirected_distance_to_selected(
        graph,
        action_node_id,
        selected_actions or [],
        max(0, int(diversity_depth)),
    )
    diversity_penalty = 0.0
    if distance is not None and float(diversity_penalty_weight) > 0.0:
        diversity_penalty = float(diversity_penalty_weight) / float(distance + 1)
    hybrid_pred = (
        return_pred
        + reward_pred
        + hard_reduction_total * coverage_scale
    )
    residual_alpha = max(
        -float(getattr(model, "bounded_residual_alpha_bound", 0.25)),
        min(float(getattr(model, "bounded_residual_alpha", 1.0)), float(getattr(model, "bounded_residual_alpha_bound", 0.25))),
    )
    bounded_residual_hybrid_pred = hard_reduction_total * coverage_scale + residual_alpha * (reward_pred + return_pred)
    derived_hard_reduction_hybrid_pred = derived_hard_reduction_total * coverage_scale
    return {
        "node": node,
        "type": action_type,
        "q_pred": q_pred,
        "score_pred": float(out["score_pred"].detach().cpu().item()),
        "reward_pred": reward_pred,
        "fc_pred": reward_pred / coverage_scale,
        "pattern_pred": float(out["pattern_pred"].detach().cpu().item()),
        "return_pred": return_pred,
        "guarded_reward": min(reward_pred, return_pred),
        "hard_reduction_total_pred": hard_reduction_total,
        "hard_reduction_sa0_pred": hard_reduction_sa0,
        "hard_reduction_sa1_pred": hard_reduction_sa1,
        "derived_hard_count_pre_total_pred": derived_hard_count_pre_total,
        "derived_hard_count_pre_sa0_pred": derived_hard_count_pre_sa0,
        "derived_hard_count_pre_sa1_pred": derived_hard_count_pre_sa1,
        "derived_hard_count_post_total_pred": derived_hard_count_post_total,
        "derived_hard_count_post_sa0_pred": derived_hard_count_post_sa0,
        "derived_hard_count_post_sa1_pred": derived_hard_count_post_sa1,
        "derived_hard_reduction_total_pred": derived_hard_reduction_total,
        "derived_hard_reduction_sa0_pred": derived_hard_reduction_sa0,
        "derived_hard_reduction_sa1_pred": derived_hard_reduction_sa1,
        "derived_hard_reduction_hybrid_pred": derived_hard_reduction_hybrid_pred,
        "hybrid_pred": hybrid_pred,
        "bounded_residual_hybrid_pred": bounded_residual_hybrid_pred,
        "diversity_penalty": diversity_penalty,
        "_z_pred": out["z_pred"].detach(),
    }


ENSEMBLE_SCORE_FIELDS = [
    "q_pred",
    "score_pred",
    "reward_pred",
    "return_pred",
    "guarded_reward",
    "hard_reduction_total_pred",
    "derived_hard_reduction_hybrid_pred",
    "hybrid_pred",
    "bounded_residual_hybrid_pred",
]


CONTEXT_SCORE_FIELDS = [
    "q_pred_context",
    "reward_pred_context",
    "guarded_reward_context",
    "hybrid_pred_context",
    "bounded_residual_hybrid_pred_context",
]


@torch.no_grad()
def score_candidate_ensemble_from_latents(
    models: list[TPIWorldModel],
    graph: GraphData,
    z_states: list[torch.Tensor],
    candidate: tuple[str, str],
    device: torch.device,
    relation_mode: str = "basic",
    relation_depth: int = 8,
    selected_actions: list[tuple[str, str]] | None = None,
    diversity_penalty_weight: float = 0.0,
    diversity_depth: int = 4,
    lcb_alpha: float = 1.0,
) -> dict:
    """Score a candidate with an ensemble and expose mean/std/lower-confidence fields."""

    rows = [
        score_candidate_from_latent(
            model,
            graph,
            z_state,
            candidate,
            device,
            relation_mode,
            relation_depth,
            selected_actions,
            diversity_penalty_weight,
            diversity_depth,
        )
        for model, z_state in zip(models, z_states)
    ]
    base = dict(rows[0])
    base["_z_preds"] = [row["_z_pred"] for row in rows]
    alpha = float(lcb_alpha)
    for field in ENSEMBLE_SCORE_FIELDS:
        values = torch.tensor([float(row[field]) for row in rows], dtype=torch.float32)
        mean = float(values.mean().item())
        std = float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0
        lcb = mean - alpha * std
        base[field] = mean
        base[f"{field}_mean"] = mean
        base[f"{field}_std"] = std
        base[f"{field}_lcb"] = lcb
    coverage_scale = float(getattr(models[0], "coverage_scale", 100.0))
    base["fc_pred"] = float(base["reward_pred"]) / coverage_scale
    return base


def _pool_zscores(rows: list[dict], field: str) -> list[float]:
    """Return robust within-pool z-scores for one numeric score field."""

    if not rows or not all(field in row for row in rows):
        return [0.0 for _ in rows]
    values = torch.tensor([float(row[field]) for row in rows], dtype=torch.float32)
    if values.numel() < 2:
        return [0.0 for _ in rows]
    center = values.median()
    mad = (values - center).abs().median()
    scale = 1.4826 * float(mad.item())
    if scale < 1e-6:
        scale = float(values.std(unbiased=False).item())
    if scale < 1e-6:
        return [0.0 for _ in rows]
    return [max(-6.0, min(6.0, float(item))) for item in ((values - center) / scale).tolist()]


def add_candidate_context_scores(rows: list[dict]) -> None:
    """Add pool-relative multi-head support scores to already scored candidates."""

    if not rows:
        return
    zscores = {
        field: _pool_zscores(rows, field)
        for field in [
            "q_pred",
            "reward_pred",
            "return_pred",
            "guarded_reward",
            "hard_reduction_total_pred",
            "derived_hard_reduction_hybrid_pred",
            "hybrid_pred",
            "bounded_residual_hybrid_pred",
        ]
    }

    def z(field: str, index: int) -> float:
        values = zscores.get(field)
        return values[index] if values is not None else 0.0

    for index, row in enumerate(rows):
        hard_support = 0.5 * z("derived_hard_reduction_hybrid_pred", index) + 0.5 * z(
            "hard_reduction_total_pred",
            index,
        )
        q_support = 0.45 * z("reward_pred", index) + 0.35 * z("return_pred", index) + 0.20 * hard_support
        reward_support = 0.35 * z("q_pred", index) + 0.35 * z("return_pred", index) + 0.30 * hard_support
        guarded_support = 0.35 * z("q_pred", index) + 0.35 * z("reward_pred", index) + 0.30 * hard_support
        hybrid_support = 0.40 * z("q_pred", index) + 0.35 * z("guarded_reward", index) + 0.25 * hard_support

        q_base = z("q_pred", index)
        reward_base = z("reward_pred", index)
        guarded_base = z("guarded_reward", index)
        hybrid_base = z("hybrid_pred", index)
        bounded_hybrid_base = z("bounded_residual_hybrid_pred", index)

        row["q_pred_context"] = q_base + 0.35 * q_support - 0.15 * max(0.0, q_base - q_support)
        row["reward_pred_context"] = reward_base + 0.35 * reward_support - 0.15 * max(
            0.0,
            reward_base - reward_support,
        )
        row["guarded_reward_context"] = guarded_base + 0.30 * guarded_support - 0.15 * max(
            0.0,
            guarded_base - guarded_support,
        )
        row["hybrid_pred_context"] = hybrid_base + 0.30 * hybrid_support - 0.10 * max(
            0.0,
            hybrid_base - hybrid_support,
        )
        row["bounded_residual_hybrid_pred_context"] = bounded_hybrid_base + 0.30 * hybrid_support - 0.10 * max(
            0.0,
            bounded_hybrid_base - hybrid_support,
        )


def _apply_step_value(row: dict, score_field: str) -> float:
    """Use the selected score field after optional planner-level penalties."""

    adjusted = float(row[score_field]) - float(row.get("diversity_penalty") or 0.0)
    row["score_adjusted"] = adjusted
    return adjusted


def _limit_candidates(candidates: list[tuple[str, str]], limit: int | None) -> list[tuple[str, str]]:
    """Apply an optional positive candidate limit."""

    if limit is None:
        return candidates
    return candidates[: max(0, int(limit))]


def _resolve_candidate_stage_limits(
    max_candidates: int | None,
    k_recall: int | None,
    k_model: int | None,
    k_plan: int | None,
) -> tuple[int | None, int | None, int | None]:
    """Resolve legacy and staged candidate limits into recall/model/planner sizes."""

    recall_limit = int(k_recall) if k_recall is not None else max_candidates
    model_limit = int(k_model) if k_model is not None else recall_limit
    plan_limit = int(k_plan) if k_plan is not None else model_limit
    if recall_limit is not None:
        recall_limit = max(0, recall_limit)
    if model_limit is not None:
        model_limit = max(0, model_limit)
        if recall_limit is not None:
            model_limit = min(model_limit, recall_limit)
    if plan_limit is not None:
        plan_limit = max(0, plan_limit)
        if model_limit is not None:
            plan_limit = min(plan_limit, model_limit)
    return recall_limit, model_limit, plan_limit


def _stage_candidate_recall(
    graph: GraphData,
    selected: list[tuple[str, str]],
    recall_limit: int | None,
    candidate_strategy: str,
    candidate_cache_dir: str | Path | None,
    candidate_sample_seed: int,
    real_fault_benchmark_id: str | None = None,
    real_fault_prior_path: str | Path | None = None,
    activation_prior_path: str | Path | None = None,
) -> list[tuple[str, str]]:
    """Run only the heuristic recall stage."""

    return enumerate_candidates(
        graph,
        selected,
        recall_limit,
        candidate_strategy,
        real_fault_benchmark_id=real_fault_benchmark_id,
        real_fault_prior_path=real_fault_prior_path,
        activation_prior_path=activation_prior_path,
        candidate_cache_dir=candidate_cache_dir,
        candidate_sample_seed=candidate_sample_seed,
    )


@torch.no_grad()
def greedy_plan(
    model: TPIWorldModel,
    graph: GraphData,
    budget: int,
    device: torch.device,
    max_candidates: int | None = 8,
    score_field: str = "q_pred",
    feature_mode: str = "basic",
    relation_mode: str = "basic",
    relation_depth: int = 8,
    candidate_strategy: str = "testability",
    candidate_diversity_penalty: float = 0.0,
    candidate_diversity_depth: int = 4,
    candidate_cache_dir: str | Path | None = None,
    candidate_sample_seed: int = 0,
    k_recall: int | None = None,
    k_model: int | None = None,
    k_plan: int | None = None,
    candidate_real_fault_prior_path: str | Path | None = None,
    candidate_activation_prior_path: str | Path | None = None,
    ensemble_models: list[TPIWorldModel] | None = None,
    ensemble_lcb_alpha: float = 1.0,
) -> list[dict]:
    """Select actions greedily by rolling the latent state forward."""

    active_models = ensemble_models or [model]
    selected: list[tuple[str, str]] = []
    rows: list[dict] = []
    recall_limit, model_limit, _ = _resolve_candidate_stage_limits(max_candidates, k_recall, k_model, k_plan)
    base_features = make_base_node_features(
        graph,
        feature_mode,
        benchmark_id=_REAL_FAULT_BENCHMARK_ID,
        real_fault_prior_path=_REAL_FAULT_PRIOR_PATH,
        activation_prior_path=_ACTIVATION_PRIOR_PATH,
    )
    x_state = make_state_features(graph, selected, base_features).to(device)
    edge_src = graph.edge_src.to(device)
    edge_dst = graph.edge_dst.to(device)
    gate_type_ids = graph.gate_type_ids.to(device)
    z_state = model.online_encoder(x_state, edge_src, edge_dst, gate_type_ids)
    ensemble_z_states = [
        active_model.online_encoder(x_state, edge_src, edge_dst, gate_type_ids)
        for active_model in active_models
    ]

    for step in range(1, budget + 1):
        recalled = _stage_candidate_recall(
            graph,
            selected,
            recall_limit,
            candidate_strategy,
            candidate_cache_dir,
            candidate_sample_seed,
            real_fault_benchmark_id=_REAL_FAULT_BENCHMARK_ID,
            real_fault_prior_path=candidate_real_fault_prior_path,
            activation_prior_path=candidate_activation_prior_path,
        )
        candidates = _limit_candidates(recalled, model_limit)
        if not candidates:
            break
        scored = []
        for candidate in candidates:
            if ensemble_models:
                row = score_candidate_ensemble_from_latents(
                    active_models,
                    graph,
                    ensemble_z_states,
                    candidate,
                    device,
                    relation_mode,
                    relation_depth,
                    selected,
                    candidate_diversity_penalty,
                    candidate_diversity_depth,
                    ensemble_lcb_alpha,
                )
            else:
                row = score_candidate_from_latent(
                    model,
                    graph,
                    z_state,
                    candidate,
                    device,
                    relation_mode,
                    relation_depth,
                    selected,
                    candidate_diversity_penalty,
                    candidate_diversity_depth,
                )
            scored.append(row)
        add_candidate_context_scores(scored)
        for row in scored:
            _apply_step_value(row, score_field)
        best = max(scored, key=lambda row: row["score_adjusted"])
        selected.append((best["node"], best["type"]))
        z_state = best.pop("_z_pred")
        if ensemble_models:
            ensemble_z_states = best.pop("_z_preds")
        step_value = float(best["score_adjusted"])
        sequence_score = (rows[-1]["sequence_score"] if rows else 0.0) + step_value
        rows.append(
            {
                "step": step,
                **best,
                "step_value": step_value,
                "sequence_score": sequence_score,
                "lookahead_score": sequence_score,
                "candidate_strategy": candidate_strategy,
                "planner": "greedy",
            }
        )
        print(
            f"[plan] step={step} node={best['node']} type={best['type']} "
            f"{score_field}={best[score_field]:.6f} sequence={sequence_score:.6f}"
        )
    return rows


def _expand_beam_paths(
    model: TPIWorldModel,
    graph: GraphData,
    beams: list[BeamPath],
    prefix_selected: list[tuple[str, str]],
    device: torch.device,
    max_candidates: int | None,
    score_field: str,
    beam_width: int,
    objective: str,
    discount_gamma: float,
    relation_mode: str,
    relation_depth: int,
    candidate_strategy: str,
    candidate_diversity_penalty: float,
    candidate_diversity_depth: int,
    candidate_cache_dir: str | Path | None,
    candidate_sample_seed: int,
    k_recall: int | None = None,
    k_model: int | None = None,
    k_plan: int | None = None,
    candidate_real_fault_prior_path: str | Path | None = None,
    candidate_activation_prior_path: str | Path | None = None,
) -> list[BeamPath]:
    """Expand beam paths by one action and keep the strongest suffixes."""

    expanded: list[BeamPath] = []
    recall_limit, model_limit, plan_limit = _resolve_candidate_stage_limits(max_candidates, k_recall, k_model, k_plan)
    for beam in beams:
        selected_so_far = prefix_selected + beam.selected
        recalled = _stage_candidate_recall(
            graph,
            selected_so_far,
            recall_limit,
            candidate_strategy,
            candidate_cache_dir,
            candidate_sample_seed,
            real_fault_benchmark_id=_REAL_FAULT_BENCHMARK_ID,
            real_fault_prior_path=candidate_real_fault_prior_path,
            activation_prior_path=candidate_activation_prior_path,
        )
        candidates = _limit_candidates(recalled, model_limit)
        scored_rows: list[tuple[tuple[str, str], dict, float]] = []
        for candidate in candidates:
            scored = score_candidate_from_latent(
                model,
                graph,
                beam.z_state,
                candidate,
                device,
                relation_mode,
                relation_depth,
                selected_so_far,
                candidate_diversity_penalty,
                candidate_diversity_depth,
            )
            scored_rows.append((candidate, scored, 0.0))
        add_candidate_context_scores([scored for _, scored, _ in scored_rows])
        scored_rows = [
            (candidate, scored, _apply_step_value(scored, score_field))
            for candidate, scored, _ in scored_rows
        ]
        scored_rows.sort(key=lambda item: item[2], reverse=True)
        for candidate, scored, step_value in (scored_rows if plan_limit is None else scored_rows[: max(0, int(plan_limit))]):
            values = beam.values + [step_value]
            objective_score = sequence_objective(values, objective, discount_gamma)
            row = {
                **scored,
                "step_value": step_value,
                "sequence_score": beam.sequence_score + step_value,
                "objective_score": objective_score,
                "objective": objective,
                "candidate_strategy": candidate_strategy,
                "planner": "beam",
            }
            expanded.append(
                BeamPath(
                    selected=beam.selected + [candidate],
                    z_state=scored["_z_pred"],
                    rows=beam.rows + [row],
                    values=values,
                    sequence_score=beam.sequence_score + step_value,
                    objective_score=objective_score,
                )
            )
    expanded.sort(key=lambda path: path.objective_score, reverse=True)
    return expanded[:beam_width]


@torch.no_grad()
def beam_rollout_plan(
    model: TPIWorldModel,
    graph: GraphData,
    budget: int,
    device: torch.device,
    max_candidates: int | None = 8,
    beam_width: int = 4,
    lookahead_depth: int = 3,
    score_field: str = "q_pred",
    objective: str = "cumulative",
    discount_gamma: float = 1.0,
    feature_mode: str = "basic",
    relation_mode: str = "basic",
    relation_depth: int = 8,
    candidate_strategy: str = "testability",
    candidate_diversity_penalty: float = 0.0,
    candidate_diversity_depth: int = 4,
    candidate_cache_dir: str | Path | None = None,
    candidate_sample_seed: int = 0,
    k_recall: int | None = None,
    k_model: int | None = None,
    k_plan: int | None = None,
    candidate_real_fault_prior_path: str | Path | None = None,
    candidate_activation_prior_path: str | Path | None = None,
) -> list[dict]:
    """Plan with receding-horizon beam search over latent rollout states."""

    selected: list[tuple[str, str]] = []
    rows: list[dict] = []
    base_features = make_base_node_features(
        graph,
        feature_mode,
        benchmark_id=_REAL_FAULT_BENCHMARK_ID,
        real_fault_prior_path=_REAL_FAULT_PRIOR_PATH,
        activation_prior_path=_ACTIVATION_PRIOR_PATH,
    )
    x_state = make_state_features(graph, selected, base_features).to(device)
    edge_src = graph.edge_src.to(device)
    edge_dst = graph.edge_dst.to(device)
    gate_type_ids = graph.gate_type_ids.to(device)
    z_state = model.online_encoder(x_state, edge_src, edge_dst, gate_type_ids)
    cumulative_score = 0.0

    for step in range(1, budget + 1):
        beams = [
            BeamPath(
                selected=[],
                z_state=z_state,
                rows=[],
                values=[],
                sequence_score=0.0,
                objective_score=0.0,
            )
        ]
        for _ in range(max(1, lookahead_depth)):
            beams = _expand_beam_paths(
                model,
                graph,
                beams,
                selected,
                device,
                max_candidates,
                score_field,
                max(1, beam_width),
                objective,
                discount_gamma,
                relation_mode,
                relation_depth,
                candidate_strategy,
                candidate_diversity_penalty,
                candidate_diversity_depth,
                candidate_cache_dir,
                candidate_sample_seed,
                k_recall,
                k_model,
                k_plan,
                candidate_real_fault_prior_path,
                candidate_activation_prior_path,
            )
            if not beams:
                break
        if not beams or not beams[0].rows:
            break

        best_path = beams[0]
        first = dict(best_path.rows[0])
        chosen = (first["node"], first["type"])
        selected.append(chosen)
        z_state = first.pop("_z_pred")
        step_value = float(first["step_value"])
        cumulative_score += step_value
        first.update(
            {
                "step": step,
                "step_value": step_value,
                "sequence_score": cumulative_score,
                "lookahead_score": best_path.sequence_score,
                "objective_score": best_path.objective_score,
                "objective": objective,
                "planner": "beam",
            }
        )
        rows.append(first)
        print(
            f"[plan] step={step} node={first['node']} type={first['type']} "
            f"{score_field}={step_value:.6f} lookahead={best_path.sequence_score:.6f} "
            f"objective={best_path.objective_score:.6f} sequence={cumulative_score:.6f}"
        )
    return rows


@torch.no_grad()
def beam_full_sequence_plan(
    model: TPIWorldModel,
    graph: GraphData,
    budget: int,
    device: torch.device,
    max_candidates: int | None = 8,
    beam_width: int = 4,
    score_field: str = "q_pred",
    objective: str = "cumulative",
    discount_gamma: float = 1.0,
    feature_mode: str = "basic",
    relation_mode: str = "basic",
    relation_depth: int = 8,
    candidate_strategy: str = "testability",
    candidate_diversity_penalty: float = 0.0,
    candidate_diversity_depth: int = 4,
    candidate_cache_dir: str | Path | None = None,
    candidate_sample_seed: int = 0,
    k_recall: int | None = None,
    k_model: int | None = None,
    k_plan: int | None = None,
    candidate_real_fault_prior_path: str | Path | None = None,
    candidate_activation_prior_path: str | Path | None = None,
) -> list[dict]:
    """Select a full action sequence once with beam search."""

    selected: list[tuple[str, str]] = []
    base_features = make_base_node_features(
        graph,
        feature_mode,
        benchmark_id=_REAL_FAULT_BENCHMARK_ID,
        real_fault_prior_path=_REAL_FAULT_PRIOR_PATH,
        activation_prior_path=_ACTIVATION_PRIOR_PATH,
    )
    x_state = make_state_features(graph, selected, base_features).to(device)
    edge_src = graph.edge_src.to(device)
    edge_dst = graph.edge_dst.to(device)
    gate_type_ids = graph.gate_type_ids.to(device)
    z_state = model.online_encoder(x_state, edge_src, edge_dst, gate_type_ids)
    beams = [
        BeamPath(
            selected=[],
            z_state=z_state,
            rows=[],
            values=[],
            sequence_score=0.0,
            objective_score=0.0,
        )
    ]
    for _ in range(max(1, budget)):
        beams = _expand_beam_paths(
            model,
            graph,
            beams,
            selected,
            device,
            max_candidates,
            score_field,
            max(1, beam_width),
            objective,
            discount_gamma,
            relation_mode,
            relation_depth,
            candidate_strategy,
            candidate_diversity_penalty,
            candidate_diversity_depth,
            candidate_cache_dir,
            candidate_sample_seed,
            k_recall,
            k_model,
            k_plan,
            candidate_real_fault_prior_path,
            candidate_activation_prior_path,
        )
        if not beams:
            break

    if not beams:
        return []
    best_path = beams[0]
    rows: list[dict] = []
    cumulative_score = 0.0
    for step, row in enumerate(best_path.rows, start=1):
        clean = dict(row)
        clean.pop("_z_pred", None)
        cumulative_score += float(clean["step_value"])
        clean.update(
            {
                "step": step,
                "sequence_score": cumulative_score,
                "lookahead_score": best_path.sequence_score,
                "objective_score": best_path.objective_score,
                "objective": objective,
                "planner": "beam_full",
            }
        )
        rows.append(clean)
        print(
            f"[plan] step={step} node={clean['node']} type={clean['type']} "
            f"{score_field}={clean[score_field]:.6f} objective={best_path.objective_score:.6f} "
            f"sequence={cumulative_score:.6f}"
        )
    return rows


def write_plan_csv(path: str | Path, rows: list[dict]) -> None:
    """Write selected greedy plan rows to CSV."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        clean_rows = [{key: value for key, value in row.items() if not key.startswith("_")} for row in rows]
        fieldnames = [name for name in PLAN_FIELDNAMES if any(name in row for row in clean_rows)]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(clean_rows)


def main() -> None:
    """CLI entry point for greedy planning."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--k-recall", type=int, default=None)
    parser.add_argument("--k-model", type=int, default=None)
    parser.add_argument("--k-plan", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--planner", choices=["greedy", "beam", "beam_full"], default="greedy")
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--lookahead-depth", type=int, default=3)
    parser.add_argument(
        "--score-field",
        choices=[
            "q_pred",
            "q_pred_lcb",
            "q_pred_context",
            "reward_pred",
            "reward_pred_lcb",
            "reward_pred_context",
            "fc_pred",
            "score_pred",
            "score_pred_lcb",
            "pattern_pred",
            "return_pred",
            "return_pred_lcb",
            "guarded_reward",
            "guarded_reward_lcb",
            "guarded_reward_context",
            "hard_reduction_total_pred",
            "hard_reduction_total_pred_lcb",
            "hybrid_pred",
            "hybrid_pred_lcb",
            "hybrid_pred_context",
            "bounded_residual_hybrid_pred",
            "bounded_residual_hybrid_pred_lcb",
            "bounded_residual_hybrid_pred_context",
            "derived_hard_reduction_total_pred",
            "derived_hard_reduction_hybrid_pred",
            "derived_hard_reduction_hybrid_pred_lcb",
        ],
        default="q_pred",
    )
    parser.add_argument(
        "--beam-objective",
        choices=["cumulative", "terminal", "mean", "discounted"],
        default="cumulative",
    )
    parser.add_argument("--discount-gamma", type=float, default=1.0)
    parser.add_argument("--feature-mode", default=None)
    parser.add_argument("--real-fault-priors", default=None)
    parser.add_argument("--candidate-real-fault-priors", default=None)
    parser.add_argument("--activation-priors", default=None)
    parser.add_argument("--candidate-activation-priors", default=None)
    parser.add_argument("--relation-mode", default=None)
    parser.add_argument("--relation-depth", type=int, default=None)
    parser.add_argument(
        "--candidate-strategy",
        choices=[
            "netlist",
            "testability",
            "hard_fault",
            "hard_fault_cone",
            "hard_fault_cluster",
            "hard_fault_ranked",
            "hard_fault_recall_union",
            "reconvergence",
            "reconvergence_ranked",
            "ffr",
            "ffr_ranked",
            "ffr_hier",
            "mixed",
            "mixed_ranked",
            "testability_ranked",
            "recall_pool",
            "heuristic_recall_pool",
            "cached_netlist",
            "cached_hard_cone",
            "cached_stride",
            "cached_random",
        ],
        default=None,
    )
    parser.add_argument("--candidate-diversity-penalty", type=float, default=None)
    parser.add_argument("--candidate-diversity-depth", type=int, default=None)
    parser.add_argument("--candidate-cache-dir", default=None)
    parser.add_argument("--candidate-sample-seed", type=int, default=0)
    parser.add_argument(
        "--ensemble-checkpoints",
        default=None,
        help="Comma-separated checkpoints for greedy ensemble scoring. The primary --checkpoint is used if omitted.",
    )
    parser.add_argument("--ensemble-lcb-alpha", type=float, default=1.0)
    parser.add_argument("--torch-threads", type=int, default=int(os.environ.get("TPI_PLAN_THREADS", "1")))
    args = parser.parse_args()

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    model, config = load_checkpoint(args.checkpoint, device)
    ensemble_models = None
    if args.ensemble_checkpoints:
        if args.planner != "greedy":
            raise SystemExit("--ensemble-checkpoints currently supports --planner greedy only")
        ensemble_paths = [item.strip() for item in args.ensemble_checkpoints.split(",") if item.strip()]
        if len(ensemble_paths) < 2:
            raise SystemExit("--ensemble-checkpoints requires at least two checkpoints")
        loaded = [load_checkpoint(path, device) for path in ensemble_paths]
        ensemble_models = [item[0] for item in loaded]
        print(
            f"[plan] ensemble_checkpoints={len(ensemble_models)} "
            f"score_field={args.score_field} lcb_alpha={args.ensemble_lcb_alpha}"
        )
    feature_mode = args.feature_mode or str(config.get("feature_mode", "basic"))
    real_fault_priors = args.real_fault_priors or config.get("real_fault_priors") or config.get("real_fault_prior_path")
    activation_priors = args.activation_priors or config.get("activation_priors") or config.get("activation_prior_path")
    candidate_real_fault_priors = args.candidate_real_fault_priors or real_fault_priors
    candidate_activation_priors = args.candidate_activation_priors or activation_priors
    set_real_fault_context(args.benchmark_id, real_fault_priors, activation_priors)
    relation_mode = args.relation_mode or str(config.get("relation_mode", "basic"))
    relation_depth = args.relation_depth if args.relation_depth is not None else int(config.get("relation_depth", 8))
    candidate_strategy = args.candidate_strategy or str(config.get("candidate_strategy", "testability"))
    candidate_diversity_penalty = (
        args.candidate_diversity_penalty
        if args.candidate_diversity_penalty is not None
        else float(config.get("candidate_diversity_penalty", 0.0))
    )
    candidate_diversity_depth = (
        args.candidate_diversity_depth
        if args.candidate_diversity_depth is not None
        else int(config.get("candidate_diversity_depth", 4))
    )
    k_recall = args.k_recall if args.k_recall is not None else config.get("k_recall")
    k_model = args.k_model if args.k_model is not None else config.get("k_model")
    k_plan = args.k_plan if args.k_plan is not None else config.get("k_plan")
    k_recall = int(k_recall) if k_recall is not None else None
    k_model = int(k_model) if k_model is not None else None
    k_plan = int(k_plan) if k_plan is not None else None
    graph = build_graph(parse_bench(find_bench_path(args.benchmark_id)))
    if args.planner == "beam":
        rows = beam_rollout_plan(
            model,
            graph,
            args.budget,
            device,
            args.max_candidates,
            args.beam_width,
            args.lookahead_depth,
            args.score_field,
            args.beam_objective,
            args.discount_gamma,
            feature_mode,
            relation_mode,
            relation_depth,
            candidate_strategy,
            candidate_diversity_penalty,
            candidate_diversity_depth,
            args.candidate_cache_dir,
            args.candidate_sample_seed,
            k_recall,
            k_model,
            k_plan,
            candidate_real_fault_priors,
            candidate_activation_priors,
        )
    elif args.planner == "beam_full":
        rows = beam_full_sequence_plan(
            model,
            graph,
            args.budget,
            device,
            args.max_candidates,
            args.beam_width,
            args.score_field,
            args.beam_objective,
            args.discount_gamma,
            feature_mode,
            relation_mode,
            relation_depth,
            candidate_strategy,
            candidate_diversity_penalty,
            candidate_diversity_depth,
            args.candidate_cache_dir,
            args.candidate_sample_seed,
            k_recall,
            k_model,
            k_plan,
            candidate_real_fault_priors,
            candidate_activation_priors,
        )
    else:
        rows = greedy_plan(
            model,
            graph,
            args.budget,
            device,
            args.max_candidates,
            args.score_field,
            feature_mode,
            relation_mode,
            relation_depth,
            candidate_strategy,
            candidate_diversity_penalty,
            candidate_diversity_depth,
            args.candidate_cache_dir,
            args.candidate_sample_seed,
            k_recall,
            k_model,
            k_plan,
            candidate_real_fault_priors,
            candidate_activation_priors,
            ensemble_models,
            args.ensemble_lcb_alpha,
        )
    out = Path(args.out) if args.out else Path(args.checkpoint).parent / "plans" / f"{args.benchmark_id}_plan.csv"
    write_plan_csv(out, rows)
    print(f"saved={out}")


if __name__ == "__main__":
    main()
