"""Testability-centric graph features and sparse propagation helpers."""

from __future__ import annotations

from collections import deque

import torch

from .graph import GATE_TYPES, GraphData


STRUCTURAL_DIM = 6
SCOAP_START = len(GATE_TYPES) + STRUCTURAL_DIM
SCOAP_END = SCOAP_START + 3
REGION_FEATURE_DIM = 8
TRANSPARENT_GATES = {"BUF", "BUFF", "INV", "NOT", "WIRE"}


def _gate_name(graph: GraphData, node: int) -> str:
    return GATE_TYPES[int(graph.gate_type_ids[node].item())]


def _bounded_reach(start: int, adjacency: list[list[int]], max_depth: int) -> set[int]:
    seen: set[int] = set()
    queue = deque([(start, 0)])
    while queue:
        node, depth = queue.popleft()
        if node in seen or depth > max_depth:
            continue
        seen.add(node)
        if depth == max_depth:
            continue
        for nxt in adjacency[node]:
            queue.append((nxt, depth + 1))
    return seen


def _bounded_count(start: int, adjacency: list[list[int]], max_depth: int) -> int:
    return len(_bounded_reach(start, adjacency, max_depth))


def compute_reconvergence_pressure(graph: GraphData, max_depth: int = 6) -> torch.Tensor:
    """Estimate how strongly each fanout stem reconverges within a local window."""

    values = torch.zeros(graph.num_nodes, dtype=torch.float32)
    for node, fanouts in enumerate(graph.fanout_lists):
        if len(fanouts) < 2:
            continue
        branch_sets = [_bounded_reach(dst, graph.fanout_lists, max_depth) for dst in fanouts]
        union: set[int] = set()
        duplicate_hits = 0
        seen_once: set[int] = set()
        for branch in branch_sets:
            union.update(branch)
            duplicate_hits += len(seen_once & branch)
            seen_once.update(branch)
        if union:
            values[node] = float(duplicate_hits) / float(len(union))
    return values.clamp(0.0, 1.0)


def compute_ffr_span(graph: GraphData, max_depth: int = 32) -> torch.Tensor:
    """Approximate fanout-free-region span starting from each node."""

    spans = torch.zeros(graph.num_nodes, dtype=torch.float32)
    denom = torch.log1p(torch.tensor(float(max(1, min(max_depth, graph.num_nodes)))))
    for node in range(graph.num_nodes):
        span = 0
        current = node
        visited: set[int] = set()
        while span < max_depth and current not in visited:
            visited.add(current)
            fanouts = graph.fanout_lists[current]
            if len(fanouts) != 1:
                break
            nxt = fanouts[0]
            span += 1
            if len(graph.fanin_lists[nxt]) > 1:
                break
            current = nxt
        spans[node] = torch.log1p(torch.tensor(float(span))) / denom.clamp_min(1.0)
    return spans.clamp(0.0, 1.0)


def compute_transparent_chain_score(graph: GraphData, max_depth: int = 16) -> torch.Tensor:
    """Score nodes that are part of low-information buffer/inverter/wire chains."""

    scores = torch.zeros(graph.num_nodes, dtype=torch.float32)
    denom = float(max(1, max_depth))
    for node in range(graph.num_nodes):
        if _gate_name(graph, node) not in TRANSPARENT_GATES:
            continue
        length = 1
        current = node
        while length < max_depth and len(graph.fanout_lists[current]) == 1:
            nxt = graph.fanout_lists[current][0]
            if _gate_name(graph, nxt) not in TRANSPARENT_GATES:
                break
            length += 1
            current = nxt
        current = node
        while length < max_depth and len(graph.fanin_lists[current]) == 1:
            prev = graph.fanin_lists[current][0]
            if _gate_name(graph, prev) not in TRANSPARENT_GATES:
                break
            length += 1
            current = prev
        scores[node] = float(length) / denom
    return scores.clamp(0.0, 1.0)


def compute_cone_pressure(graph: GraphData, max_depth: int = 4) -> torch.Tensor:
    """Local TFI/TFO density proxy used as a cheap cone-region feature."""

    rows: list[float] = []
    denom = float(max(1, graph.num_nodes))
    for node in range(graph.num_nodes):
        fanin = _bounded_count(node, graph.fanin_lists, max_depth)
        fanout = _bounded_count(node, graph.fanout_lists, max_depth)
        rows.append(float(fanin + fanout) / denom)
    x = torch.tensor(rows, dtype=torch.float32)
    return x / x.max().clamp_min(1.0)


def compute_testability_region_features(graph: GraphData, base_features: torch.Tensor) -> torch.Tensor:
    """Return node-level features for hard regions, FFRs, reconvergence, and chains."""

    if base_features.shape[1] < SCOAP_END:
        raise ValueError(
            f"base_features has {base_features.shape[1]} columns, needs at least {SCOAP_END}"
        )
    scoap = base_features[:, SCOAP_START:SCOAP_END]
    cc0 = scoap[:, 0]
    cc1 = scoap[:, 1]
    co = scoap[:, 2]
    hard_control = torch.maximum(cc0, cc1)
    control_imbalance = (cc0 - cc1).abs()
    hard_observe = co
    reconvergence = compute_reconvergence_pressure(graph)
    ffr_span = compute_ffr_span(graph)
    transparent_chain = compute_transparent_chain_score(graph)
    cone_pressure = compute_cone_pressure(graph)
    hard_fault_proxy = (hard_control + hard_observe + reconvergence) / 3.0
    region = torch.stack(
        [
            hard_control,
            control_imbalance,
            hard_observe,
            hard_fault_proxy,
            reconvergence,
            ffr_span,
            transparent_chain,
            cone_pressure,
        ],
        dim=1,
    )
    scale = region.abs().amax(dim=0).clamp_min(1.0)
    return region / scale


def make_edge_weights(
    x: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    mode: str = "mean",
    keep_ratio: float = 1.0,
) -> torch.Tensor | None:
    """Create optional edge weights for testability-guided sparse message passing."""

    if edge_src.numel() == 0 or mode in {"", "mean", "uniform"}:
        return None
    if x.shape[1] < SCOAP_END:
        return None
    scoap = x[:, SCOAP_START:SCOAP_END]
    cc0 = scoap[:, 0]
    cc1 = scoap[:, 1]
    co = scoap[:, 2]
    src_hard = (torch.maximum(cc0[edge_src], cc1[edge_src]) + co[edge_src]) * 0.5
    dst_hard = (torch.maximum(cc0[edge_dst], cc1[edge_dst]) + co[edge_dst]) * 0.5
    co_drop = (co[edge_src] - co[edge_dst]).abs()
    if mode == "fault_path":
        weights = 0.45 * src_hard + 0.35 * dst_hard + 0.20 * co_drop
    elif mode == "observe":
        weights = 0.25 * src_hard + 0.55 * co[edge_src] + 0.20 * co_drop
    elif mode == "control":
        weights = 0.55 * torch.maximum(cc0[edge_dst], cc1[edge_dst]) + 0.25 * src_hard + 0.20 * co_drop
    else:
        weights = 0.40 * src_hard + 0.40 * dst_hard + 0.20 * co_drop
    weights = weights.clamp_min(0.0)
    weights = weights / weights.max().clamp_min(1.0)
    weights = 0.10 + 0.90 * weights
    keep_ratio = float(keep_ratio)
    if 0.0 < keep_ratio < 1.0 and weights.numel() > 1:
        keep = max(1, int(round(weights.numel() * keep_ratio)))
        threshold = torch.topk(weights, k=keep).values.min()
        weights = torch.where(weights >= threshold, weights, torch.zeros_like(weights))
    return weights
