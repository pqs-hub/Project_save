"""Feature builders for circuit states and candidate actions."""

from __future__ import annotations

import csv
from functools import lru_cache
import json
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F

from .bench import parse_bench
from .graph import GATE_TYPES, GraphData, build_graph, compute_structural_features
from .scoap import compute_scoap_proxy
from .testability import compute_testability_region_features


ACTION_TYPES = ["control0", "control1", "observe"]
ACTION_TO_ID = {name: idx for idx, name in enumerate(ACTION_TYPES)}
STRUCTURAL_DIM = 6
SCOAP_DIM = 3
ACTION_MASK_DIM = len(ACTION_TYPES)
SCOAP_START = len(GATE_TYPES) + STRUCTURAL_DIM
SCOAP_END = SCOAP_START + SCOAP_DIM
REAL_FAULT_FEATURE_DIM = 3
TYPED_REAL_FAULT_FEATURE_DIM = 2
ACTIVATION_FEATURE_DIM = 3


def action_type_to_id(action_type: str) -> int:
    """Map canonical action type text to a stable integer id."""

    key = action_type.lower()
    if key not in ACTION_TO_ID:
        raise ValueError(f"Unsupported action type: {action_type!r}")
    return ACTION_TO_ID[key]


def _action_node_id(graph: GraphData, node: int | str) -> int:
    """Accept either a node id or a node name and return the integer id."""

    if isinstance(node, int):
        if node < 0 or node >= graph.num_nodes:
            raise ValueError(f"Action node id out of range: {node}")
        return node
    try:
        return graph.node_names.index(node)
    except ValueError as exc:
        raise ValueError(f"Action node name not found in graph: {node!r}") from exc


@lru_cache(maxsize=8)
def _load_real_fault_priors(path_text: str) -> dict[str, dict[str, tuple[float, float, float]]]:
    path = Path(path_text)
    priors: dict[str, dict[str, tuple[float, float, float]]] = {}
    if not path.exists():
        raise FileNotFoundError(f"real fault prior file not found: {path}")
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        rows = data if isinstance(data, list) else data.get("rows", [])
    else:
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
    for row in rows:
        benchmark_id = str(row.get("benchmark_id", "")).strip()
        net = str(row.get("net", "")).strip()
        if not benchmark_id or not net:
            continue
        fault_count = float(row.get("fault_count") or 0.0)
        hard_count = float(row.get("hard_fault_count") or 0.0)
        ratio = float(row.get("hard_fault_ratio") or 0.0)
        priors.setdefault(benchmark_id, {})[net] = (ratio, hard_count, fault_count)
    return priors


def make_real_fault_features(
    graph: GraphData,
    benchmark_id: str | None,
    real_fault_prior_path: str | Path | None,
) -> torch.Tensor:
    """Return per-node features from real TMAX fault logs when available."""

    features = torch.zeros((graph.num_nodes, REAL_FAULT_FEATURE_DIM), dtype=torch.float32)
    if not benchmark_id or not real_fault_prior_path:
        return features
    priors = _load_real_fault_priors(str(real_fault_prior_path)).get(str(benchmark_id), {})
    if not priors:
        return features
    for idx, name in enumerate(graph.node_names):
        ratio, hard_count, fault_count = priors.get(name, (0.0, 0.0, 0.0))
        features[idx, 0] = float(ratio)
        features[idx, 1] = float(hard_count)
        features[idx, 2] = float(fault_count)
    if features[:, 1].max() > 0:
        features[:, 1] = torch.log1p(features[:, 1]) / torch.log1p(features[:, 1].max())
    if features[:, 2].max() > 0:
        features[:, 2] = torch.log1p(features[:, 2]) / torch.log1p(features[:, 2].max())
    return features.clamp(0.0, 1.0)


@lru_cache(maxsize=8)
def _load_typed_real_fault_priors(path_text: str) -> dict[str, dict[str, tuple[float, float]]]:
    """Read optional SA0/SA1 hard-fault counts without changing model feature width."""

    path = Path(path_text)
    priors: dict[str, dict[str, tuple[float, float]]] = {}
    if not path.exists():
        raise FileNotFoundError(f"real fault prior file not found: {path}")
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        rows = data if isinstance(data, list) else data.get("rows", [])
    else:
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
    for row in rows:
        benchmark_id = str(row.get("benchmark_id", "")).strip()
        net = str(row.get("net", "")).strip()
        if not benchmark_id or not net:
            continue
        sa0 = float(row.get("sa0_hard_fault_count") or 0.0)
        sa1 = float(row.get("sa1_hard_fault_count") or 0.0)
        priors.setdefault(benchmark_id, {})[net] = (sa0, sa1)
    return priors


def make_typed_real_fault_features(
    graph: GraphData,
    benchmark_id: str | None,
    real_fault_prior_path: str | Path | None,
) -> torch.Tensor:
    """Return normalized per-node hard SA0/SA1 counts for heuristic planning only."""

    features = torch.zeros((graph.num_nodes, TYPED_REAL_FAULT_FEATURE_DIM), dtype=torch.float32)
    if not benchmark_id or not real_fault_prior_path:
        return features
    priors = _load_typed_real_fault_priors(str(real_fault_prior_path)).get(str(benchmark_id), {})
    if not priors:
        return features
    for idx, name in enumerate(graph.node_names):
        features[idx] = torch.tensor(priors.get(name, (0.0, 0.0)), dtype=torch.float32)
    for column in range(TYPED_REAL_FAULT_FEATURE_DIM):
        maximum = features[:, column].max()
        if maximum > 0:
            features[:, column] = torch.log1p(features[:, column]) / torch.log1p(maximum)
    return features.clamp(0.0, 1.0)


@lru_cache(maxsize=8)
def _load_activation_priors(path_text: str) -> dict[str, dict[str, tuple[float, float, float]]]:
    path = Path(path_text)
    priors: dict[str, dict[str, tuple[float, float, float]]] = {}
    if not path.exists():
        raise FileNotFoundError(f"activation prior file not found: {path}")
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        rows = data if isinstance(data, list) else data.get("rows", [])
    else:
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
    for row in rows:
        benchmark_id = str(row.get("benchmark_id", "")).strip()
        net = str(row.get("net", "")).strip()
        if not benchmark_id or not net:
            continue
        p_one = float(row.get("p_one") or row.get("sa0_activation_prob") or 0.0)
        p_zero = float(row.get("p_zero") or row.get("sa1_activation_prob") or (1.0 - p_one))
        rare = float(row.get("activation_min_prob") or min(p_one, p_zero))
        bias = float(row.get("activation_bias") or abs(p_one - p_zero))
        priors.setdefault(benchmark_id, {})[net] = (p_one, rare, bias)
    return priors


def make_activation_features(
    graph: GraphData,
    benchmark_id: str | None,
    activation_prior_path: str | Path | None,
) -> torch.Tensor:
    """Return per-node random-pattern activation features when available.

    Columns are P(net=1), min(P(net=1), P(net=0)), and abs(P(net=1)-P(net=0)).
    The min-probability column is the useful scalar for stuck-at activation
    hardness: averaging sa0 and sa1 activation would always be 0.5.
    """

    features = torch.zeros((graph.num_nodes, ACTIVATION_FEATURE_DIM), dtype=torch.float32)
    if not benchmark_id or not activation_prior_path:
        return features
    priors = _load_activation_priors(str(activation_prior_path)).get(str(benchmark_id), {})
    if not priors:
        return features
    for idx, name in enumerate(graph.node_names):
        features[idx] = torch.tensor(priors.get(name, (0.0, 0.0, 0.0)), dtype=torch.float32)
    return features.clamp(0.0, 1.0)


def make_base_node_features(
    graph: GraphData,
    feature_mode: str = "basic",
    benchmark_id: str | None = None,
    real_fault_prior_path: str | Path | None = None,
    activation_prior_path: str | Path | None = None,
) -> torch.Tensor:
    """Build action-independent node features for one graph."""

    gate_one_hot = F.one_hot(graph.gate_type_ids, num_classes=len(GATE_TYPES)).float()
    structural = compute_structural_features(graph)
    scoap = compute_scoap_proxy(graph)
    base = torch.cat([gate_one_hot, structural, scoap], dim=1)
    mode = (feature_mode or "basic").lower()
    real_fault = make_real_fault_features(graph, benchmark_id, real_fault_prior_path)
    activation = make_activation_features(graph, benchmark_id, activation_prior_path)
    use_real_fault = bool(real_fault_prior_path)
    use_activation = bool(activation_prior_path)
    extras = []
    if use_real_fault:
        extras.append(real_fault)
    if use_activation:
        extras.append(activation)
    if mode in {"basic", "base", "none"}:
        return torch.cat([base, *extras], dim=1) if extras else base
    if mode in {"region", "testability", "full", "tac"}:
        parts = [base, compute_testability_region_features(graph, base)]
        parts.extend(extras)
        return torch.cat(parts, dim=1)
    raise ValueError(f"Unsupported feature_mode: {feature_mode!r}")


def make_state_features(
    graph: GraphData,
    inserted_actions: list[tuple[int | str, str]],
    base_features: torch.Tensor | None = None,
    feature_mode: str = "basic",
    benchmark_id: str | None = None,
    real_fault_prior_path: str | Path | None = None,
    activation_prior_path: str | Path | None = None,
    state_update_mode: str = "static",
    update_depth: int = 8,
) -> torch.Tensor:
    """Build node features for a graph plus the current inserted-testpoint state."""

    if base_features is None:
        base_features = make_base_node_features(
            graph,
            feature_mode,
            benchmark_id=benchmark_id,
            real_fault_prior_path=real_fault_prior_path,
            activation_prior_path=activation_prior_path,
        )
    mode = (state_update_mode or "static").lower()
    if mode in {"static", "mask", "none"}:
        state_base = base_features
    elif mode in {"proxy", "scoap_proxy", "action_proxy"}:
        state_base = apply_action_scoap_proxy_updates(graph, base_features, inserted_actions, update_depth)
    else:
        raise ValueError(f"Unsupported state_update_mode: {state_update_mode!r}")

    masks = torch.zeros((graph.num_nodes, ACTION_MASK_DIM), dtype=torch.float32)

    for node, action_type in inserted_actions:
        node_id = _action_node_id(graph, node)
        action_id = action_type_to_id(action_type)
        masks[node_id, action_id] = 1.0

    return torch.cat([state_base, masks], dim=1)


def apply_action_scoap_proxy_updates(
    graph: GraphData,
    base_features: torch.Tensor,
    inserted_actions: list[tuple[int | str, str]],
    max_depth: int = 8,
) -> torch.Tensor:
    """Return base features with action-conditioned SCOAP proxy updates.

    This is a conservative proxy update, not a patched-netlist simulation:
    CP0/CP1 reduce the selected node's controllability and attenuate through
    the fanout cone, while OP reduces observability through the fanin cone.
    The original graph, edges, and non-SCOAP feature columns are unchanged.
    """

    if not inserted_actions:
        return base_features
    updated = base_features.clone()
    scoap = updated[:, SCOAP_START:SCOAP_END]
    for node, action_type in inserted_actions:
        node_id = _action_node_id(graph, node)
        action_id = action_type_to_id(action_type)
        if action_id == ACTION_TO_ID["observe"]:
            distances = _distances_from(node_id, graph.fanin_lists, max_depth)
            column = 2
        else:
            distances = _distances_from(node_id, graph.fanout_lists, max_depth)
            column = 0 if action_id == ACTION_TO_ID["control0"] else 1
        for target, distance in distances.items():
            strength = 1.0 / (1.0 + float(distance))
            factor = 1.0 - 0.75 * strength
            scoap[target, column] = torch.minimum(
                scoap[target, column],
                scoap[target, column] * factor,
            )
    updated[:, SCOAP_START:SCOAP_END] = scoap.clamp(0.0, 1.0)
    return updated


def _distances_from(start: int, adjacency: list[list[int]], max_depth: int = 8) -> dict[int, int]:
    """Compute short unweighted distances from one node in one direction."""

    dist = {start: 0}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if dist[node] >= max_depth:
            continue
        for nxt in adjacency[node]:
            if nxt not in dist:
                dist[nxt] = dist[node] + 1
                queue.append(nxt)
    return dist


def make_action_relation_features(
    graph: GraphData,
    action_node_id: int,
    relation_mode: str = "basic",
    max_depth: int = 8,
) -> torch.Tensor:
    """Build relation features from every node to the candidate action node."""

    if action_node_id < 0 or action_node_id >= graph.num_nodes:
        raise ValueError(f"Action node id out of range: {action_node_id}")

    fanout_dist = _distances_from(action_node_id, graph.fanout_lists, max_depth)
    fanin_dist = _distances_from(action_node_id, graph.fanin_lists, max_depth)
    mode = (relation_mode or "basic").lower()
    fanin_scale = max(1.0, max((len(items) for items in graph.fanin_lists), default=1))
    fanout_scale = max(1.0, max((len(items) for items in graph.fanout_lists), default=1))
    rows: list[list[float]] = []
    for node in range(graph.num_nodes):
        is_action = float(node == action_node_id)
        is_fanin_side = float(node in fanin_dist and node != action_node_id)
        is_fanout_side = float(node in fanout_dist and node != action_node_id)
        distances = []
        if node in fanin_dist:
            distances.append(fanin_dist[node])
        if node in fanout_dist:
            distances.append(fanout_dist[node])
        best_dist = min(distances) if distances else 999
        dist_proxy = 1.0 / (1.0 + float(best_dist)) if best_dist != 999 else 0.0
        row = [is_action, is_fanin_side, is_fanout_side, dist_proxy]
        if mode in {"cone", "testability", "full", "tac"}:
            fanin_proxy = 1.0 / (1.0 + float(fanin_dist[node])) if node in fanin_dist else 0.0
            fanout_proxy = 1.0 / (1.0 + float(fanout_dist[node])) if node in fanout_dist else 0.0
            in_action_cone = float(node in fanin_dist or node in fanout_dist)
            fanin_norm = float(len(graph.fanin_lists[node])) / fanin_scale
            fanout_norm = float(len(graph.fanout_lists[node])) / fanout_scale
            is_reconvergent = float(len(graph.fanin_lists[node]) > 1 or len(graph.fanout_lists[node]) > 1)
            is_input_boundary = float(bool(graph.input_mask[node].item()) and node in fanin_dist)
            is_output_boundary = float(bool(graph.output_mask[node].item()) and node in fanout_dist)
            row.extend(
                [
                    fanin_proxy,
                    fanout_proxy,
                    in_action_cone,
                    fanin_norm,
                    fanout_norm,
                    is_reconvergent,
                    is_input_boundary,
                    is_output_boundary,
                ]
            )
        elif mode not in {"basic", "base", "none"}:
            raise ValueError(f"Unsupported relation_mode: {relation_mode!r}")
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32)


def _main() -> None:
    """CLI summary used by the step-by-step validation plan."""

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("bench")
    args = parser.parse_args()

    graph = build_graph(parse_bench(Path(args.bench)))
    node_id = next((idx for idx, is_input in enumerate(graph.input_mask.tolist()) if not is_input), 0)
    x_pre = make_state_features(graph, [])
    x_post = make_state_features(graph, [(node_id, "control1")])
    rel = make_action_relation_features(graph, node_id)
    print(f"x_pre_shape={tuple(x_pre.shape)}")
    print(f"x_post_shape={tuple(x_post.shape)}")
    print(f"changed_mask={float((x_post[:, -3:] - x_pre[:, -3:]).abs().sum().item()):.1f}")
    print(f"relation_shape={tuple(rel.shape)}")
    print(f"finite={bool(torch.isfinite(x_post).all().item() and torch.isfinite(rel).all().item())}")


if __name__ == "__main__":
    _main()
