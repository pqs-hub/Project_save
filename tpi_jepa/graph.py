"""Build tensor-friendly graph data from parsed BENCH circuits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import deque

import torch

from .bench import Circuit, parse_bench


GATE_TYPES = [
    "PI",
    "PO",
    "DFF",
    "AND",
    "OR",
    "NAND",
    "NOR",
    "XOR",
    "XNOR",
    "NOT",
    "INV",
    "BUF",
    "MUX",
    "CONST0",
    "CONST1",
    "LUT",
    "WIRE",
    "OTHER",
]
GATE_TO_ID = {gate: idx for idx, gate in enumerate(GATE_TYPES)}


@dataclass
class GraphData:
    """A circuit graph with directed edges from fanin nodes to fanout nodes."""

    node_names: list[str]
    gate_type_ids: torch.Tensor
    edge_src: torch.Tensor
    edge_dst: torch.Tensor
    fanin_lists: list[list[int]]
    fanout_lists: list[list[int]]
    input_mask: torch.Tensor
    output_mask: torch.Tensor

    @property
    def num_nodes(self) -> int:
        """Return the number of graph nodes."""

        return len(self.node_names)


def _gate_id(gate: str) -> int:
    """Map raw BENCH gate names to a compact integer id."""

    gate = gate.upper()
    if gate == "BUFF":
        gate = "BUF"
    if gate == "NOT":
        gate = "NOT"
    return GATE_TO_ID.get(gate, GATE_TO_ID["OTHER"])


def build_graph(circuit: Circuit, dff_as_comb_boundary: bool = True) -> GraphData:
    """Convert a parsed circuit into integer ids and directed edge tensors.

    When `dff_as_comb_boundary` is true, each `Q = DFF(D)` is treated as a
    scan-style combinational boundary: `Q` becomes a PI-like source, `D`
    becomes a PO-like sink, and the sequential `D -> Q` edge is omitted.
    """

    name_to_id = {name: idx for idx, name in enumerate(circuit.node_names)}
    n = len(circuit.node_names)
    fanin_lists: list[list[int]] = [[] for _ in range(n)]
    fanout_lists: list[list[int]] = [[] for _ in range(n)]
    edge_src: list[int] = []
    edge_dst: list[int] = []
    dff_outputs: set[str] = set()
    dff_inputs: set[str] = set()

    if dff_as_comb_boundary:
        for name in circuit.node_names:
            if circuit.gate_types.get(name, "").upper() != "DFF":
                continue
            dff_outputs.add(name)
            dff_inputs.update(src for src in circuit.fanins.get(name, []) if src in name_to_id)

    for dst_name in circuit.node_names:
        if dst_name in dff_outputs:
            continue
        dst = name_to_id[dst_name]
        for src_name in circuit.fanins.get(dst_name, []):
            if src_name not in name_to_id:
                continue
            src = name_to_id[src_name]
            fanin_lists[dst].append(src)
            fanout_lists[src].append(dst)
            edge_src.append(src)
            edge_dst.append(dst)

    gate_ids = [
        _gate_id("PI" if name in dff_outputs else circuit.gate_types.get(name, "OTHER"))
        for name in circuit.node_names
    ]
    input_set = set(circuit.inputs) | dff_outputs
    output_set = set(circuit.outputs) | dff_inputs
    input_mask = torch.tensor([name in input_set for name in circuit.node_names], dtype=torch.bool)
    output_mask = torch.tensor([name in output_set for name in circuit.node_names], dtype=torch.bool)

    return GraphData(
        node_names=circuit.node_names,
        gate_type_ids=torch.tensor(gate_ids, dtype=torch.long),
        edge_src=torch.tensor(edge_src, dtype=torch.long),
        edge_dst=torch.tensor(edge_dst, dtype=torch.long),
        fanin_lists=fanin_lists,
        fanout_lists=fanout_lists,
        input_mask=input_mask,
        output_mask=output_mask,
    )


def _bounded_reach_count(starts: list[int], adjacency: list[list[int]], max_depth: int = 4) -> int:
    """Count nearby reachable nodes up to a small depth for a cheap proxy."""

    seen: set[int] = set()
    queue = deque((node, 1) for node in starts)
    while queue:
        node, depth = queue.popleft()
        if node in seen or depth > max_depth:
            continue
        seen.add(node)
        for nxt in adjacency[node]:
            queue.append((nxt, depth + 1))
    return len(seen)


def _multi_source_distances(starts: list[int], adjacency: list[list[int]]) -> list[int]:
    """Compute shortest unweighted distances from any source node."""

    dist = [-1 for _ in adjacency]
    queue = deque()
    for start in starts:
        if 0 <= start < len(adjacency) and dist[start] < 0:
            dist[start] = 0
            queue.append(start)
    while queue:
        node = queue.popleft()
        for nxt in adjacency[node]:
            if dist[nxt] < 0:
                dist[nxt] = dist[node] + 1
                queue.append(nxt)
    return dist


def compute_structural_features(graph: GraphData) -> torch.Tensor:
    """Compute six simple structural features for every graph node."""

    input_starts = [idx for idx, is_input in enumerate(graph.input_mask.tolist()) if is_input]
    output_starts = [idx for idx, is_output in enumerate(graph.output_mask.tolist()) if is_output]
    dist_from_input = _multi_source_distances(input_starts, graph.fanout_lists)
    dist_to_output = _multi_source_distances(output_starts, graph.fanin_lists)
    unreachable = graph.num_nodes + 1
    rows: list[list[float]] = []
    for idx in range(graph.num_nodes):
        fanin_count = len(graph.fanin_lists[idx])
        fanout_count = len(graph.fanout_lists[idx])
        forward_proxy = _bounded_reach_count(graph.fanout_lists[idx], graph.fanout_lists)
        backward_proxy = _bounded_reach_count(graph.fanin_lists[idx], graph.fanin_lists)
        input_distance = dist_from_input[idx] if dist_from_input[idx] >= 0 else unreachable
        output_distance = dist_to_output[idx] if dist_to_output[idx] >= 0 else unreachable
        rows.append(
            [
                float(fanin_count),
                float(fanout_count),
                float(input_distance),
                float(output_distance),
                float(forward_proxy),
                float(backward_proxy),
            ]
        )
    x = torch.tensor(rows, dtype=torch.float32)
    scale = x.abs().amax(dim=0).clamp_min(1.0)
    return x / scale


def _main() -> None:
    """CLI summary used by the step-by-step validation plan."""

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("bench")
    args = parser.parse_args()

    graph = build_graph(parse_bench(Path(args.bench)))
    features = compute_structural_features(graph)
    print(f"nodes={graph.num_nodes}")
    print(f"edges={graph.edge_src.numel()}")
    print(f"structural_shape={tuple(features.shape)}")
    print(f"finite={bool(torch.isfinite(features).all().item())}")


if __name__ == "__main__":
    _main()
