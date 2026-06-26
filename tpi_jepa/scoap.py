"""Small SCOAP-style controllability and observability proxies."""

from __future__ import annotations

from pathlib import Path

import torch

from .bench import parse_bench
from .graph import GATE_TYPES, GraphData, build_graph


def _gate_name(graph: GraphData, node: int) -> str:
    """Return the normalized gate name for one node id."""

    return GATE_TYPES[int(graph.gate_type_ids[node].item())]


def _cc_update(gate: str, fin: list[int], cc0: torch.Tensor, cc1: torch.Tensor) -> tuple[float, float]:
    """Compute one approximate cc0/cc1 update from fanin values."""

    if not fin or gate in {"PI", "DFF", "WIRE"}:
        return 1.0, 1.0
    z = cc0[fin]
    o = cc1[fin]
    if gate == "AND":
        return float(z.min().item() + 1.0), float(o.sum().item() + 1.0)
    if gate == "NAND":
        a0, a1 = float(z.min().item() + 1.0), float(o.sum().item() + 1.0)
        return a1, a0
    if gate == "OR":
        return float(z.sum().item() + 1.0), float(o.min().item() + 1.0)
    if gate == "NOR":
        a0, a1 = float(z.sum().item() + 1.0), float(o.min().item() + 1.0)
        return a1, a0
    if gate in {"NOT", "INV"}:
        return float(o[0].item() + 1.0), float(z[0].item() + 1.0)
    if gate in {"BUF", "PO"}:
        return float(z[0].item() + 1.0), float(o[0].item() + 1.0)
    if gate in {"XOR", "XNOR"}:
        cost = float((z + o).mean().item() + 1.0)
        return cost, cost
    return float(z.mean().item() + 1.0), float(o.mean().item() + 1.0)


def _other_fanin_cost(gate: str, fin: list[int], skip: int, cc0: torch.Tensor, cc1: torch.Tensor) -> float:
    """Estimate the side-input cost for observing one fanin through a gate."""

    others = [node for node in fin if node != skip]
    if not others:
        return 0.0
    if gate in {"AND", "NAND"}:
        return float(cc1[others].sum().item())
    if gate in {"OR", "NOR"}:
        return float(cc0[others].sum().item())
    return float(torch.minimum(cc0[others], cc1[others]).sum().item())


def _topological_order(graph: GraphData) -> list[int]:
    """Return a fanin-before-fanout order when the graph is acyclic."""

    indegree = [len(fanins) for fanins in graph.fanin_lists]
    queue = [idx for idx, degree in enumerate(indegree) if degree == 0]
    order: list[int] = []
    cursor = 0
    while cursor < len(queue):
        node = queue[cursor]
        cursor += 1
        order.append(node)
        for dst in graph.fanout_lists[node]:
            indegree[dst] -= 1
            if indegree[dst] == 0:
                queue.append(dst)
    if len(order) == graph.num_nodes:
        return order
    return list(range(graph.num_nodes))


def compute_scoap_proxy(graph: GraphData) -> torch.Tensor:
    """Return normalized `[cc0, cc1, co]` features."""

    n = graph.num_nodes
    cc0 = torch.ones(n, dtype=torch.float32)
    cc1 = torch.ones(n, dtype=torch.float32)
    order = _topological_order(graph)

    for node in order:
        gate = _gate_name(graph, node)
        v0, v1 = _cc_update(gate, graph.fanin_lists[node], cc0, cc1)
        cc0[node] = min(v0, 1_000.0)
        cc1[node] = min(v1, 1_000.0)

    co = torch.full((n,), 1_000.0, dtype=torch.float32)
    co[graph.output_mask] = 0.0
    for dst in reversed(order):
        gate = _gate_name(graph, dst)
        dst_co = float(co[dst].item())
        for src in graph.fanin_lists[dst]:
            side = _other_fanin_cost(gate, graph.fanin_lists[dst], src, cc0, cc1)
            co[src] = min(float(co[src].item()), dst_co + side + 1.0)

    raw = torch.stack(
        [
            cc0,
            cc1,
            co.clamp_max(1_000.0),
        ],
        dim=1,
    )
    raw = torch.log1p(raw.clamp_min(0.0))
    scale = raw.abs().amax(dim=0).clamp_min(1.0)
    return raw / scale


def _main() -> None:
    """CLI summary used by the step-by-step validation plan."""

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("bench")
    args = parser.parse_args()

    graph = build_graph(parse_bench(Path(args.bench)))
    x = compute_scoap_proxy(graph)
    print(f"scoap_shape={tuple(x.shape)}")
    print(f"finite={bool(torch.isfinite(x).all().item())}")
    print(f"nonzero={bool((x.abs().sum() > 0).item())}")
    print(f"cc0_mean={float(x[:, 0].mean().item()):.6f}")
    print(f"cc1_mean={float(x[:, 1].mean().item()):.6f}")
    print(f"co_mean={float(x[:, 2].mean().item()):.6f}")


if __name__ == "__main__":
    _main()
