"""Estimate per-node stuck-at activation priors with random good-circuit simulation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tpi_jepa.bench import parse_bench  # noqa: E402
from tpi_jepa.graph import GATE_TYPES, build_graph  # noqa: E402
from tpi_jepa.labels import DEFAULT_LABELS, find_bench_path, load_labels  # noqa: E402
from tpi_jepa.protocol import eval_benchmarks_from_protocol, parse_benchmark_list  # noqa: E402


def benchmark_ids(args: argparse.Namespace) -> list[str]:
    if args.benchmarks:
        return sorted(parse_benchmark_list(args.benchmarks))
    rows = load_labels(args.labels)
    benches = {row.benchmark_id for row in rows}
    if args.eval_protocol and not args.include_eval:
        benches -= eval_benchmarks_from_protocol(args.eval_protocol)
    benches -= parse_benchmark_list(args.extra_exclude)
    return sorted(benches)


def topo_order(graph) -> list[int]:
    indegree = [len(items) for items in graph.fanin_lists]
    queue = [idx for idx, deg in enumerate(indegree) if deg == 0]
    order: list[int] = []
    cursor = 0
    while cursor < len(queue):
        node = queue[cursor]
        cursor += 1
        order.append(node)
        for nxt in graph.fanout_lists[node]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    if len(order) < graph.num_nodes:
        seen = set(order)
        order.extend(idx for idx in range(graph.num_nodes) if idx not in seen)
    return order


def _fanin_values(values: torch.Tensor, fanins: list[int]) -> torch.Tensor:
    return values[torch.tensor(fanins, dtype=torch.long, device=values.device)]


def simulate_batch(graph, order: list[int], batch_size: int, generator: torch.Generator, device: torch.device) -> torch.Tensor:
    values = torch.zeros((graph.num_nodes, batch_size), dtype=torch.bool, device=device)
    input_ids = torch.nonzero(graph.input_mask, as_tuple=False).flatten().to(device)
    if input_ids.numel():
        values[input_ids] = torch.rand(
            (int(input_ids.numel()), batch_size),
            generator=generator,
            device=device,
        ) < 0.5

    for node in order:
        if bool(graph.input_mask[node].item()):
            continue
        gate = GATE_TYPES[int(graph.gate_type_ids[node].item())]
        fanins = graph.fanin_lists[node]
        if gate == "CONST0":
            values[node] = False
        elif gate == "CONST1":
            values[node] = True
        elif not fanins:
            continue
        elif gate in {"BUF", "WIRE", "PO"}:
            values[node] = values[fanins[0]]
        elif gate in {"NOT", "INV"}:
            values[node] = ~values[fanins[0]]
        elif gate == "AND":
            values[node] = _fanin_values(values, fanins).all(dim=0)
        elif gate == "NAND":
            values[node] = ~_fanin_values(values, fanins).all(dim=0)
        elif gate == "OR":
            values[node] = _fanin_values(values, fanins).any(dim=0)
        elif gate == "NOR":
            values[node] = ~_fanin_values(values, fanins).any(dim=0)
        elif gate == "XOR":
            values[node] = _fanin_values(values, fanins).sum(dim=0).remainder(2).bool()
        elif gate == "XNOR":
            values[node] = ~_fanin_values(values, fanins).sum(dim=0).remainder(2).bool()
        elif gate == "MUX" and len(fanins) >= 3:
            sel = values[fanins[-1]]
            values[node] = torch.where(sel, values[fanins[1]], values[fanins[0]])
        else:
            values[node] = values[fanins[0]]
    return values


@torch.no_grad()
def activation_rows_for_benchmark(
    benchmark_id: str,
    *,
    patterns: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    graph = build_graph(parse_bench(find_bench_path(benchmark_id)))
    order = topo_order(graph)
    ones = torch.zeros(graph.num_nodes, dtype=torch.float64, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    remaining = int(patterns)
    while remaining > 0:
        current = min(int(batch_size), remaining)
        values = simulate_batch(graph, order, current, generator, device)
        ones += values.sum(dim=1, dtype=torch.float64)
        remaining -= current

    p_one = (ones / float(patterns)).cpu()
    rows: list[dict[str, Any]] = []
    for idx, net in enumerate(graph.node_names):
        one = float(p_one[idx].item())
        zero = 1.0 - one
        rows.append(
            {
                "benchmark_id": benchmark_id,
                "net": net,
                "patterns": patterns,
                "p_one": one,
                "p_zero": zero,
                "sa0_activation_prob": one,
                "sa1_activation_prob": zero,
                "activation_min_prob": min(one, zero),
                "activation_bias": abs(one - zero),
            }
        )
    return rows


FIELDS = [
    "benchmark_id",
    "net",
    "patterns",
    "p_one",
    "p_zero",
    "sa0_activation_prob",
    "sa1_activation_prob",
    "activation_min_prob",
    "activation_bias",
]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--eval-protocol", default="configs/eval_protocol_coverage_only.json")
    parser.add_argument("--include-eval", action="store_true")
    parser.add_argument("--extra-exclude", default="")
    parser.add_argument("--benchmarks", default="")
    parser.add_argument("--patterns", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    all_rows: list[dict[str, Any]] = []
    benches = benchmark_ids(args)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
    for offset, benchmark_id in enumerate(benches):
        print(json.dumps({"benchmark_id": benchmark_id, "status": "started"}, sort_keys=True), flush=True)
        rows = activation_rows_for_benchmark(
            benchmark_id,
            patterns=args.patterns,
            batch_size=args.batch_size,
            seed=args.seed + 1009 * offset,
            device=device,
        )
        all_rows.extend(rows)
        with args.out_csv.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writerows(rows)
        print(json.dumps({"benchmark_id": benchmark_id, "rows": len(rows)}, sort_keys=True), flush=True)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(all_rows, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"benchmarks": len(benches), "rows": len(all_rows), "out_csv": str(args.out_csv)}, indent=2))


if __name__ == "__main__":
    main()
