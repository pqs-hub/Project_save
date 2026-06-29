"""Report whether heuristic candidate slices retain the labeled next action."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tpi_jepa.bench import parse_bench
from tpi_jepa.graph import build_graph
from tpi_jepa.labels import DEFAULT_LABELS, load_labels, row_to_transition
from tpi_jepa.plan import enumerate_candidates


def _parse_top_k(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _empty_hits(top_ks: list[int]) -> dict[int, int]:
    return {k: 0 for k in top_ks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    parser.add_argument("--candidate-strategy", default="mixed")
    parser.add_argument("--top-k", default="8,16,32")
    parser.add_argument("--max-sequences", type=int, default=256)
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args()

    rows = load_labels(args.labels)
    top_ks = _parse_top_k(args.top_k)
    action_hits = _empty_hits(top_ks)
    node_hits = _empty_hits(top_ks)
    benchmark_action_hits = defaultdict(lambda: _empty_hits(top_ks))
    benchmark_node_hits = defaultdict(lambda: _empty_hits(top_ks))
    benchmark_counts = defaultdict(int)
    type_action_hits = defaultdict(lambda: _empty_hits(top_ks))
    type_node_hits = defaultdict(lambda: _empty_hits(top_ks))
    type_counts = defaultdict(int)
    checked = 0
    graph_cache = {}
    node_set_cache = {}
    for row in rows:
        spec = row_to_transition(row)
        if spec.benchmark_id not in graph_cache:
            graph_cache[spec.benchmark_id] = build_graph(parse_bench(spec.bench_path))
            node_set_cache[spec.benchmark_id] = set(graph_cache[spec.benchmark_id].node_names)
        graph = graph_cache[spec.benchmark_id]
        action = (spec.action_node, spec.action_type)
        if spec.action_node not in node_set_cache[spec.benchmark_id]:
            continue
        for k in top_ks:
            candidates = enumerate_candidates(graph, spec.pre_actions, k, args.candidate_strategy)
            candidate_nodes = {node for node, _ in candidates}
            action_hit = int(action in candidates)
            node_hit = int(spec.action_node in candidate_nodes)
            action_hits[k] += action_hit
            node_hits[k] += node_hit
            benchmark_action_hits[spec.benchmark_id][k] += action_hit
            benchmark_node_hits[spec.benchmark_id][k] += node_hit
            type_action_hits[spec.action_type][k] += action_hit
            type_node_hits[spec.action_type][k] += node_hit
        benchmark_counts[spec.benchmark_id] += 1
        type_counts[spec.action_type] += 1
        checked += 1
        if checked >= args.max_sequences:
            break
    print(f"candidate_strategy={args.candidate_strategy}")
    print(f"checked={checked}")
    for k in top_ks:
        print(f"candidate_recall_at_{k}={action_hits[k] / max(1, checked):.6f}")
        print(f"candidate_action_recall_at_{k}={action_hits[k] / max(1, checked):.6f}")
        print(f"candidate_node_recall_at_{k}={node_hits[k] / max(1, checked):.6f}")
    if args.details:
        for benchmark_id in sorted(benchmark_counts):
            count = benchmark_counts[benchmark_id]
            for k in top_ks:
                action_rate = benchmark_action_hits[benchmark_id][k] / max(1, count)
                node_rate = benchmark_node_hits[benchmark_id][k] / max(1, count)
                print(
                    f"per_benchmark benchmark_id={benchmark_id} k={k} count={count} "
                    f"action_recall={action_rate:.6f} node_recall={node_rate:.6f}"
                )
        for action_type in sorted(type_counts):
            count = type_counts[action_type]
            for k in top_ks:
                action_rate = type_action_hits[action_type][k] / max(1, count)
                node_rate = type_node_hits[action_type][k] / max(1, count)
                print(
                    f"per_action_type action_type={action_type} k={k} count={count} "
                    f"action_recall={action_rate:.6f} node_recall={node_rate:.6f}"
                )


if __name__ == "__main__":
    main()
