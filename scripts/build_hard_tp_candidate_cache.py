"""Build backend-valid TP candidate caches ranked by current hard-fault cones."""

from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TPI_EVAL_ROOT = Path("/data4/pengqingsong/DFT/Dataset")
if str(TPI_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(TPI_EVAL_ROOT))

from tpi_eval.bench import parse_bench as parse_eval_bench  # noqa: E402
from tpi_eval.candidates import generate_candidates  # noqa: E402
from tpi_jepa.bench import parse_bench  # noqa: E402
from tpi_jepa.graph import build_graph  # noqa: E402
from tpi_jepa.labels import find_bench_path  # noqa: E402
from tpi_jepa.scoap import compute_scoap_proxy  # noqa: E402

ACTION_TYPES = ("control0", "control1", "observe")


def parse_csv_values(text: str) -> list[str]:
    """Parse comma-separated command-line values."""

    return [item.strip() for item in text.split(",") if item.strip()]


def discover_baseline_file(root: Path | None, benchmark_id: str, filename: str) -> Path | None:
    """Find the newest baseline artifact for one benchmark under an eval root."""

    if root is None:
        return None
    matches = list(root.glob(f"**/{benchmark_id}/baseline/{filename}"))
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def formatted_path(template: str | None, benchmark_id: str) -> Path | None:
    """Resolve a format-string path template if provided."""

    if not template:
        return None
    return Path(template.format(benchmark_id=benchmark_id))


def load_hard_fault_weights(nodes_csv: Path | None, faults_csv: Path | None) -> tuple[dict[str, dict[str, float]], str]:
    """Load hard-fault weights keyed by net and fault polarity."""

    weights: dict[str, dict[str, float]] = {}
    source = "none"
    if nodes_csv and nodes_csv.exists():
        with nodes_csv.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                node = (row.get("node") or "").strip()
                if not node:
                    continue
                entry = weights.setdefault(node, {"total": 0.0, "sa0": 0.0, "sa1": 0.0})
                entry["sa0"] += float(row.get("output_sa0") or 0.0) + float(row.get("input_edge_sa0") or 0.0)
                entry["sa1"] += float(row.get("output_sa1") or 0.0) + float(row.get("input_edge_sa1") or 0.0)
                entry["total"] += float(row.get("total_undetected_faults") or 0.0)
        source = str(nodes_csv)
    elif faults_csv and faults_csv.exists():
        with faults_csv.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row.get("is_hard") or "1").strip() in {"0", "false", "False"}:
                    continue
                node = (row.get("node") or row.get("net") or row.get("dst_node") or row.get("src_node") or "").strip()
                if not node:
                    continue
                entry = weights.setdefault(node, {"total": 0.0, "sa0": 0.0, "sa1": 0.0})
                sa_value = str(row.get("sa_value") or "").strip()
                if sa_value == "0":
                    entry["sa0"] += 1.0
                elif sa_value == "1":
                    entry["sa1"] += 1.0
                entry["total"] += 1.0
        source = str(faults_csv)
    return weights, source


def distance_map(start: int, adjacency: list[list[int]], max_depth: int) -> dict[int, int]:
    """Return bounded graph distances from one source."""

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


def normalize(values: torch.Tensor) -> torch.Tensor:
    """Normalize a nonnegative tensor by its max."""

    return values / values.max().clamp_min(1.0)


def hard_cone_action_scores(
    benchmark_id: str,
    fanin_depth: int,
    fanout_depth: int,
    max_hard_nodes: int,
    hard_weights: dict[str, dict[str, float]],
) -> tuple[dict[str, dict[str, float]], dict[str, int | float]]:
    """Score every graph node/action from hard-fault cones and SCOAP."""

    graph = build_graph(parse_bench(find_bench_path(benchmark_id)))
    node_to_id = {name: idx for idx, name in enumerate(graph.node_names)}
    scoap = compute_scoap_proxy(graph)
    cc0 = scoap[:, 0]
    cc1 = scoap[:, 1]
    co = scoap[:, 2]
    fanout = torch.tensor([len(items) for items in graph.fanout_lists], dtype=torch.float32)
    fanout = normalize(fanout)

    seeds = [
        (node_to_id[node], weights)
        for node, weights in hard_weights.items()
        if node in node_to_id and float(weights.get("total") or 0.0) > 0.0
    ]
    seeds.sort(key=lambda item: float(item[1].get("total") or 0.0), reverse=True)
    seeds = seeds[: max(1, max_hard_nodes)]

    fanin_sa0 = torch.zeros(graph.num_nodes, dtype=torch.float32)
    fanin_sa1 = torch.zeros(graph.num_nodes, dtype=torch.float32)
    fanin_total = torch.zeros(graph.num_nodes, dtype=torch.float32)
    fanout_total = torch.zeros(graph.num_nodes, dtype=torch.float32)
    direct_total = torch.zeros(graph.num_nodes, dtype=torch.float32)

    max_total = max((float(weights.get("total") or 0.0) for _, weights in seeds), default=1.0)
    for hard_id, weights in seeds:
        total = float(weights.get("total") or 0.0) / max_total
        sa0 = float(weights.get("sa0") or 0.0) / max_total
        sa1 = float(weights.get("sa1") or 0.0) / max_total
        direct_total[hard_id] += total

        for node, dist in distance_map(hard_id, graph.fanin_lists, fanin_depth).items():
            decay = 1.0 / float(dist + 1)
            fanin_total[node] += total * decay
            fanin_sa0[node] += sa0 * decay
            fanin_sa1[node] += sa1 * decay
        for node, dist in distance_map(hard_id, graph.fanout_lists, fanout_depth).items():
            decay = 1.0 / float(dist + 1)
            fanout_total[node] += total * decay

    fanin_sa0 = normalize(fanin_sa0)
    fanin_sa1 = normalize(fanin_sa1)
    fanin_total = normalize(fanin_total)
    fanout_total = normalize(fanout_total)
    direct_total = normalize(direct_total)

    control0 = normalize(1.15 * fanin_sa1 + 0.55 * fanin_total + 0.25 * direct_total + 0.30 * cc0 + 0.10 * fanout)
    control1 = normalize(1.15 * fanin_sa0 + 0.55 * fanin_total + 0.25 * direct_total + 0.30 * cc1 + 0.10 * fanout)
    observe = normalize(1.35 * fanout_total + 0.45 * direct_total + 0.35 * co)

    scores = {}
    for name, idx in node_to_id.items():
        scores[name] = {
            "control0": float(control0[idx].item()),
            "control1": float(control1[idx].item()),
            "observe": float(observe[idx].item()),
        }
    stats = {
        "hard_seed_count": len(seeds),
        "hard_node_count": len(hard_weights),
        "max_hard_node_weight": max_total,
    }
    return scores, stats


def build_cache(
    benchmark_id: str,
    hard_fault_root: Path | None,
    undetected_nodes_template: str | None,
    faults_template: str | None,
    fanin_depth: int,
    fanout_depth: int,
    max_hard_nodes: int,
) -> dict:
    """Build one benchmark cache payload."""

    bench_path = find_bench_path(benchmark_id)
    eval_circuit = parse_eval_bench(bench_path)
    base_candidates = generate_candidates(eval_circuit)
    nodes_csv = formatted_path(undetected_nodes_template, benchmark_id) or discover_baseline_file(
        hard_fault_root, benchmark_id, "undetected_nodes.csv"
    )
    faults_csv = formatted_path(faults_template, benchmark_id) or discover_baseline_file(
        hard_fault_root, benchmark_id, "faults_undetected.csv"
    )
    hard_weights, hard_source = load_hard_fault_weights(nodes_csv, faults_csv)
    action_scores, stats = hard_cone_action_scores(
        benchmark_id,
        fanin_depth,
        fanout_depth,
        max_hard_nodes,
        hard_weights,
    )

    candidates = []
    for candidate in base_candidates:
        scores = action_scores.get(candidate.net, {action_type: 0.0 for action_type in ACTION_TYPES})
        priority = max(float(scores.get(action_type, 0.0)) for action_type in ACTION_TYPES)
        row = candidate.to_json()
        row["priority"] = priority
        row["score"] = priority
        row["action_scores"] = {action_type: float(scores.get(action_type, 0.0)) for action_type in ACTION_TYPES}
        row["reason"] = "hard_fault_cone+scoap" if hard_weights else "scoap_fallback"
        candidates.append(row)
    candidates.sort(key=lambda item: item["priority"], reverse=True)
    return {
        "benchmark_id": benchmark_id,
        "bench_path": str(bench_path),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "source": "tpi_eval.candidates.generate_candidates intersected with hard_fault_cone_scoap_rank",
        "hard_fault_source": hard_source,
        "fanin_depth": fanin_depth,
        "fanout_depth": fanout_depth,
        **stats,
    }


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description="Build hard-fault-cone ranked insertable TP caches.")
    parser.add_argument("--benchmarks", required=True, help="Comma-separated benchmark ids.")
    parser.add_argument("--out-dir", required=True, help="Directory for {benchmark_id}.json cache files.")
    parser.add_argument("--hard-fault-root", default=None, help="Eval root containing **/{benchmark}/baseline artifacts.")
    parser.add_argument("--undetected-nodes-template", default=None, help="Optional path template with {benchmark_id}.")
    parser.add_argument("--faults-template", default=None, help="Optional path template with {benchmark_id}.")
    parser.add_argument("--fanin-depth", type=int, default=8)
    parser.add_argument("--fanout-depth", type=int, default=8)
    parser.add_argument("--max-hard-nodes", type=int, default=1024)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(args.hard_fault_root) if args.hard_fault_root else None
    for benchmark_id in parse_csv_values(args.benchmarks):
        payload = build_cache(
            benchmark_id,
            root,
            args.undetected_nodes_template,
            args.faults_template,
            args.fanin_depth,
            args.fanout_depth,
            args.max_hard_nodes,
        )
        path = out_dir / f"{benchmark_id}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(
            f"{benchmark_id}\tcandidates={payload['candidate_count']}\t"
            f"hard_seeds={payload['hard_seed_count']}\tpath={path}"
        )


if __name__ == "__main__":
    main()
