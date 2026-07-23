#!/usr/bin/env python3
"""Build long structural-only rollout labels from training subcircuits.

The output deliberately contains no synthetic ATPG reward or hard-fault targets.
It is intended for JEPA state-transition consistency fine-tuning only.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tpi_jepa.bench import parse_bench  # noqa: E402
from tpi_jepa.graph import build_graph  # noqa: E402
from tpi_jepa.labels import DEFAULT_LABELS, find_bench_path  # noqa: E402
from tpi_jepa.plan import enumerate_candidates  # noqa: E402
from tpi_jepa.protocol import eval_benchmarks_from_protocol  # noqa: E402


ACTION_TO_RAW = {"control0": "CP0", "control1": "CP1", "observe": "OP"}
FIELDNAMES = [
    "benchmark_id",
    "sequence_id",
    "step",
    "net",
    "type",
    "insertion_sequence",
    "delta_test_coverage",
    "status",
]


def source_benchmark_ids(labels: Path) -> list[str]:
    """Return sorted benchmark ids that have at least one valid source action."""

    benchmark_ids: set[str] = set()
    with labels.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") == "ok" and int(row.get("step") or 0) >= 1:
                benchmark_id = (row.get("benchmark_id") or "").strip()
                if benchmark_id:
                    benchmark_ids.add(benchmark_id)
    return sorted(benchmark_ids)


def unique_node_candidates(candidates: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Keep the highest-ranked action for every physical net."""

    selected: list[tuple[str, str]] = []
    seen_nodes: set[str] = set()
    for node, action_type in candidates:
        if node in seen_nodes or action_type not in ACTION_TO_RAW:
            continue
        seen_nodes.add(node)
        selected.append((node, action_type))
    return selected


def select_trajectory_actions(
    candidates: list[tuple[str, str]],
    *,
    length: int,
    pool_multiplier: int,
    rng: random.Random,
) -> list[tuple[str, str]]:
    """Sample an ordered trajectory from a high-ranked, unique-net pool."""

    unique = unique_node_candidates(candidates)
    pool = unique[: max(length, length * pool_multiplier)]
    if len(pool) < length:
        return []
    return rng.sample(pool, length)


def trajectory_rows(
    benchmark_id: str,
    sequence_id: str,
    actions: list[tuple[str, str]],
) -> list[dict[str, object]]:
    """Convert one structural action trajectory into load_labels-compatible rows."""

    prefix: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    for step, (node, action_type) in enumerate(actions, start=1):
        raw_type = ACTION_TO_RAW[action_type]
        prefix.append({"net": node, "type": raw_type})
        rows.append(
            {
                "benchmark_id": benchmark_id,
                "sequence_id": sequence_id,
                "step": step,
                "net": node,
                "type": raw_type,
                "insertion_sequence": json.dumps(prefix, separators=(",", ":")),
                # This is a structural-consistency dataset. Reward losses must
                # be disabled in the corresponding training configuration.
                "delta_test_coverage": 0.0,
                "status": "ok",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--eval-protocol", type=Path, default=Path("configs/eval_protocol_coverage_only.json"))
    parser.add_argument("--num-benchmarks", type=int, default=160)
    parser.add_argument("--trajectories-per-benchmark", type=int, default=2)
    parser.add_argument("--trajectory-length", type=int, default=64)
    parser.add_argument("--pool-multiplier", type=int, default=4)
    parser.add_argument("--max-nodes", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=260719)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    if args.num_benchmarks <= 0 or args.trajectories_per_benchmark <= 0:
        raise SystemExit("num-benchmarks and trajectories-per-benchmark must be positive")
    if args.trajectory_length <= 0 or args.pool_multiplier <= 0:
        raise SystemExit("trajectory-length and pool-multiplier must be positive")

    excluded = eval_benchmarks_from_protocol(args.eval_protocol)
    benchmark_ids = [item for item in source_benchmark_ids(args.labels) if item not in excluded]
    rng = random.Random(args.seed)
    rng.shuffle(benchmark_ids)
    rows: list[dict[str, object]] = []
    accepted: list[str] = []
    skipped: dict[str, int] = {"missing_bench": 0, "too_large": 0, "too_few_candidates": 0}

    for benchmark_id in benchmark_ids:
        if len(accepted) >= args.num_benchmarks:
            break
        try:
            graph = build_graph(parse_bench(find_bench_path(benchmark_id)))
        except FileNotFoundError:
            skipped["missing_bench"] += 1
            continue
        if graph.num_nodes > args.max_nodes:
            skipped["too_large"] += 1
            continue
        ranked = enumerate_candidates(graph, [], None, strategy="testability")
        if len(unique_node_candidates(ranked)) < args.trajectory_length:
            skipped["too_few_candidates"] += 1
            continue
        benchmark_rows: list[dict[str, object]] = []
        for trajectory_index in range(args.trajectories_per_benchmark):
            local_rng = random.Random(rng.randrange(2**63))
            actions = select_trajectory_actions(
                ranked,
                length=args.trajectory_length,
                pool_multiplier=args.pool_multiplier,
                rng=local_rng,
            )
            if not actions:
                benchmark_rows = []
                break
            sequence_id = f"structural:{benchmark_id}:{trajectory_index:02d}"
            benchmark_rows.extend(trajectory_rows(benchmark_id, sequence_id, actions))
        if not benchmark_rows:
            skipped["too_few_candidates"] += 1
            continue
        rows.extend(benchmark_rows)
        accepted.append(benchmark_id)

    if len(accepted) != args.num_benchmarks:
        raise SystemExit(f"accepted only {len(accepted)} of requested {args.num_benchmarks} benchmarks")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "source_labels": str(args.labels),
        "eval_protocol": str(args.eval_protocol),
        "excluded_benchmarks": sorted(excluded),
        "accepted_benchmarks": accepted,
        "num_benchmarks": len(accepted),
        "trajectories_per_benchmark": args.trajectories_per_benchmark,
        "trajectory_length": args.trajectory_length,
        "rows": len(rows),
        "pool_multiplier": args.pool_multiplier,
        "max_nodes": args.max_nodes,
        "seed": args.seed,
        "reward_targets": "structural_only_zero_placeholders; disable reward/hard losses",
        "skipped": skipped,
        "out": str(args.out),
    }
    manifest_path = args.manifest or args.out.with_name("manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
