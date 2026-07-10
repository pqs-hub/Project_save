"""Profile candidate generation time without world-model scoring or TC eval."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tpi_jepa.bench import parse_bench  # noqa: E402
from tpi_jepa.graph import build_graph  # noqa: E402
from tpi_jepa.labels import find_bench_path  # noqa: E402
from tpi_jepa.plan import clear_planner_caches, enumerate_candidates  # noqa: E402


def choose_probe_action(
    recalled: list[tuple[str, str]],
    selected: list[tuple[str, str]],
    used_nodes: set[str],
    type_counts: dict[str, int],
    last_type: str,
) -> tuple[str, str] | None:
    """Pick one action so the next recall sees realistic selected history."""

    if not recalled:
        return None
    for candidate in recalled:
        node, action_type = candidate
        if node not in used_nodes and action_type != last_type:
            return candidate
    fresh = [candidate for candidate in recalled if candidate[0] not in used_nodes]
    pool = fresh if fresh else recalled
    return min(pool, key=lambda item: (type_counts.get(item[1], 0), item[1], item in selected))


def profile_strategy(
    *,
    graph,
    benchmark_id: str,
    strategy: str,
    budget: int,
    max_candidates: int,
    real_fault_priors: str | None,
    activation_priors: str | None,
    sample_seed: int,
) -> tuple[list[dict], dict]:
    """Run repeated candidate recall and return per-step plus aggregate timing."""

    clear_planner_caches()
    selected: list[tuple[str, str]] = []
    used_nodes: set[str] = set()
    type_counts: dict[str, int] = {}
    last_type = ""
    rows: list[dict] = []
    total_candidates = 0
    started = time.perf_counter()
    for step in range(1, budget + 1):
        step_started = time.perf_counter()
        recalled = enumerate_candidates(
            graph,
            selected,
            max_candidates,
            strategy,
            real_fault_benchmark_id=benchmark_id,
            real_fault_prior_path=real_fault_priors,
            activation_prior_path=activation_priors,
            candidate_sample_seed=sample_seed,
        )
        elapsed = time.perf_counter() - step_started
        total_candidates += len(recalled)
        chosen = choose_probe_action(recalled, selected, used_nodes, type_counts, last_type)
        if chosen is None:
            break
        selected.append(chosen)
        used_nodes.add(chosen[0])
        last_type = chosen[1]
        type_counts[last_type] = type_counts.get(last_type, 0) + 1
        rows.append(
            {
                "strategy": strategy,
                "benchmark_id": benchmark_id,
                "step": step,
                "candidate_count": len(recalled),
                "elapsed_sec": elapsed,
                "chosen_node": chosen[0],
                "chosen_type": chosen[1],
            }
        )
    total_elapsed = time.perf_counter() - started
    step_times = [float(row["elapsed_sec"]) for row in rows]
    summary = {
        "strategy": strategy,
        "benchmark_id": benchmark_id,
        "budget_requested": budget,
        "steps": len(rows),
        "max_candidates": max_candidates,
        "total_candidates": total_candidates,
        "first_step_sec": step_times[0] if step_times else 0.0,
        "recall_elapsed_sec_sum": sum(step_times),
        "wall_elapsed_sec": total_elapsed,
        "avg_recall_sec_per_step": (sum(step_times) / len(step_times)) if step_times else 0.0,
        "avg_recall_sec_per_candidate": (sum(step_times) / total_candidates) if total_candidates else 0.0,
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--max-candidates", type=int, default=96)
    parser.add_argument(
        "--candidate-strategies",
        default="hard_fault_cone,hard_fault_cluster,heuristic_recall_pool",
        help="Comma-separated candidate strategies to profile.",
    )
    parser.add_argument("--real-fault-priors", default=None)
    parser.add_argument("--activation-priors", default=None)
    parser.add_argument("--candidate-sample-seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("autoresearch/candidate-generation-profile"))
    args = parser.parse_args()

    graph = build_graph(parse_bench(find_bench_path(args.benchmark_id)))
    strategies = [item.strip() for item in args.candidate_strategies.split(",") if item.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    summaries: list[dict] = []
    for strategy in strategies:
        rows, summary = profile_strategy(
            graph=graph,
            benchmark_id=args.benchmark_id,
            strategy=strategy,
            budget=args.budget,
            max_candidates=args.max_candidates,
            real_fault_priors=args.real_fault_priors,
            activation_priors=args.activation_priors,
            sample_seed=args.candidate_sample_seed,
        )
        all_rows.extend(rows)
        summaries.append(summary)
        print(
            "[candidate-profile] "
            f"strategy={strategy} steps={summary['steps']} "
            f"first={summary['first_step_sec']:.6f}s "
            f"recall_sum={summary['recall_elapsed_sec_sum']:.6f}s "
            f"wall={summary['wall_elapsed_sec']:.6f}s "
            f"avg_step={summary['avg_recall_sec_per_step']:.6f}s"
        )

    with (args.out_dir / "candidate_generation_steps.tsv").open("w", newline="") as f:
        fieldnames = ["strategy", "benchmark_id", "step", "candidate_count", "elapsed_sec", "chosen_node", "chosen_type"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(all_rows)
    with (args.out_dir / "candidate_generation_summary.tsv").open("w", newline="") as f:
        fieldnames = [
            "strategy",
            "benchmark_id",
            "budget_requested",
            "steps",
            "max_candidates",
            "total_candidates",
            "first_step_sec",
            "recall_elapsed_sec_sum",
            "wall_elapsed_sec",
            "avg_recall_sec_per_step",
            "avg_recall_sec_per_candidate",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(summaries)
    print(f"wrote={args.out_dir / 'candidate_generation_summary.tsv'}")


if __name__ == "__main__":
    main()
