"""Write a heuristic candidate-generator baseline plan without loading a world model."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tpi_jepa.bench import parse_bench  # noqa: E402
from tpi_jepa.graph import build_graph  # noqa: E402
from tpi_jepa.labels import find_bench_path  # noqa: E402
from tpi_jepa.plan import PLAN_FIELDNAMES, enumerate_candidates  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--candidate-strategy", default="hard_fault_cone")
    parser.add_argument("--real-fault-priors", default=None)
    parser.add_argument("--activation-priors", default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    graph = build_graph(parse_bench(find_bench_path(args.benchmark_id)))
    candidates = enumerate_candidates(
        graph,
        [],
        args.budget,
        args.candidate_strategy,
        real_fault_benchmark_id=args.benchmark_id,
        real_fault_prior_path=args.real_fault_priors,
        activation_prior_path=args.activation_priors,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        fieldnames = [field for field in PLAN_FIELDNAMES if field in {"step", "node", "type", "candidate_strategy", "planner"}]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for step, (node, action_type) in enumerate(candidates, start=1):
            writer.writerow(
                {
                    "step": step,
                    "node": node,
                    "type": action_type,
                    "candidate_strategy": args.candidate_strategy,
                    "planner": "candidate_baseline",
                }
            )
    print(f"saved={args.out}")
    for step, (node, action_type) in enumerate(candidates, start=1):
        print(f"[candidate_baseline] step={step} node={node} type={action_type}")


if __name__ == "__main__":
    main()
