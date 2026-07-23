"""Fuse ranked test-point plans with weighted reciprocal-rank voting."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_plan(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", nargs="+", type=Path, required=True)
    parser.add_argument("--weights", nargs="+", type=float, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--rrf-k", type=float, default=60.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if len(args.plans) != len(args.weights):
        parser.error("--plans and --weights must have the same length")
    if args.budget <= 0 or args.rrf_k < 0.0:
        parser.error("--budget must be positive and --rrf-k non-negative")

    scores: dict[tuple[str, str], float] = {}
    best_rows: dict[tuple[str, str], tuple[float, dict[str, str]]] = {}
    fieldnames: list[str] = []
    for path, weight in zip(args.plans, args.weights, strict=True):
        fields, rows = read_plan(path)
        if not fieldnames:
            fieldnames = fields
        for rank, row in enumerate(rows, start=1):
            key = (row["node"], row["type"])
            vote = float(weight) / (args.rrf_k + rank)
            scores[key] = scores.get(key, 0.0) + vote
            previous = best_rows.get(key)
            if previous is None or vote > previous[0]:
                best_rows[key] = (vote, row)

    ranked_actions = sorted(scores, key=lambda key: (-scores[key], key[0], key[1]))
    ranked: list[tuple[str, str]] = []
    used_nodes: set[str] = set()
    for key in ranked_actions:
        node, _ = key
        if node in used_nodes:
            continue
        used_nodes.add(node)
        ranked.append(key)
        if len(ranked) == args.budget:
            break
    if len(ranked) < args.budget:
        raise SystemExit(f"only {len(ranked)} unique actions for budget {args.budget}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for step, key in enumerate(ranked, start=1):
            row = dict(best_rows[key][1])
            row["step"] = str(step)
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    print(f"saved={args.out} actions={len(ranked)}")


if __name__ == "__main__":
    main()
