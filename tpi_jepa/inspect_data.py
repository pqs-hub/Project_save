"""Inspect label availability before training."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from .labels import DEFAULT_LABELS, find_bench_path, load_labels


def inspect_labels(labels_csv: str | Path) -> dict:
    """Return basic data statistics for a label CSV."""

    total = 0
    status_counts: Counter[str] = Counter()
    step_counts: Counter[str] = Counter()
    with Path(labels_csv).open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            status_counts[row.get("status") or ""] += 1
            step_counts[row.get("step") or ""] += 1

    rows = load_labels(labels_csv)
    type_counts = Counter(row.raw_type for row in rows)
    bench_ids = sorted({row.benchmark_id for row in rows})
    missing = []
    for bench_id in bench_ids:
        try:
            find_bench_path(bench_id)
        except FileNotFoundError:
            missing.append(bench_id)
    pattern_values = [row.delta_pattern for row in rows if row.delta_pattern is not None]
    return {
        "total_rows": total,
        "valid_action_rows": len(rows),
        "status_counts": dict(status_counts),
        "step_counts": dict(step_counts),
        "type_counts": dict(type_counts),
        "benchmarks": bench_ids,
        "missing_benchmarks": missing,
        "pattern_target_present": bool(pattern_values),
        "pattern_target_valid": bool(pattern_values) and len(set(pattern_values)) > 1,
    }


def main() -> None:
    """CLI entry point for label inspection."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    args = parser.parse_args()

    stats = inspect_labels(args.labels)
    print(f"total_rows={stats['total_rows']}")
    print(f"valid_action_rows={stats['valid_action_rows']}")
    print(f"benchmarks={len(stats['benchmarks'])}")
    print(f"status_counts={stats['status_counts']}")
    print(f"type_counts={stats['type_counts']}")
    print(f"step_counts={stats['step_counts']}")
    print(f"missing_benchmarks={stats['missing_benchmarks']}")
    print(f"pattern_target_present={stats['pattern_target_present']}")
    print(f"pattern_target_valid={stats['pattern_target_valid']}")


if __name__ == "__main__":
    main()

