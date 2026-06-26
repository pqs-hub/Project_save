"""Run backend relabeling in small sequence batches.

This is useful for large circuits where submitting hundreds of long Atalanta
jobs to one ThreadPool can be brittle.  Each batch reuses the same relabel
output directory, so completed sequence folders are preserved and the final
batch merge sees all sequence labels already present under the output tree.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def load_grouped(labels: Path) -> tuple[list[str], list[tuple[str, list[dict[str, str]]]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    with labels.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            if (row.get("step") or "0") == "0":
                continue
            key = f"{row.get('benchmark_id')}::{row.get('sequence_id')}"
            grouped.setdefault(key, []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row.get("step") or 0))
    return fieldnames, sorted(grouped.items())


def write_batch(path: Path, fieldnames: list[str], items: list[tuple[str, list[dict[str, str]]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for _, rows in items:
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--parallel-jobs", type=int, default=4)
    parser.add_argument("--backend", default="atalanta-bist")
    parser.add_argument("--patterns", type=int, default=300000)
    parser.add_argument("--timeout-sec", type=int, default=14400)
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    fieldnames, items = load_grouped(args.labels)
    args.batch_dir.mkdir(parents=True, exist_ok=True)
    if not items:
        raise SystemExit("no sequences found")

    total = len(items)
    for batch_idx, start in enumerate(range(0, total, args.batch_size), start=1):
        batch_items = items[start : start + args.batch_size]
        batch_csv = args.batch_dir / f"batch_{batch_idx:04d}.csv"
        write_batch(batch_csv, fieldnames, batch_items)
        print(
            f"batch {batch_idx}: sequences {start + 1}-{start + len(batch_items)} / {total}",
            flush=True,
        )
        cmd = [
            sys.executable,
            "scripts/relabel_sequences_with_backend.py",
            "--labels",
            str(batch_csv),
            "--out-dir",
            str(args.out_dir),
            "--backend",
            args.backend,
            "--patterns",
            str(args.patterns),
            "--parallel-jobs",
            str(args.parallel_jobs),
            "--timeout-sec",
            str(args.timeout_sec),
            "--force",
            "--resume",
            "--cleanup-workdir",
            *args.extra_args,
        ]
        subprocess.run(cmd, check=True)
    print("all batches finished", flush=True)


if __name__ == "__main__":
    main()
