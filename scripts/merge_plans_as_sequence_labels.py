#!/usr/bin/env python3
"""Convert per-benchmark planner CSVs into relabel_sequences input rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = ["benchmark_id", "sequence_id", "step", "net", "type", "status"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.plans_dir.glob("subckt_*.csv")):
        benchmark_id = path.stem
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(
                    {
                        "benchmark_id": benchmark_id,
                        "sequence_id": f"onpolicy:{benchmark_id}",
                        "step": int(row["step"]),
                        "net": row["node"],
                        "type": row["type"],
                        "status": "ok",
                    }
                )
    if not rows:
        raise SystemExit(f"no plan rows found under {args.plans_dir}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"plans={len({row['benchmark_id'] for row in rows})} rows={len(rows)} out={args.out}")


if __name__ == "__main__":
    main()
