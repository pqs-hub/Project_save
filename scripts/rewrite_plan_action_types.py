"""Rewrite selected action types in a plan while preserving its node set."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--from-types", required=True, help="Comma-separated source action types.")
    parser.add_argument("--to-type", required=True, choices=("control0", "control1", "observe"))
    parser.add_argument("--min-step", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source_types = {item.strip() for item in args.from_types.split(",") if item.strip()}
    with args.plan.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    changed = 0
    for row in rows:
        if int(row["step"]) >= args.min_step and row["type"] in source_types:
            row["type"] = args.to_type
            changed += 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved={args.out} rows={len(rows)} changed={changed}")


if __name__ == "__main__":
    main()
