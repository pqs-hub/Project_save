"""Create a shorter plan by removing its lowest-ranked suffix."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--remove", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.remove <= 0:
        parser.error("--remove must be positive")
    with args.plan.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if args.remove >= len(rows):
        parser.error("--remove must be smaller than the plan length")
    rows = rows[: -args.remove]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved={args.out} rows={len(rows)} removed={args.remove}")


if __name__ == "__main__":
    main()
