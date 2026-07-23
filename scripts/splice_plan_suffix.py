"""Replace a ranked plan suffix with novel nodes from another ranked plan."""

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
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--secondary", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--replace", type=int, required=True)
    parser.add_argument(
        "--secondary-offset",
        type=int,
        default=0,
        help="Skip this many eligible secondary nodes before filling the suffix.",
    )
    parser.add_argument(
        "--secondary-indices",
        default="",
        help=(
            "Comma-separated zero-based eligible-secondary ranks to select. "
            "When set, its length must equal --replace and --secondary-offset must be zero."
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.replace <= args.budget or args.secondary_offset < 0:
        parser.error("--replace must be between zero and --budget")
    try:
        secondary_indices = [
            int(value.strip()) for value in args.secondary_indices.split(",") if value.strip()
        ]
    except ValueError:
        parser.error("--secondary-indices must contain only comma-separated integers")
    if secondary_indices:
        if args.secondary_offset:
            parser.error("--secondary-indices and a nonzero --secondary-offset are mutually exclusive")
        if len(secondary_indices) != args.replace:
            parser.error("--secondary-indices must contain exactly --replace entries")
        if len(set(secondary_indices)) != len(secondary_indices) or min(secondary_indices) < 0:
            parser.error("--secondary-indices must be unique and nonnegative")
        secondary_indices.sort()

    fields, primary = read_plan(args.primary)
    _, secondary = read_plan(args.secondary)
    selected = [dict(row) for row in primary[: args.budget - args.replace]]
    used_nodes = {row["node"] for row in selected}
    eligible_seen = 0
    secondary_index_set = set(secondary_indices)
    for source_index, source in enumerate((secondary, primary[args.budget - args.replace :])):
        for row in source:
            if row["node"] in used_nodes:
                continue
            if source_index == 0:
                if secondary_indices:
                    eligible_index = eligible_seen
                    eligible_seen += 1
                    if eligible_index not in secondary_index_set:
                        continue
                elif eligible_seen < args.secondary_offset:
                    eligible_seen += 1
                    continue
            selected.append(dict(row))
            used_nodes.add(row["node"])
            if len(selected) == args.budget:
                break
        if len(selected) == args.budget:
            break
    if len(selected) != args.budget:
        raise SystemExit(f"only found {len(selected)} unique nodes for budget {args.budget}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for step, row in enumerate(selected, start=1):
            row["step"] = str(step)
            writer.writerow({field: row.get(field, "") for field in fields})
    print(
        f"saved={args.out} rows={len(selected)} replaced={args.replace} "
        f"secondary_offset={args.secondary_offset} secondary_indices={secondary_indices}"
    )


if __name__ == "__main__":
    main()
