"""Merge original and on-policy TMAX label CSV files without sequence collisions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_csv_values(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def collect_inputs(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for item in parse_csv_values(args.labels):
        paths.append(Path(item))
    for pattern in args.glob:
        paths.extend(sorted(Path().glob(pattern)))
    seen = set()
    unique = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge label CSVs for retraining.")
    parser.add_argument("--labels", default="", help="Comma-separated labels.csv files.")
    parser.add_argument("--glob", action="append", default=[], help="Additional glob pattern for labels.csv files.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--no-remap-sequence-ids", action="store_true")
    args = parser.parse_args()

    inputs = collect_inputs(args)
    if not inputs:
        raise SystemExit("no input labels provided")

    fieldnames: list[str] = []
    rows: list[dict[str, str]] = []
    seen_keys = set()
    remap = not args.no_remap_sequence_ids
    for source_index, path in enumerate(inputs):
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                for field in reader.fieldnames:
                    if field not in fieldnames:
                        fieldnames.append(field)
            if "source_label_csv" not in fieldnames:
                fieldnames.append("source_label_csv")
            for raw in reader:
                row = dict(raw)
                if remap and (row.get("sequence_id") or "") != "":
                    row["sequence_id"] = f"{source_index}:{row.get('sequence_id')}"
                row["source_label_csv"] = str(path)
                key = (
                    row.get("benchmark_id"),
                    row.get("sequence_id"),
                    row.get("step"),
                    row.get("net"),
                    row.get("type"),
                    row.get("insertion_sequence"),
                    row.get("source_plan_csv"),
                    row.get("source_label_csv"),
                )
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    print(f"merged_files={len(inputs)}")
    print(f"merged_rows={len(rows)}")
    print(f"out={args.out}")


if __name__ == "__main__":
    main()
