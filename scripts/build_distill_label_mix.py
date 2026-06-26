"""Mix legacy sequence labels with repeated distilled on-policy labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def add_fields(fieldnames: list[str], *fields: str) -> list[str]:
    out = list(fieldnames)
    for field in fields:
        if field not in out:
            out.append(field)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-labels", type=Path, required=True)
    parser.add_argument("--distill-labels", type=Path, required=True)
    parser.add_argument("--distill-repeat", type=int, default=50)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    base_fields, base_rows = read_rows(args.base_labels)
    distill_fields, distill_rows = read_rows(args.distill_labels)
    fieldnames = add_fields(base_fields, *distill_fields, "source_label_csv", "source_sequence_id", "distill_repeat")

    mixed: list[dict[str, str]] = []
    for row in base_rows:
        new_row = dict(row)
        new_row["source_label_csv"] = str(args.base_labels)
        mixed.append(new_row)

    for repeat in range(max(0, args.distill_repeat)):
        for row in distill_rows:
            new_row = dict(row)
            new_row["source_label_csv"] = str(args.distill_labels)
            new_row["source_sequence_id"] = row.get("sequence_id", "")
            new_row["sequence_id"] = f"distill{repeat:03d}:{row.get('benchmark_id', '')}:{row.get('sequence_id', '')}"
            new_row["distill_repeat"] = str(repeat)
            mixed.append(new_row)

    write_rows(args.out, fieldnames, mixed)
    manifest = {
        "base_labels": str(args.base_labels),
        "base_rows": len(base_rows),
        "distill_labels": str(args.distill_labels),
        "distill_rows": len(distill_rows),
        "distill_repeat": args.distill_repeat,
        "mixed_rows": len(mixed),
        "out": str(args.out),
    }
    manifest_path = args.manifest or args.out.with_name("manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
