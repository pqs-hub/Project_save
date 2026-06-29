"""Merge oracle action TSV files with group/action-level de-duplication."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
from typing import Any


DEDUP_KEY = ["benchmark_id", "state_id", "candidate_strategy", "action_key"]


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def row_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in DEDUP_KEY)


def group_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        str(row.get("benchmark_id", "")),
        str(row.get("state_id", "")),
        str(row.get("candidate_strategy", "")),
    )


def merge_fieldnames(base: list[str], incoming: list[str]) -> list[str]:
    merged = list(base)
    seen = set(merged)
    for field in incoming:
        if field not in seen:
            merged.append(field)
            seen.add(field)
    return merged


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Merged Oracle Action TSV",
        "",
        f"generated_at: `{payload['generated_at']}`",
        "",
        "## Summary",
        "",
        "| item | count |",
        "|---|---:|",
        f"| input files | {len(payload['inputs'])} |",
        f"| input rows | {payload['summary']['input_rows']} |",
        f"| output rows | {payload['summary']['output_rows']} |",
        f"| duplicate rows dropped | {payload['summary']['duplicates_dropped']} |",
        f"| output groups | {payload['summary']['output_groups']} |",
        f"| output benchmarks | {payload['summary']['output_benchmarks']} |",
        "",
        "## Inputs",
        "",
        "| path | rows | added | duplicates |",
        "|---|---:|---:|---:|",
    ]
    for item in payload["inputs"]:
        lines.append(f"| `{item['path']}` | {item['rows']} | {item['added']} | {item['duplicates']} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--out-tsv", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--keep", choices=["first", "last"], default="first")
    args = parser.parse_args()

    fieldnames: list[str] = []
    rows_by_key: dict[tuple[str, ...], dict[str, str]] = {}
    order: list[tuple[str, ...]] = []
    input_reports: list[dict[str, Any]] = []
    input_rows = 0
    duplicates = 0

    for path in args.input:
        incoming_fields, rows = read_tsv(path)
        missing = [field for field in DEDUP_KEY if field not in incoming_fields]
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}")
        fieldnames = merge_fieldnames(fieldnames, incoming_fields)
        added_for_file = 0
        duplicates_for_file = 0
        for row in rows:
            input_rows += 1
            key = row_key(row)
            if key in rows_by_key:
                duplicates += 1
                duplicates_for_file += 1
                if args.keep == "last":
                    rows_by_key[key] = row
                continue
            rows_by_key[key] = row
            order.append(key)
            added_for_file += 1
        input_reports.append(
            {
                "path": str(path),
                "rows": len(rows),
                "added": added_for_file,
                "duplicates": duplicates_for_file,
            }
        )

    merged = [rows_by_key[key] for key in order]
    groups = {group_key(row) for row in merged}
    benchmarks = {str(row.get("benchmark_id", "")) for row in merged}
    by_strategy = Counter(str(row.get("candidate_strategy", "")) for row in merged)
    by_type = Counter(str(row.get("type", "")) for row in merged)

    write_tsv(args.out_tsv, merged, fieldnames)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dedup_key": DEDUP_KEY,
        "keep": args.keep,
        "inputs": input_reports,
        "outputs": {
            "out_tsv": str(args.out_tsv),
            "out_report": str(args.out_report),
        },
        "summary": {
            "input_rows": input_rows,
            "output_rows": len(merged),
            "duplicates_dropped": duplicates,
            "output_groups": len(groups),
            "output_benchmarks": len(benchmarks),
        },
        "by_strategy": dict(sorted(by_strategy.items())),
        "by_type": dict(sorted(by_type.items())),
    }
    manifest = args.manifest or args.out_report.with_suffix(".json")
    write_json(manifest, payload)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(markdown_report(payload))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

