"""Collect per-variant Q-v1 train/eval summaries into one table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


FIELDS = [
    "variant",
    "score_field",
    "expanded_spearman",
    "expanded_negative_top1",
    "expanded_top1_real_delta",
    "expanded_top1_regret",
    "transfer_spearman",
    "transfer_negative_top1",
    "transfer_top1_real_delta",
    "transfer_top1_regret",
    "verdict",
    "reasons",
]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Q-v1 Parallel Summary",
        "",
        "| variant | score | verdict | expanded Spearman | expanded neg top1 | transfer Spearman | transfer neg top1 | transfer regret |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {variant} | `{score_field}` | {verdict} | {expanded_spearman} | {expanded_negative_top1} | "
            "{transfer_spearman} | {transfer_negative_top1} | {transfer_top1_regret} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("autoresearch/q-v1-parallel-260630"))
    args = parser.parse_args()
    summary_dir = args.out_dir / "summaries"
    rows: list[dict[str, Any]] = []
    for path in sorted(summary_dir.glob("*_summary.tsv")):
        rows.extend(read_tsv(path))
    if not rows:
        raise FileNotFoundError(f"no *_summary.tsv files under {summary_dir}")
    write_tsv(args.out_dir / "q_v1_parallel_summary.tsv", rows)
    write_report(args.out_dir / "q_v1_parallel_report.md", rows)
    print(f"wrote {args.out_dir / 'q_v1_parallel_summary.tsv'}")
    print(f"wrote {args.out_dir / 'q_v1_parallel_report.md'}")


if __name__ == "__main__":
    main()
