"""Build hard-fault priors from TMAX per-fault status CSV artifacts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from typing import Any


HARD_STATUSES = {"--", "UR", "AU", "UD", "UU", "AB", "ND", "NC", "NO", "UNDETECTED", "ABORT"}


def _is_hard(row: dict[str, str]) -> bool:
    value = (row.get("is_hard") or "").strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    return (row.get("status") or "").strip().upper() in HARD_STATUSES


def _fault_polarity(row: dict[str, str]) -> str | None:
    """Return the canonical stuck-at polarity when the fault report exposes it."""

    text = (row.get("fault_type") or "").strip().lower().replace("-", "")
    if text in {"sa0", "stuckat0", "stuck0"}:
        return "sa0"
    if text in {"sa1", "stuckat1", "stuck1"}:
        return "sa1"
    sa_value = (row.get("sa_value") or "").strip().lower()
    if sa_value in {"0", "sa0"}:
        return "sa0"
    if sa_value in {"1", "sa1"}:
        return "sa1"
    return None


def _labels_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file() and path.name == "labels.csv":
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("labels.csv")))
    return sorted(dict.fromkeys(files))


def _resolve_csv(row: dict[str, str], labels_csv: Path) -> Path | None:
    text = (row.get("fault_csv_path") or "").strip()
    if text:
        path = Path(text)
        if not path.is_absolute():
            path = labels_csv.parent / path
        if path.is_file():
            return path
    work_dir = (row.get("work_dir") or "").strip()
    if work_dir:
        path = Path(work_dir) / "faults_all_status.csv"
        if path.is_file():
            return path
    return None


def _update_aggregate(
    agg: dict[str, Any],
    *,
    benchmark_id: str,
    fault_csv: Path,
    source_label: Path,
) -> None:
    with fault_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            net = (row.get("net") or row.get("tmax_site") or "").strip()
            if not net:
                continue
            key = (benchmark_id, net)
            item = agg.setdefault(
                key,
                {
                    "benchmark_id": benchmark_id,
                    "net": net,
                    "fault_count": 0,
                    "hard_fault_count": 0,
                    "sa0_fault_count": 0,
                    "sa1_fault_count": 0,
                    "sa0_hard_fault_count": 0,
                    "sa1_hard_fault_count": 0,
                    "status_counts": Counter(),
                    "source_fault_csvs": set(),
                    "source_labels": set(),
                },
            )
            item["fault_count"] += 1
            is_hard = _is_hard(row)
            item["hard_fault_count"] += int(is_hard)
            polarity = _fault_polarity(row)
            if polarity is not None:
                item[f"{polarity}_fault_count"] += 1
                item[f"{polarity}_hard_fault_count"] += int(is_hard)
            item["status_counts"][(row.get("status") or "").strip().upper()] += 1
            item["source_fault_csvs"].add(str(fault_csv))
            item["source_labels"].add(str(source_label))


def _selected_label_rows(labels_csv: Path, *, last_row_only: bool) -> list[dict[str, str]]:
    with labels_csv.open(newline="") as f:
        rows = [dict(row) for row in csv.DictReader(f)]
    if not last_row_only:
        return rows
    non_baseline = [row for row in rows if int(row.get("step") or 0) > 0]
    return non_baseline[-1:] if non_baseline else rows[-1:]


def build_priors(labels_files: list[Path], *, last_row_only: bool = False) -> list[dict[str, Any]]:
    agg: dict[tuple[str, str], dict[str, Any]] = {}
    for labels_csv in labels_files:
        for row in _selected_label_rows(labels_csv, last_row_only=last_row_only):
            benchmark_id = (row.get("benchmark_id") or "").strip()
            if not benchmark_id:
                continue
            fault_csv = _resolve_csv(row, labels_csv)
            if fault_csv is None:
                continue
            _update_aggregate(agg, benchmark_id=benchmark_id, fault_csv=fault_csv, source_label=labels_csv)

    rows: list[dict[str, Any]] = []
    for item in agg.values():
        fault_count = int(item["fault_count"])
        hard_count = int(item["hard_fault_count"])
        sa0_count = int(item["sa0_fault_count"])
        sa1_count = int(item["sa1_fault_count"])
        sa0_hard_count = int(item["sa0_hard_fault_count"])
        sa1_hard_count = int(item["sa1_hard_fault_count"])
        rows.append(
            {
                "benchmark_id": item["benchmark_id"],
                "net": item["net"],
                "fault_count": fault_count,
                "hard_fault_count": hard_count,
                "hard_fault_ratio": (hard_count / fault_count) if fault_count else 0.0,
                "sa0_fault_count": sa0_count,
                "sa1_fault_count": sa1_count,
                "sa0_hard_fault_count": sa0_hard_count,
                "sa1_hard_fault_count": sa1_hard_count,
                "sa0_hard_fault_ratio": (sa0_hard_count / sa0_count) if sa0_count else 0.0,
                "sa1_hard_fault_ratio": (sa1_hard_count / sa1_count) if sa1_count else 0.0,
                "status_counts": dict(sorted(item["status_counts"].items())),
                "source_fault_csv_count": len(item["source_fault_csvs"]),
                "source_label_count": len(item["source_labels"]),
            }
        )
    return sorted(rows, key=lambda row: (row["benchmark_id"], -row["hard_fault_ratio"], -row["hard_fault_count"], row["net"]))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "benchmark_id",
        "net",
        "fault_count",
        "hard_fault_count",
        "hard_fault_ratio",
        "sa0_fault_count",
        "sa1_fault_count",
        "sa0_hard_fault_count",
        "sa1_hard_fault_count",
        "sa0_hard_fault_ratio",
        "sa1_hard_fault_ratio",
        "status_counts",
        "source_fault_csv_count",
        "source_label_count",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "status_counts": json.dumps(row["status_counts"], sort_keys=True)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="labels.csv files or directories containing evaluator outputs.")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument(
        "--last-row-only",
        action="store_true",
        help="Build a residual prior from only the last non-baseline row of each labels.csv.",
    )
    args = parser.parse_args()

    labels_files = _labels_files(args.paths)
    rows = build_priors(labels_files, last_row_only=args.last_row_only)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    write_csv(args.out_csv, rows)
    print(
        json.dumps(
            {
                "labels_files": len(labels_files),
                "last_row_only": args.last_row_only,
                "prior_rows": len(rows),
                "out_json": str(args.out_json),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
