#!/usr/bin/env python3
"""Summarize held-out eval8 final TC against the DeepTPI/Table-II result."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path


PROTOCOL_PATH = Path("configs/eval_protocol_coverage_only.json")

OLD_DELTA_TC = {
    "epfl__arithmetic__max__max": 0.11048,
    "epfl__random_control__i2c__i2c": 0.01196,
    "iscas99__b15_1": 0.10949,
    "iscas99__b17": 0.01357,
    "iscas99__b20": 0.05351,
    "iscas99__b21": 0.06013,
    "iscas99__b22": 0.03542,
    "openabcd__mem_ctrl_orig": 0.03630,
}


csv.field_size_limit(sys.maxsize)


def load_table_rows() -> dict[str, dict[str, str | float]]:
    data = json.loads(PROTOCOL_PATH.read_text())
    rows = {}
    for row in data.get("table_rows", []):
        bench = row.get("benchmark_id")
        if bench:
            rows[bench] = row
    return rows


def finite_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def measured_baseline_tc(row: dict[str, str]) -> float | None:
    eval_dir = row.get("eval_dir", "")
    if not eval_dir:
        return None
    labels_csv = Path(eval_dir) / "labels.csv"
    if not labels_csv.exists():
        return None
    try:
        with labels_csv.open(newline="") as f:
            labels = list(csv.DictReader(f))
    except (OSError, csv.Error):
        return None
    if not labels:
        return None
    final = labels[-1]
    baseline = finite_float(final.get("baseline_test_coverage", ""))
    if baseline is not None:
        return baseline
    return finite_float(labels[0].get("test_coverage", ""))


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: summarize_vs_old.py <summary.tsv>", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    rows = list(csv.DictReader(path.open(newline=""), delimiter="\t"))
    table_rows = load_table_rows()
    out_path = path.with_name("comparison_final_tc_vs_deeptpi.tsv")
    fields = [
        "id",
        "circuit",
        "model",
        "benchmark_id",
        "score_field",
        "baseline_tc_pct",
        "model_final_tc_pct",
        "deeptpi_final_tc_pct",
        "gap_vs_deeptpi_pp",
        "beats_deeptpi",
        "status",
        "budget",
        "delta_test_coverage_pct",
    ]
    comparison = []
    for row in rows:
        bench = row.get("benchmark_id", "")
        delta = finite_float(row.get("delta_test_coverage", ""))
        table_row = table_rows.get(bench)
        baseline = measured_baseline_tc(row)
        if delta is None or baseline is None or table_row is None:
            continue
        deeptpi_final = float(table_row["paper_reference_tc"])
        model_final = baseline + delta
        gap = model_final - deeptpi_final
        comparison.append(
            {
                "id": table_row.get("id", ""),
                "circuit": table_row.get("circuit", ""),
                "model": row.get("model", ""),
                "benchmark_id": bench,
                "score_field": row.get("score_field", ""),
                "baseline_tc_pct": f"{baseline * 100.0:.3f}",
                "model_final_tc_pct": f"{model_final * 100.0:.3f}",
                "deeptpi_final_tc_pct": f"{deeptpi_final * 100.0:.3f}",
                "gap_vs_deeptpi_pp": f"{gap * 100.0:+.3f}",
                "beats_deeptpi": str(gap > 0.0),
                "status": row.get("status", ""),
                "budget": row.get("budget", ""),
                "delta_test_coverage_pct": f"{delta * 100.0:.3f}",
            }
        )
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(comparison)

    print(f"comparison: {out_path}")
    for model in sorted({row["model"] for row in comparison}):
        model_rows = [row for row in comparison if row["model"] == model]
        gaps = [float(row["gap_vs_deeptpi_pp"]) for row in model_rows]
        final_tcs = [float(row["model_final_tc_pct"]) for row in model_rows]
        if not gaps:
            continue
        print(
            f"{model}\tn={len(gaps)}\tmacro_final_tc={sum(final_tcs) / len(final_tcs):.3f}%"
            f"\tmin_final_tc={min(final_tcs):.3f}%"
            f"\tbeats_deeptpi={sum(g > 0.0 for g in gaps)}/{len(gaps)}"
            f"\tworst_gap={min(gaps):+.3f}pp"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
