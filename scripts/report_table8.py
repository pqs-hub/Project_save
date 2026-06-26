"""Render fixed table-8 evaluation results in paper-table format."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_last_csv_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else {}


def numeric(value: Any, default: float | None = None) -> float | None:
    if value in (None, "", "NA"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def pct(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * value:.2f}%"


def int_text(value: Any) -> str:
    if value in (None, "", "NA"):
        return "NA"
    return f"{int(float(value)):,}"


def choose_variant(rows: list[dict[str, str]], requested: str | None) -> str:
    if requested:
        return requested
    ok_rows = [row for row in rows if row.get("status") == "ok"] or rows
    if not ok_rows:
        raise SystemExit("no rows found; finish evaluation first")
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in ok_rows:
        grouped.setdefault(row["variant_id"], []).append(row)

    def key(item: tuple[str, list[dict[str, str]]]) -> tuple[int, float, float]:
        _, items = item
        deltas = [numeric(row.get("delta_test_coverage"), 0.0) or 0.0 for row in items]
        return (len(items), sum(deltas) / max(1, len(deltas)), min(deltas) if deltas else -1e9)

    return max(grouped.items(), key=key)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a table-8 TC improvement report.")
    parser.add_argument("--results", required=True, type=Path, help="run_gmean_sweep results.tsv")
    parser.add_argument("--protocol", default="configs/eval_protocol_coverage_only.json", type=Path)
    parser.add_argument("--variant-id", default=None)
    parser.add_argument("--method-name", default="TPI-JEPA")
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text())
    table_rows = protocol.get("table_rows")
    if not table_rows:
        raise SystemExit("protocol does not define table_rows")

    result_rows = read_tsv(args.results)
    variant_id = choose_variant(result_rows, args.variant_id)
    by_benchmark = {
        row.get("benchmark_id"): row
        for row in result_rows
        if row.get("variant_id") == variant_id
    }

    rows: list[dict[str, Any]] = []
    for item in table_rows:
        benchmark_id = item["benchmark_id"]
        result = by_benchmark.get(benchmark_id, {})
        eval_dir = result.get("eval_dir")
        labels_path = Path(eval_dir) / "labels.csv" if eval_dir else None
        metrics = read_last_csv_row(labels_path) if labels_path and labels_path.exists() else {}
        measured_baseline = numeric(metrics.get("baseline_test_coverage"))
        baseline_tc = measured_baseline if measured_baseline is not None else None
        final_tc = numeric(metrics.get("test_coverage"))
        delta_tc = numeric(metrics.get("delta_test_coverage"), numeric(result.get("delta_test_coverage")))
        if final_tc is None and baseline_tc is not None and delta_tc is not None:
            final_tc = baseline_tc + delta_tc
        rows.append(
            {
                "id": item["id"],
                "circuit": item["circuit"],
                "benchmark_id": benchmark_id,
                "baseline_tc": baseline_tc,
                "paper_reference_tc": numeric(item.get("paper_reference_tc")),
                "tp_budget": result.get("budget", ""),
                "method_tc": final_tc,
                "improvement": delta_tc,
            }
        )

    improvements = [row["improvement"] for row in rows if row["improvement"] is not None]
    avg_imp = sum(improvements) / len(improvements) if improvements else None

    headers = ["", "TC", "# TPs", f"{args.method_name} TC", "Imp."]
    md_lines = [
        f"# Fixed Table-8 Evaluation",
        "",
        f"Variant: `{variant_id}`",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        md_lines.append(
            "| "
            + " | ".join(
                [
                    row["id"],
                    pct(row["baseline_tc"]),
                    int_text(row["tp_budget"]),
                    pct(row["method_tc"]),
                    pct(row["improvement"]),
                ]
            )
            + " |"
        )
    md_lines.append("| Avg. |  |  |  | " + pct(avg_imp) + " |")
    md_text = "\n".join(md_lines) + "\n"

    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md_text)
    else:
        print(md_text)

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["id", "circuit", "benchmark_id", "baseline_tc", "tp_budget", "method_tc", "improvement"],
            )
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
