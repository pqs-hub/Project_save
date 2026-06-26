"""Summarize TAC framework-sweep results by testability technique."""

from __future__ import annotations

import argparse
import csv
import json
from math import isnan
from pathlib import Path
from typing import Any


FIELDS = [
    "framework_variant",
    "runs",
    "ok_runs",
    "best_delta_test_coverage",
    "mean_delta_test_coverage",
    "best_variant",
    "best_plan_csv",
    "components",
    "hypothesis",
]


def numeric(value: Any) -> float:
    if value in (None, "", "NA"):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


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


def summarize(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row.get("framework_variant", "unknown"), []).append(row)

    summaries: list[dict[str, Any]] = []
    for framework, group in sorted(groups.items()):
        metric_rows = [
            row
            for row in group
            if not isnan(numeric(row.get("delta_test_coverage")))
        ]
        values = [numeric(row.get("delta_test_coverage")) for row in metric_rows]
        best = max(metric_rows, key=lambda row: numeric(row.get("delta_test_coverage")), default={})
        exemplar = best or (group[0] if group else {})
        summaries.append(
            {
                "framework_variant": framework,
                "runs": len(group),
                "ok_runs": len(metric_rows),
                "best_delta_test_coverage": numeric(best.get("delta_test_coverage")) if best else "NA",
                "mean_delta_test_coverage": sum(values) / len(values) if values else "NA",
                "best_variant": best.get("variant", ""),
                "best_plan_csv": best.get("plan_csv", ""),
                "components": exemplar.get("framework_components", ""),
                "hypothesis": exemplar.get("framework_hypothesis", ""),
            }
        )
    summaries.sort(
        key=lambda row: numeric(row.get("best_delta_test_coverage")),
        reverse=True,
    )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize TAC framework sweep results.")
    parser.add_argument("--results", type=Path, default=Path("autoresearch/tac-framework-sweep-dev/results.tsv"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    summaries = summarize(read_tsv(args.results))
    out = args.out or args.results.with_name("technique_summary.tsv")
    write_tsv(out, summaries)
    payload = {
        "results": str(args.results),
        "summary": str(out),
        "best": summaries[0] if summaries else None,
    }
    args.results.with_name("technique_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
