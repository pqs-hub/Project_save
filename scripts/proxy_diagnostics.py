"""Diagnose whether planner proxy scores rank true TMAX TC gains well."""

from __future__ import annotations

import argparse
import csv
import json
from math import isnan
from pathlib import Path
from statistics import mean
from typing import Any


PROXY_FIELDS = [
    "plan_score_sum",
    "plan_fc_sum",
    "plan_return_sum",
    "plan_sequence_sum",
    "plan_objective_sum",
]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def numeric(value: Any, default: float = float("nan")) -> float:
    if value in (None, "", "NA"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            result[order[k]] = rank
        i = j + 1
    return result


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return float("nan")
    x_mean = mean(xs)
    y_mean = mean(ys)
    cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    if x_var <= 0.0 or y_var <= 0.0:
        return float("nan")
    return cov / ((x_var * y_var) ** 0.5)


def spearman(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if not isnan(x) and not isnan(y)]
    if len(pairs) < 2:
        return float("nan")
    clean_xs = [x for x, _ in pairs]
    clean_ys = [y for _, y in pairs]
    return pearson(ranks(clean_xs), ranks(clean_ys))


def top_rows(grouped_rows: list[dict[str, str]], top_k: int) -> list[dict[str, str]]:
    def key(row: dict[str, str]) -> tuple[int, float, float, float]:
        safe = 1 if row.get("safe") == "True" else 0
        return (
            safe,
            numeric(row.get("macro_mean_delta_tc"), -1e9),
            numeric(row.get("min_delta_tc"), -1e9),
            numeric(row.get("router_delta_tc"), -1e9),
        )

    return sorted(grouped_rows, key=key, reverse=True)[:top_k]


def diagnostics_for_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    true_values = [numeric(row.get("delta_test_coverage")) for row in rows]
    true_best = max(true_values) if true_values else float("nan")
    report: dict[str, Any] = {}
    for proxy in PROXY_FIELDS:
        proxy_values = [numeric(row.get(proxy)) for row in rows]
        corr = spearman(proxy_values, true_values)
        ordered = sorted(
            [row for row in rows if not isnan(numeric(row.get(proxy)))],
            key=lambda row: numeric(row.get(proxy)),
            reverse=True,
        )
        top1 = ordered[0] if ordered else None
        top3 = ordered[:3]
        top1_delta = numeric(top1.get("delta_test_coverage")) if top1 else float("nan")
        true_best_variant = max(rows, key=lambda row: numeric(row.get("delta_test_coverage"))) if rows else None
        top3_hit = bool(true_best_variant and any(row.get("variant_id") == true_best_variant.get("variant_id") for row in top3))
        report[proxy] = {
            "spearman": corr,
            "top1_variant": top1.get("variant_id") if top1 else "",
            "top1_delta_tc": top1_delta,
            "top1_regret": true_best - top1_delta if not isnan(true_best) and not isnan(top1_delta) else float("nan"),
            "top3_hit": top3_hit,
            "true_best_variant": true_best_variant.get("variant_id") if true_best_variant else "",
            "true_best_delta_tc": true_best,
        }
    return report


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and isnan(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Proxy-vs-TMAX ranking diagnostics.")
    parser.add_argument("--grouped-results", required=True, type=Path)
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--router-benchmark", default="epfl__random_control__router__router")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    grouped_rows = read_tsv(args.grouped_results)
    results_path = args.results or args.grouped_results.parent / "results.tsv"
    result_rows = read_tsv(results_path)
    selected_variants = {row["variant_id"] for row in top_rows(grouped_rows, args.top_k)}
    selected_rows = [
        row for row in result_rows if row.get("variant_id") in selected_variants and row.get("status") == "ok"
    ]

    by_benchmark: dict[str, list[dict[str, str]]] = {}
    for row in selected_rows:
        by_benchmark.setdefault(row.get("benchmark_id", ""), []).append(row)

    per_benchmark = {
        benchmark: diagnostics_for_rows(rows)
        for benchmark, rows in sorted(by_benchmark.items())
    }
    all_rows_report = diagnostics_for_rows(selected_rows)

    rank_rows: list[dict[str, Any]] = []
    for benchmark, rows in sorted(by_benchmark.items()):
        for true_rank, row in enumerate(
            sorted(rows, key=lambda item: numeric(item.get("delta_test_coverage")), reverse=True),
            start=1,
        ):
            rank_rows.append(
                {
                    "benchmark_id": benchmark,
                    "true_rank": true_rank,
                    "variant_id": row.get("variant_id"),
                    "delta_test_coverage": row.get("delta_test_coverage"),
                    **{field: row.get(field, "") for field in PROXY_FIELDS},
                    "plan_csv": row.get("plan_csv"),
                    "eval_dir": row.get("eval_dir"),
                }
            )

    router_rows = by_benchmark.get(args.router_benchmark, [])
    router_false_positives = []
    if router_rows:
        for proxy in PROXY_FIELDS:
            ordered = sorted(router_rows, key=lambda row: numeric(row.get(proxy)), reverse=True)
            for row in ordered[:3]:
                if numeric(row.get("delta_test_coverage")) < 0.0:
                    router_false_positives.append(
                        {
                            "proxy": proxy,
                            "variant_id": row.get("variant_id"),
                            "proxy_value": numeric(row.get(proxy)),
                            "delta_test_coverage": numeric(row.get("delta_test_coverage")),
                            "plan_csv": row.get("plan_csv"),
                            "eval_dir": row.get("eval_dir"),
                        }
                    )

    report = {
        "grouped_results": str(args.grouped_results),
        "results": str(results_path),
        "top_k": args.top_k,
        "selected_variants": sorted(selected_variants),
        "overall": all_rows_report,
        "per_benchmark": per_benchmark,
        "router_false_positives": router_false_positives,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "proxy_report.json").write_text(json.dumps(json_ready(report), indent=2, sort_keys=True) + "\n")
    write_tsv(
        args.out_dir / "per_benchmark_rank.tsv",
        rank_rows,
        ["benchmark_id", "true_rank", "variant_id", "delta_test_coverage", *PROXY_FIELDS, "plan_csv", "eval_dir"],
    )

    lines = ["# Proxy Diagnostics", ""]
    lines.append(f"Selected variants: {len(selected_variants)}")
    lines.append("")
    for benchmark, metrics in per_benchmark.items():
        lines.append(f"## {benchmark}")
        for proxy in PROXY_FIELDS:
            item = metrics[proxy]
            spearman_text = "NA" if item["spearman"] is None else f"{item['spearman']:.4f}"
            regret_text = "NA" if item["top1_regret"] is None else f"{item['top1_regret']:.6f}"
            lines.append(
                f"- {proxy}: spearman={spearman_text}, top1={item['top1_variant']}, "
                f"top1_delta={item['top1_delta_tc']}, regret={regret_text}, top3_hit={item['top3_hit']}"
            )
        lines.append("")
    if router_false_positives:
        lines.append("## Router False Positives")
        for item in router_false_positives:
            lines.append(
                f"- {item['proxy']} {item['variant_id']}: proxy={item['proxy_value']:.6f}, "
                f"delta={item['delta_test_coverage']:.6f}"
            )
    (args.out_dir / "proxy_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
