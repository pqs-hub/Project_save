#!/usr/bin/env python3
"""Audit and summarize the five-circuit exact-candidate evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CIRCUITS = ("b15_C", "b20_C", "b21_C", "b22_C", "b17_C")

# DeepTPI final test coverage from Table IV, not the table's baseline TC column.
DEEPTPI_FINAL_TC_PCT = {
    "b15_C": 93.20,
    "b20_C": 95.02,
    "b21_C": 94.51,
    "b22_C": 95.59,
    "b17_C": 91.67,
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def resolve(repo: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repo / candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-root",
        default="autoresearch/exact-itc99-current-best-q-lcb-260712",
    )
    parser.add_argument(
        "--mapping-root",
        default="autoresearch/original-netlist-recovery-260712/exact_itc99",
    )
    parser.add_argument(
        "--old-best",
        default="autoresearch/improve-260706-0959/current_best.json",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    eval_root = resolve(repo, args.eval_root)
    mapping_root = resolve(repo, args.mapping_root)
    old_best = json.loads(resolve(repo, args.old_best).read_text())
    old_rows = {row["circuit"]: row for row in old_best["per_circuit_results"]}

    summary: list[dict[str, object]] = []
    for circuit in CIRCUITS:
        result_rows = read_tsv(eval_root / circuit / "results.tsv")
        successful_rows = [row for row in result_rows if row["status"] == "ok"]
        if len(successful_rows) != 1:
            raise RuntimeError(
                f"{circuit}: expected one successful result (failed retries are allowed), "
                f"got {len(successful_rows)} successful rows from {len(result_rows)} attempts"
            )
        result = successful_rows[0]
        plan_path = resolve(repo, result["plan_csv"])
        plan = read_csv(plan_path)
        allowed = {
            line.strip()
            for line in (mapping_root / circuit / "exact_candidate_nodes.txt").read_text().splitlines()
            if line.strip()
        }
        illegal = sorted({row["node"] for row in plan} - allowed)
        if illegal:
            raise RuntimeError(f"{circuit}: plan contains illegal nodes: {illegal[:10]}")
        if len(plan) != int(result["budget"]):
            raise RuntimeError(f"{circuit}: plan length {len(plan)} != budget {result['budget']}")

        labels_path = resolve(repo, result["eval_dir"]) / "labels.csv"
        labels = read_csv(labels_path)
        baseline = next(row for row in labels if int(row["step"]) == 0)
        final = max(labels, key=lambda row: int(row["step"]))
        old = old_rows[circuit]
        baseline_pct = 100.0 * float(baseline["test_coverage"])
        final_pct = 100.0 * float(final["test_coverage"])
        delta_pct = 100.0 * float(final["delta_test_coverage"])
        paper_pct = DEEPTPI_FINAL_TC_PCT[circuit]
        old_final_pct = float(old["model_final_tc_pct"])
        summary.append(
            {
                "circuit": circuit,
                "benchmark_id": result["benchmark_id"],
                "budget": int(result["budget"]),
                "allowlist_nodes": len(allowed),
                "plan_nodes": len(plan),
                "legal_plan_nodes": len(plan) - len(illegal),
                "legal_ratio_pct": 100.0,
                "baseline_tc_pct": baseline_pct,
                "filtered_final_tc_pct": final_pct,
                "filtered_delta_tc_pp": delta_pct,
                "unfiltered_final_tc_pct": old_final_pct,
                "change_vs_unfiltered_pp": final_pct - old_final_pct,
                "deeptpi_final_tc_pct": paper_pct,
                "gap_vs_deeptpi_pp": final_pct - paper_pct,
                "beats_deeptpi": final_pct > paper_pct,
                "plan_elapsed_sec": float(result["plan_elapsed_sec"]),
                "eval_elapsed_sec": float(result["eval_elapsed_sec"]),
                "plan_csv": str(plan_path.relative_to(repo)),
                "eval_dir": str(resolve(repo, result["eval_dir"]).relative_to(repo)),
            }
        )

    numeric_fields = (
        "baseline_tc_pct",
        "filtered_final_tc_pct",
        "filtered_delta_tc_pp",
        "unfiltered_final_tc_pct",
        "change_vs_unfiltered_pp",
        "deeptpi_final_tc_pct",
        "gap_vs_deeptpi_pp",
    )
    aggregate = {
        "circuits": len(summary),
        **{
            f"macro_{field}": sum(float(row[field]) for row in summary) / len(summary)
            for field in numeric_fields
        },
        "beats_deeptpi": sum(bool(row["beats_deeptpi"]) for row in summary),
        "legal_plan_nodes": sum(int(row["legal_plan_nodes"]) for row in summary),
        "plan_nodes": sum(int(row["plan_nodes"]) for row in summary),
        "all_beat_deeptpi": all(bool(row["beats_deeptpi"]) for row in summary),
    }
    payload = {"aggregate": aggregate, "per_circuit": summary}
    refresh_manifest_path = eval_root / "refresh_manifest.json"
    if refresh_manifest_path.is_file():
        refresh_manifest = json.loads(refresh_manifest_path.read_text())
        payload["method"] = refresh_manifest.get("method", {})
    (eval_root / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")

    fields = list(summary[0])
    with (eval_root / "comparison.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
