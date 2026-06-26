"""Evaluate existing plan CSVs on multiple benchmarks and summarize deltas."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import time
from typing import Any


FIELDS = [
    "benchmark_id",
    "status",
    "baseline_test_coverage",
    "final_test_coverage",
    "delta_test_coverage",
    "baseline_fault_coverage",
    "final_fault_coverage",
    "delta_fault_coverage",
    "pattern_count",
    "effective_pattern_count",
    "rows",
    "plan_csv",
    "eval_dir",
    "elapsed_sec",
    "error",
]


def parse_csv_values(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def run_one(args: argparse.Namespace, benchmark_id: str) -> dict[str, Any]:
    started = time.time()
    plan_csv = args.plan_dir / f"{benchmark_id}.csv"
    eval_dir = args.out_dir / "evals" / benchmark_id
    log_path = args.out_dir / "logs" / f"{benchmark_id}.log"
    row: dict[str, Any] = {
        "benchmark_id": benchmark_id,
        "plan_csv": str(plan_csv),
        "eval_dir": str(eval_dir),
    }
    if not plan_csv.is_file():
        row.update({"status": "missing_plan", "error": f"missing plan: {plan_csv}"})
        return row

    cmd = [
        "python",
        "-m",
        "tpi_jepa.evaluate_plan_tmax",
        "--benchmark-id",
        benchmark_id,
        "--plan-csv",
        str(plan_csv),
        "--out-dir",
        str(eval_dir),
        "--patterns",
        str(args.patterns),
        "--seed",
        str(args.seed),
        "--backend",
        args.backend,
        "--timeout-sec",
        str(args.timeout_sec),
        "--force",
    ]
    if args.backend == "atalanta-bist":
        cmd.extend(["--atalanta-bin", args.atalanta_bin])
    if args.cleanup_workdir:
        cmd.append("--cleanup-workdir")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        completed = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    if completed.returncode != 0:
        tail = "".join(log_path.read_text(errors="replace").splitlines(True)[-40:])
        row.update({"status": "error", "error": tail.strip(), "elapsed_sec": round(time.time() - started, 3)})
        return row

    labels_csv = eval_dir / "labels.csv"
    with labels_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    base = rows[0] if rows else {}
    final = rows[-1] if rows else {}
    row.update(
        {
            "status": final.get("status") or "ok",
            "baseline_test_coverage": base.get("test_coverage", ""),
            "final_test_coverage": final.get("test_coverage", ""),
            "delta_test_coverage": final.get("delta_test_coverage", ""),
            "baseline_fault_coverage": base.get("fault_coverage", ""),
            "final_fault_coverage": final.get("fault_coverage", ""),
            "delta_fault_coverage": final.get("delta_fault_coverage", ""),
            "pattern_count": final.get("pattern_count", ""),
            "effective_pattern_count": final.get("effective_pattern_count", ""),
            "rows": len(rows),
            "elapsed_sec": round(time.time() - started, 3),
        }
    )
    return row


def numeric(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", required=True)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=["tmax", "atalanta-bist"], default="atalanta-bist")
    parser.add_argument(
        "--atalanta-bin",
        default="/data3/pengqingsong/DFT/DeepTPI-project/external/DeepTPI/src/external/Atalanta_BIST/atalanta",
    )
    parser.add_argument("--patterns", type=int, default=300000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--timeout-sec", type=int, default=14400)
    parser.add_argument("--parallel-jobs", type=int, default=4)
    parser.add_argument("--cleanup-workdir", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    benchmarks = parse_csv_values(args.benchmarks)
    rows: list[dict[str, Any]] = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.parallel_jobs)) as executor:
        futures = {executor.submit(run_one, args, bench): bench for bench in benchmarks}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    results_path = args.out_dir / "results.tsv"
    with results_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        for row in sorted(rows, key=lambda item: str(item.get("benchmark_id", ""))):
            writer.writerow({field: row.get(field, "") for field in FIELDS})

    ok_rows = [row for row in rows if row.get("status") == "ok" and row.get("delta_test_coverage") not in ("", None)]
    deltas = [numeric(row.get("delta_test_coverage")) for row in ok_rows]
    summary = {
        "backend": args.backend,
        "benchmarks": benchmarks,
        "benchmark_count": len(benchmarks),
        "ok_count": len(ok_rows),
        "macro_mean_delta_tc": sum(deltas) / len(deltas) if deltas else None,
        "min_delta_tc": min(deltas) if deltas else None,
        "positive_count": sum(1 for value in deltas if value > 0),
        "negative_count": sum(1 for value in deltas if value < 0),
        "patterns": args.patterns,
        "elapsed_sec": round(time.time() - started, 3),
        "results_tsv": str(results_path),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
