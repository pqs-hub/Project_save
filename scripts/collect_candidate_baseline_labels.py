"""Collect TMAX labels for heuristic candidate-baseline TPI sequences."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATASET_ROOT = Path(os.environ.get("DFT_ROOT", "/data4/pengqingsong/DFT")) / "Dataset"
if str(DATASET_ROOT) not in sys.path:
    sys.path.insert(0, str(DATASET_ROOT))

from tpi_jepa.bench import parse_bench  # noqa: E402
from tpi_jepa.evaluate_plan_tmax import evaluate_plan  # noqa: E402
from tpi_jepa.graph import build_graph  # noqa: E402
from tpi_jepa.labels import DEFAULT_LABELS, find_bench_path, load_labels  # noqa: E402
from tpi_jepa.plan import PLAN_FIELDNAMES, enumerate_candidates  # noqa: E402
from tpi_jepa.protocol import eval_benchmarks_from_protocol, parse_benchmark_list  # noqa: E402


def benchmark_ids(args: argparse.Namespace) -> list[str]:
    if args.benchmarks:
        return sorted(parse_benchmark_list(args.benchmarks))
    rows = load_labels(args.labels)
    benches = {row.benchmark_id for row in rows}
    benches -= eval_benchmarks_from_protocol(args.eval_protocol)
    benches -= parse_benchmark_list(args.extra_exclude)
    return sorted(benches)


def write_candidate_plan(
    *,
    benchmark_id: str,
    budget: int,
    strategy: str,
    real_fault_priors: str | None,
    activation_priors: str | None,
    out: Path,
) -> Path:
    graph = build_graph(parse_bench(find_bench_path(benchmark_id)))
    candidates = enumerate_candidates(
        graph,
        [],
        budget,
        strategy,
        real_fault_benchmark_id=benchmark_id,
        real_fault_prior_path=real_fault_priors,
        activation_prior_path=activation_priors,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field for field in PLAN_FIELDNAMES if field in {"step", "node", "type", "planner", "candidate_strategy"}]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for step, (node, action_type) in enumerate(candidates, start=1):
            writer.writerow(
                {
                    "step": step,
                    "node": node,
                    "type": action_type,
                    "planner": "candidate_baseline",
                    "candidate_strategy": strategy,
                }
            )
    return out


def evaluate_one(args: argparse.Namespace, benchmark_id: str) -> dict[str, Any]:
    started = time.time()
    bench_dir = args.out_dir / "benchmarks" / benchmark_id
    labels_path = bench_dir / "eval" / "labels.csv"
    if args.resume and labels_path.is_file():
        rows = list(csv.DictReader(labels_path.open()))
        if len(rows) >= args.budget + 1 and all(row.get("status") == "ok" for row in rows):
            return {"benchmark_id": benchmark_id, "status": "skipped", "rows": len(rows), "labels_csv": str(labels_path)}

    print(json.dumps({"benchmark_id": benchmark_id, "status": "started"}, sort_keys=True), flush=True)
    plan_csv = bench_dir / f"{args.candidate_strategy}_plan.csv"
    write_candidate_plan(
        benchmark_id=benchmark_id,
        budget=args.budget,
        strategy=args.candidate_strategy,
        real_fault_priors=args.real_fault_priors,
        activation_priors=args.activation_priors,
        out=plan_csv,
    )
    rows = evaluate_plan(
        benchmark_id=benchmark_id,
        plan_csv=plan_csv,
        out_dir=bench_dir / "eval",
        patterns=args.patterns,
        seed=args.seed,
        backend=args.backend,
        tmax_bin=args.tmax_bin,
        atalanta_bin=args.atalanta_bin,
        timeout_sec=args.timeout_sec,
        force=args.force,
        dry_run=args.dry_run,
        cleanup_workdir=args.cleanup_workdir,
    )
    status = "ok" if rows and all(row.get("status") == "ok" for row in rows) else "partial"
    summary = {
        "benchmark_id": benchmark_id,
        "status": status,
        "rows": len(rows),
        "elapsed_sec": round(time.time() - started, 3),
        "plan_csv": str(plan_csv),
        "labels_csv": str(labels_path),
    }
    (bench_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def merge_labels(out_dir: Path) -> Path:
    labels_files = sorted(out_dir.glob("benchmarks/*/eval/labels.csv"))
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    for source_index, path in enumerate(labels_files):
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for field in reader.fieldnames or []:
                if field not in fieldnames:
                    fieldnames.append(field)
            for row in reader:
                if (row.get("step") or "0") == "0":
                    continue
                merged = dict(row)
                merged["sequence_id"] = f"hfc:{source_index:04d}:{merged.get('sequence_id', '')}"
                merged["source_label_csv"] = str(path)
                rows.append(merged)
    if "source_label_csv" not in fieldnames:
        fieldnames.append("source_label_csv")
    merged_path = out_dir / "labels.csv"
    with merged_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return merged_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--eval-protocol", default="configs/eval_protocol_coverage_only.json")
    parser.add_argument("--extra-exclude", default="")
    parser.add_argument("--benchmarks", default="")
    parser.add_argument("--candidate-strategy", default="hard_fault_cone")
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--patterns", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--timeout-sec", type=int, default=7200)
    parser.add_argument("--parallel-jobs", type=int, default=3)
    parser.add_argument("--real-fault-priors", default=None)
    parser.add_argument("--activation-priors", default=None)
    parser.add_argument("--backend", choices=["tmax", "atalanta-bist"], default="tmax")
    parser.add_argument("--tmax-bin", default="/data3/pengqingsong/synopsys/txs/O-2018.06-SP1/bin/tmax")
    parser.add_argument(
        "--atalanta-bin",
        default=str(
            Path(os.environ.get("DFT_ROOT", "/data4/pengqingsong/DFT"))
            / "tool/atalanta_bist_with_ufaults/atalanta"
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cleanup-workdir", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    benches = benchmark_ids(args)
    manifest = {
        "benchmarks": benches,
        "candidate_strategy": args.candidate_strategy,
        "budget": args.budget,
        "patterns": args.patterns,
        "real_fault_priors": args.real_fault_priors,
        "activation_priors": args.activation_priors,
        "backend": args.backend,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    summaries: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.parallel_jobs)) as executor:
        futures = {executor.submit(evaluate_one, args, bench): bench for bench in benches}
        for future in as_completed(futures):
            bench = futures[future]
            try:
                summary = future.result()
            except Exception as exc:
                summary = {"benchmark_id": bench, "status": "error", "error": str(exc)}
            summaries.append(summary)
            print(json.dumps(summary, sort_keys=True), flush=True)
            (args.out_dir / "collection_status.json").write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")

    merged_path = merge_labels(args.out_dir)
    manifest["summaries"] = summaries
    manifest["merged_labels"] = str(merged_path)
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if any(summary.get("status") == "error" for summary in summaries):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
