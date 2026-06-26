"""Collect baseline and single-action TMAX fault logs for non-eval benchmarks."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATASET_ROOT = Path("/data3/pengqingsong/DFT/Dataset")
if str(DATASET_ROOT) not in sys.path:
    sys.path.insert(0, str(DATASET_ROOT))

from tpi_eval.algorithm import SelectedPoint  # noqa: E402
from tpi_eval.bench import parse_bench as parse_eval_bench  # noqa: E402
from tpi_eval.candidates import generate_candidates  # noqa: E402
from tpi_eval.random_labels import evaluate_points, make_record  # noqa: E402

from tpi_jepa.bench import parse_bench  # noqa: E402
from tpi_jepa.features import make_state_features  # noqa: E402
from tpi_jepa.graph import build_graph  # noqa: E402
from tpi_jepa.labels import DEFAULT_LABELS, find_bench_path, load_labels  # noqa: E402
from tpi_jepa.plan import (  # noqa: E402
    enumerate_candidates,
    load_checkpoint,
    score_candidate_from_latent,
)
from tpi_jepa.protocol import eval_benchmarks_from_protocol, parse_benchmark_list  # noqa: E402


PLAN_TO_TPI_EVAL_TYPE = {
    "control0": "CP0",
    "control1": "CP1",
    "observe": "OP",
}


FIELDNAMES = [
    "benchmark_id",
    "sequence_id",
    "step",
    "rank",
    "seed",
    "random_patterns",
    "candidate_count",
    "inserted_count",
    "candidate_id",
    "net",
    "type",
    "insertion_sequence",
    "total_faults",
    "detected_faults",
    "test_coverage",
    "fault_coverage",
    "pattern_count",
    "baseline_test_coverage",
    "baseline_fault_coverage",
    "delta_test_coverage",
    "delta_fault_coverage",
    "work_dir",
    "status",
    "error",
    "score_pred",
    "reward_pred",
    "fc_pred",
    "return_pred",
    "fault_list_path",
    "fault_csv_path",
    "hard_fault_summary_path",
    "hard_fault_count",
    "undetected_fault_count",
    "fault_status_counts",
    "elapsed_sec",
]

CANDIDATE_FIELDS = [
    "rank",
    "node",
    "type",
    "score_pred",
    "reward_pred",
    "fc_pred",
    "pattern_pred",
    "return_pred",
]


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if value is None:
        return ""
    return value


def _fault_fields(report: dict[str, Any] | None) -> dict[str, Any]:
    fault = (report or {}).get("fault_report") or {}
    if not isinstance(fault, dict):
        return {}
    return {
        "fault_list_path": fault.get("fault_list_path"),
        "fault_csv_path": fault.get("fault_csv_path"),
        "hard_fault_summary_path": fault.get("hard_fault_summary_path"),
        "hard_fault_count": fault.get("hard_fault_count"),
        "undetected_fault_count": fault.get("undetected_fault_count"),
        "fault_status_counts": fault.get("fault_status_counts"),
    }


def _record_dict(
    record: Any,
    *,
    report: dict[str, Any] | None,
    rank: int,
    score: dict[str, Any] | None,
    elapsed_sec: float,
) -> dict[str, Any]:
    payload = record.to_json() if hasattr(record, "to_json") else asdict(record)
    payload["rank"] = rank
    payload["elapsed_sec"] = elapsed_sec
    payload.update(_fault_fields(report))
    if score:
        for key in ("score_pred", "reward_pred", "fc_pred", "return_pred"):
            payload[key] = score.get(key)
    return payload


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_cell(row.get(field)) for field in FIELDNAMES})
    with path.with_suffix(".jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_candidate_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CANDIDATE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_cell(row.get(field)) for field in CANDIDATE_FIELDS})


def safe_path_token(value: Any) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in str(value))[:120] or "net"


def candidate_ids_by_net(bench_path: Path) -> tuple[int, dict[str, int]]:
    candidates = generate_candidates(parse_eval_bench(bench_path))
    by_net: dict[str, int] = {}
    for candidate in candidates:
        by_net.setdefault(candidate.net, candidate.id)
    return len(candidates), by_net


@torch.no_grad()
def rank_sparse_candidates(
    *,
    checkpoint: Path,
    benchmark_id: str,
    candidate_pool: int,
    top_k: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    model, config = load_checkpoint(checkpoint, device)
    bench_path = find_bench_path(benchmark_id)
    graph = build_graph(parse_bench(bench_path))
    feature_mode = str(config.get("feature_mode", "basic"))
    real_fault_priors = config.get("real_fault_priors") or config.get("real_fault_prior_path")
    relation_mode = str(config.get("relation_mode", "basic"))
    relation_depth = int(config.get("relation_depth", 8))
    candidate_strategy = str(config.get("candidate_strategy", "testability"))
    candidates = enumerate_candidates(
        graph,
        [],
        candidate_pool,
        candidate_strategy,
        real_fault_benchmark_id=benchmark_id,
        real_fault_prior_path=real_fault_priors,
    )

    base_features = make_state_features(
        graph,
        [],
        feature_mode=feature_mode,
        benchmark_id=benchmark_id,
        real_fault_prior_path=real_fault_priors,
    )[:, :-3]
    x_state = make_state_features(graph, [], base_features).to(device)
    edge_src = graph.edge_src.to(device)
    edge_dst = graph.edge_dst.to(device)
    z_state = model.online_encoder(x_state, edge_src, edge_dst)

    scored: list[dict[str, Any]] = []
    for node, action_type in candidates:
        row = score_candidate_from_latent(
            model,
            graph,
            z_state,
            (node, action_type),
            device,
            relation_mode=relation_mode,
            relation_depth=relation_depth,
        )
        row.pop("_z_pred", None)
        row["node"] = node
        row["type"] = action_type
        scored.append(row)
    scored.sort(key=lambda row: float(row.get("reward_pred", row.get("score_pred", 0.0))), reverse=True)
    for rank, row in enumerate(scored[:top_k], start=1):
        row["rank"] = rank
    return scored[:top_k]


def selected_from_candidate(row: dict[str, Any], candidate_ids: dict[str, int]) -> SelectedPoint:
    net = str(row["node"])
    action_type = str(row["type"])
    return SelectedPoint(
        candidate_id=candidate_ids.get(net, -1),
        net=net,
        type=PLAN_TO_TPI_EVAL_TYPE[action_type],
        score=float(row.get("reward_pred") or row.get("score_pred") or 0.0),
    )


def parse_devices(text: str, fallback: str) -> list[str]:
    devices = [item.strip() for item in text.split(",") if item.strip()]
    if not devices:
        devices = [fallback]
    normalized = []
    for item in devices:
        normalized.append(f"cuda:{item}" if item.isdigit() else item)
    return normalized


def evaluate_one_benchmark(args: argparse.Namespace, benchmark_id: str, device_name: str) -> dict[str, Any]:
    started_bench = time.time()
    bench_path = find_bench_path(benchmark_id)
    bench_dir = args.out_dir / "benchmarks" / benchmark_id
    bench_dir.mkdir(parents=True, exist_ok=True)
    expected_rows = 1 + max(0, int(args.top_k))
    labels_path = bench_dir / "labels.csv"
    if args.resume and labels_path.is_file():
        with labels_path.open(newline="") as f:
            row_count = max(0, sum(1 for _ in f) - 1)
        if row_count >= expected_rows:
            return {
                "benchmark_id": benchmark_id,
                "status": "skipped",
                "rows": row_count,
                "elapsed_sec": 0.0,
                "labels_csv": str(labels_path),
                "topk_candidates_csv": str(bench_dir / "topk_candidates.csv"),
            }
    print(json.dumps({"benchmark_id": benchmark_id, "status": "started"}, sort_keys=True), flush=True)
    device = torch.device(device_name)
    candidate_count, candidate_ids = candidate_ids_by_net(bench_path)
    rows: list[dict[str, Any]] = []

    if args.top_k > 0:
        ranked = rank_sparse_candidates(
            checkpoint=args.checkpoint,
            benchmark_id=benchmark_id,
            candidate_pool=args.candidate_pool,
            top_k=args.top_k,
            device=device,
        )
    else:
        ranked = []
    write_candidate_rows(bench_dir / "topk_candidates.csv", ranked)

    baseline_dir = bench_dir / "baseline"
    started = time.time()
    baseline_report = evaluate_points(
        circuit_path=bench_path,
        benchmark_id=benchmark_id,
        selected=[],
        work_dir=baseline_dir,
        patterns=args.patterns,
        backend=args.backend,
        tmax_bin=args.tmax_bin,
        atalanta_bin=args.atalanta_bin,
        timeout_sec=args.timeout_sec,
        seed=args.seed,
        force=args.force,
        dry_run=args.dry_run,
    )
    baseline_record = make_record(
        benchmark_id=benchmark_id,
        sequence_id=-1,
        step=0,
        seed=args.seed,
        patterns=args.patterns,
        candidate_count=candidate_count,
        selected=[],
        report=baseline_report,
        baseline_report=baseline_report,
        work_dir=baseline_dir,
    )
    rows.append(_record_dict(baseline_record, report=baseline_report, rank=0, score=None, elapsed_sec=time.time() - started))

    for rank, cand in enumerate(ranked, start=1):
        selected = [selected_from_candidate(cand, candidate_ids)]
        work_dir = bench_dir / f"rank_{rank:04d}__{safe_path_token(cand['type'])}__{safe_path_token(cand['node'])}"
        started = time.time()
        error = None
        try:
            report = evaluate_points(
                circuit_path=bench_path,
                benchmark_id=benchmark_id,
                selected=selected,
                work_dir=work_dir,
                patterns=args.patterns,
                backend=args.backend,
                tmax_bin=args.tmax_bin,
                atalanta_bin=args.atalanta_bin,
                timeout_sec=args.timeout_sec,
                seed=args.seed,
                force=args.force,
                dry_run=args.dry_run,
            )
        except Exception as exc:
            error = str(exc)
            report = {"status": "error", "fault_report": None}
        record = make_record(
            benchmark_id=benchmark_id,
            sequence_id=0,
            step=rank,
            seed=args.seed,
            patterns=args.patterns,
            candidate_count=candidate_count,
            selected=selected,
            report=report,
            baseline_report=baseline_report,
            work_dir=work_dir,
            error=error,
        )
        rows.append(_record_dict(record, report=report, rank=rank, score=cand, elapsed_sec=time.time() - started))

    write_rows(bench_dir / "labels.csv", rows)
    summary = {
        "benchmark_id": benchmark_id,
        "status": "ok" if all(row.get("status") == "ok" for row in rows) else "partial",
        "rows": len(rows),
        "elapsed_sec": round(time.time() - started_bench, 3),
        "labels_csv": str(bench_dir / "labels.csv"),
        "topk_candidates_csv": str(bench_dir / "topk_candidates.csv"),
    }
    (bench_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def benchmark_ids(args: argparse.Namespace) -> list[str]:
    if args.benchmarks:
        return sorted(parse_benchmark_list(args.benchmarks))
    rows = load_labels(args.labels)
    benches = {row.benchmark_id for row in rows}
    excluded = eval_benchmarks_from_protocol(args.eval_protocol)
    excluded.update(parse_benchmark_list(args.extra_exclude))
    return sorted(benches - excluded)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("autoresearch/tac-framework-sweep-dev-v4/runs/fwsparse_075__ctx1__ld64__el3__do0p1__lrtn0p0__h5__rs6/best.pt"))
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--eval-protocol", default="configs/eval_protocol_coverage_only.json")
    parser.add_argument("--extra-exclude", default="")
    parser.add_argument("--benchmarks", default="")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--candidate-pool", type=int, default=128)
    parser.add_argument("--patterns", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--timeout-sec", type=int, default=600)
    parser.add_argument("--parallel-jobs", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--devices", default="")
    parser.add_argument("--backend", choices=["tmax", "atalanta-bist"], default="tmax")
    parser.add_argument("--tmax-bin", default="/data3/pengqingsong/synopsys/txs/O-2018.06-SP1/bin/tmax")
    parser.add_argument(
        "--atalanta-bin",
        default="/data3/pengqingsong/DFT/DeepTPI-project/external/DeepTPI/src/external/Atalanta_BIST/atalanta",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    benches = benchmark_ids(args)
    manifest = {
        "checkpoint": str(args.checkpoint),
        "benchmarks": benches,
        "top_k": args.top_k,
        "candidate_pool": args.candidate_pool,
        "patterns": args.patterns,
        "seed": args.seed,
        "timeout_sec": args.timeout_sec,
        "parallel_jobs": args.parallel_jobs,
        "backend": args.backend,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    summaries: list[dict[str, Any]] = []
    devices = parse_devices(args.devices, args.device)
    with ThreadPoolExecutor(max_workers=max(1, args.parallel_jobs)) as executor:
        futures = {
            executor.submit(evaluate_one_benchmark, args, bench, devices[idx % len(devices)]): bench
            for idx, bench in enumerate(benches)
        }
        for future in as_completed(futures):
            bench = futures[future]
            try:
                summary = future.result()
            except Exception as exc:
                summary = {"benchmark_id": bench, "status": "error", "error": str(exc)}
            summaries.append(summary)
            print(json.dumps(summary, sort_keys=True), flush=True)
            (args.out_dir / "collection_status.json").write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")

    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    manifest["summaries"] = summaries
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if any(row.get("status") == "error" for row in summaries):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
