"""Evaluate a planned TPI sequence with the existing TMAX label backend."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

DATASET_ROOT = Path(os.environ.get("DFT_ROOT", "/data4/pengqingsong/DFT")) / "Dataset"
if str(DATASET_ROOT) not in sys.path:
    sys.path.insert(0, str(DATASET_ROOT))

from tpi_eval.algorithm import SelectedPoint  # noqa: E402
from tpi_eval.candidates import generate_candidates  # noqa: E402
from tpi_eval.random_labels import (  # noqa: E402
    cleanup_work_dir,
    evaluate_points,
    make_record,
)

from .labels import find_bench_path


PLAN_TO_TPI_EVAL_TYPE = {
    "control0": "CP0",
    "control1": "CP1",
    "observe": "OP",
    "CP0": "CP0",
    "CP1": "CP1",
    "OP": "OP",
}


FIELDNAMES = [
    "benchmark_id",
    "sequence_id",
    "step",
    "seed",
    "random_patterns",
    "candidate_count",
    "inserted_count",
    "candidate_id",
    "net",
    "type",
    "insertion_sequence",
    "source_bench_sha256",
    "inserted_bench_sha256",
    "total_faults",
    "detected_faults",
    "test_coverage",
    "fault_coverage",
    "pattern_count",
    "effective_pattern_count",
    "baseline_test_coverage",
    "baseline_fault_coverage",
    "baseline_pattern_count",
    "baseline_effective_pattern_count",
    "delta_test_coverage",
    "delta_fault_coverage",
    "delta_pattern_count",
    "delta_effective_pattern_count",
    "fault_list_path",
    "fault_csv_path",
    "hard_fault_summary_path",
    "hard_fault_count",
    "undetected_fault_count",
    "fault_status_counts",
    "work_dir",
    "status",
    "error",
    "source_plan_csv",
    "pred_score",
    "pred_fc",
    "pred_pattern",
    "elapsed_sec",
]


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return value


def _read_plan(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _candidate_ids_by_net(bench_path: Path) -> tuple[int, dict[str, int]]:
    from tpi_eval.bench import parse_bench

    candidates = generate_candidates(parse_bench(bench_path))
    by_net: dict[str, int] = {}
    for candidate in candidates:
        by_net.setdefault(candidate.net, candidate.id)
    return len(candidates), by_net


def _point_from_plan_row(row: dict[str, str], candidate_ids: dict[str, int]) -> SelectedPoint:
    net = (row.get("node") or row.get("net") or "").strip()
    action = (row.get("type") or "").strip()
    if not net:
        raise ValueError(f"plan row missing node/net: {row}")
    if action not in PLAN_TO_TPI_EVAL_TYPE:
        raise ValueError(f"unsupported plan action type {action!r} in row: {row}")
    return SelectedPoint(
        candidate_id=candidate_ids.get(net, -1),
        net=net,
        type=PLAN_TO_TPI_EVAL_TYPE[action],
        score=float(row["score_pred"]) if row.get("score_pred") else None,
    )


def _record_dict(
    record: Any,
    *,
    plan_csv: Path,
    plan_row: dict[str, str] | None,
    elapsed: float,
    report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = record.to_json() if hasattr(record, "to_json") else asdict(record)
    fault_report = (report or {}).get("fault_report") or payload.get("fault_report") or {}
    if isinstance(fault_report, dict):
        payload["fault_list_path"] = fault_report.get("fault_list_path")
        payload["fault_csv_path"] = fault_report.get("fault_csv_path")
        payload["hard_fault_summary_path"] = fault_report.get("hard_fault_summary_path")
        payload["hard_fault_count"] = fault_report.get("hard_fault_count")
        payload["undetected_fault_count"] = fault_report.get("undetected_fault_count")
        payload["fault_status_counts"] = fault_report.get("fault_status_counts")
    payload["source_plan_csv"] = str(plan_csv)
    payload["pred_score"] = plan_row.get("score_pred") if plan_row else ""
    payload["pred_fc"] = plan_row.get("fc_pred") if plan_row else ""
    payload["pred_pattern"] = plan_row.get("pattern_pred") if plan_row else ""
    payload["elapsed_sec"] = elapsed
    return payload


def _write_outputs(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "labels.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(row.get(key)) for key in FIELDNAMES})
    with (out_dir / "labels.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    compact = []
    for row in rows:
        compact.append(
            {
                "step": row.get("step"),
                "net": row.get("net"),
                "type": row.get("type"),
                "test_coverage": row.get("test_coverage"),
                "fault_coverage": row.get("fault_coverage"),
                "pattern_count": row.get("pattern_count"),
                "delta_test_coverage": row.get("delta_test_coverage"),
                "delta_fault_coverage": row.get("delta_fault_coverage"),
                "delta_pattern_count": row.get("delta_pattern_count"),
                "status": row.get("status"),
                "error": row.get("error"),
            }
        )
    (out_dir / "summary.json").write_text(json.dumps(compact, indent=2, ensure_ascii=False) + "\n")


def _write_step_training_data(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    """Persist non-baseline step labels as a training-data friendly JSONL."""
    with (out_dir / "step_training_labels.jsonl").open("w") as f:
        for row in rows:
            if int(row.get("step") or 0) <= 0:
                continue
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def evaluate_plan(
    *,
    benchmark_id: str,
    plan_csv: Path,
    out_dir: Path,
    patterns: int,
    seed: int,
    backend: str,
    tmax_bin: str,
    atalanta_bin: str,
    timeout_sec: int,
    force: bool,
    dry_run: bool,
    cleanup_workdir: bool,
    eval_step_mode: str = "all",
    save_step_training_data: bool = False,
) -> list[dict[str, Any]]:
    if eval_step_mode not in {"all", "final"}:
        raise ValueError(f"unsupported eval_step_mode={eval_step_mode!r}; expected 'all' or 'final'")
    bench_path = find_bench_path(benchmark_id)
    plan_rows = _read_plan(plan_csv)
    candidate_count, candidate_ids = _candidate_ids_by_net(bench_path)

    sequence_id = 0
    rows: list[dict[str, Any]] = []
    baseline_dir = out_dir / benchmark_id / "baseline"
    started = time.time()
    baseline_report = evaluate_points(
        circuit_path=bench_path,
        benchmark_id=benchmark_id,
        selected=[],
        work_dir=baseline_dir,
        patterns=patterns,
        backend=backend,
        tmax_bin=tmax_bin,
        atalanta_bin=atalanta_bin,
        timeout_sec=timeout_sec,
        seed=seed,
        force=force,
        dry_run=dry_run,
    )
    baseline_record = make_record(
        benchmark_id=benchmark_id,
        sequence_id=-1,
        step=0,
        seed=seed,
        patterns=patterns,
        candidate_count=candidate_count,
        selected=[],
        report=baseline_report,
        baseline_report=baseline_report,
        work_dir=baseline_dir,
    )
    rows.append(
        _record_dict(
            baseline_record,
            plan_csv=plan_csv,
            plan_row=None,
            elapsed=time.time() - started,
            report=baseline_report,
        )
    )
    baseline_dir.mkdir(parents=True, exist_ok=True)
    (baseline_dir / "label.json").write_text(
        json.dumps(rows[-1], indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    if cleanup_workdir:
        cleanup_work_dir(baseline_dir)

    selected: list[SelectedPoint] = []
    if eval_step_mode == "final":
        for plan_row in plan_rows:
            selected.append(_point_from_plan_row(plan_row, candidate_ids))
        step = len(plan_rows)
        plan_row = plan_rows[-1] if plan_rows else None
        work_dir = out_dir / benchmark_id / f"seq_{sequence_id:04d}" / f"step_{step:04d}"
        started = time.time()
        error = None
        try:
            report = evaluate_points(
                circuit_path=bench_path,
                benchmark_id=benchmark_id,
                selected=selected.copy(),
                work_dir=work_dir,
                patterns=patterns,
                backend=backend,
                tmax_bin=tmax_bin,
                atalanta_bin=atalanta_bin,
                timeout_sec=timeout_sec,
                seed=seed,
                force=force,
                dry_run=dry_run,
            )
        except Exception as exc:
            error = str(exc)
            report = {"status": "error", "fault_report": None}
        record = make_record(
            benchmark_id=benchmark_id,
            sequence_id=sequence_id,
            step=step,
            seed=seed,
            patterns=patterns,
            candidate_count=candidate_count,
            selected=selected.copy(),
            report=report,
            baseline_report=baseline_report,
            work_dir=work_dir,
            error=error,
        )
        payload = _record_dict(
            record,
            plan_csv=plan_csv,
            plan_row=plan_row,
            elapsed=time.time() - started,
            report=report,
        )
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "label.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
        rows.append(payload)
        if cleanup_workdir:
            cleanup_work_dir(work_dir)
        _write_outputs(out_dir, rows)
        if save_step_training_data:
            _write_step_training_data(out_dir, rows)
        return rows

    for step, plan_row in enumerate(plan_rows, start=1):
        selected.append(_point_from_plan_row(plan_row, candidate_ids))
        work_dir = out_dir / benchmark_id / f"seq_{sequence_id:04d}" / f"step_{step:04d}"
        started = time.time()
        error = None
        try:
            report = evaluate_points(
                circuit_path=bench_path,
                benchmark_id=benchmark_id,
                selected=selected.copy(),
                work_dir=work_dir,
                patterns=patterns,
                backend=backend,
                tmax_bin=tmax_bin,
                atalanta_bin=atalanta_bin,
                timeout_sec=timeout_sec,
                seed=seed,
                force=force,
                dry_run=dry_run,
            )
        except Exception as exc:
            error = str(exc)
            report = {"status": "error", "fault_report": None}
        record = make_record(
            benchmark_id=benchmark_id,
            sequence_id=sequence_id,
            step=step,
            seed=seed,
            patterns=patterns,
            candidate_count=candidate_count,
            selected=selected.copy(),
            report=report,
            baseline_report=baseline_report,
            work_dir=work_dir,
            error=error,
        )
        payload = _record_dict(
            record,
            plan_csv=plan_csv,
            plan_row=plan_row,
            elapsed=time.time() - started,
            report=report,
        )
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "label.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
        rows.append(payload)
        if cleanup_workdir:
            cleanup_work_dir(work_dir)

    _write_outputs(out_dir, rows)
    if save_step_training_data:
        _write_step_training_data(out_dir, rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a TPI-JEPA plan with a fault-simulation backend.")
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--plan-csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--patterns", type=int, default=300000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--backend", choices=["tmax", "atalanta-bist"], default="tmax")
    parser.add_argument("--tmax-bin", default="/data3/pengqingsong/synopsys/txs/O-2018.06-SP1/bin/tmax")
    parser.add_argument(
        "--atalanta-bin",
        default=str(
            Path(os.environ.get("DFT_ROOT", "/data4/pengqingsong/DFT"))
            / "tool/atalanta_bist_with_ufaults/atalanta"
        ),
    )
    parser.add_argument("--timeout-sec", type=int, default=7200)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cleanup-workdir", action="store_true")
    parser.add_argument(
        "--eval-step-mode",
        choices=["final", "all"],
        default="final",
        help="Use 'final' for baseline+final TC only, or 'all' to evaluate every intermediate insertion step.",
    )
    parser.add_argument(
        "--save-step-training-data",
        action="store_true",
        help="Write step_training_labels.jsonl with non-baseline step records. Use with --eval-step-mode all for dense labels.",
    )
    args = parser.parse_args()

    rows = evaluate_plan(
        benchmark_id=args.benchmark_id,
        plan_csv=args.plan_csv,
        out_dir=args.out_dir,
        patterns=args.patterns,
        seed=args.seed,
        backend=args.backend,
        tmax_bin=args.tmax_bin,
        atalanta_bin=args.atalanta_bin,
        timeout_sec=args.timeout_sec,
        force=args.force,
        dry_run=args.dry_run,
        cleanup_workdir=args.cleanup_workdir,
        eval_step_mode=args.eval_step_mode,
        save_step_training_data=args.save_step_training_data,
    )
    print(
        json.dumps(
            {
                "records": len(rows),
                "labels_csv": str(args.out_dir / "labels.csv"),
                "summary_json": str(args.out_dir / "summary.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
