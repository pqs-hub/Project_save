"""Replay existing TPI sequences and relabel them with a chosen fault backend."""

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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATASET_ROOT = Path("/data3/pengqingsong/DFT/Dataset")
if str(DATASET_ROOT) not in sys.path:
    sys.path.insert(0, str(DATASET_ROOT))

from tpi_eval.algorithm import SelectedPoint  # noqa: E402
from tpi_eval.bench import parse_bench as parse_eval_bench  # noqa: E402
from tpi_eval.candidates import generate_candidates  # noqa: E402
from tpi_eval.random_labels import cleanup_work_dir, evaluate_points, make_record  # noqa: E402

from tpi_jepa.evaluate_plan_tmax import FIELDNAMES  # noqa: E402
from tpi_jepa.labels import find_bench_path  # noqa: E402


TYPE_MAP = {
    "control0": "CP0",
    "control1": "CP1",
    "observe": "OP",
    "CP0": "CP0",
    "CP1": "CP1",
    "OP": "OP",
}


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


def _record_dict(record: Any, *, report: dict[str, Any] | None, elapsed: float) -> dict[str, Any]:
    payload = record.to_json() if hasattr(record, "to_json") else asdict(record)
    payload.update(_fault_fields(report))
    payload["elapsed_sec"] = elapsed
    return payload


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(FIELDNAMES)
    for extra in ("source_sequence_id", "source_label_csv"):
        if extra not in fieldnames:
            fieldnames.append(extra)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_cell(row.get(field)) for field in fieldnames})
    with path.with_suffix(".jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_sequences(labels: Path, max_sequences: int | None = None) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    with labels.open(newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("step") or "0") == "0":
                continue
            key = f"{row.get('benchmark_id')}::{row.get('sequence_id')}"
            grouped.setdefault(key, []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row.get("step") or 0))
    items = sorted(grouped.items())
    if max_sequences is not None and max_sequences > 0:
        items = items[:max_sequences]
    return dict(items)


def candidate_ids_by_net(bench_path: Path) -> tuple[int, dict[str, int]]:
    candidates = generate_candidates(parse_eval_bench(bench_path))
    by_net: dict[str, int] = {}
    for candidate in candidates:
        by_net.setdefault(candidate.net, candidate.id)
    return len(candidates), by_net


def selected_point(row: dict[str, str], candidate_ids: dict[str, int]) -> SelectedPoint:
    net = str(row["net"])
    action_type = TYPE_MAP[str(row["type"])]
    return SelectedPoint(candidate_id=candidate_ids.get(net, -1), net=net, type=action_type, score=None)


def relabel_one(args: argparse.Namespace, key: str, rows_in: list[dict[str, str]]) -> dict[str, Any]:
    benchmark_id = rows_in[0]["benchmark_id"]
    sequence_id = rows_in[0]["sequence_id"]
    safe_sequence = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in sequence_id)
    seq_dir = args.out_dir / "sequences" / benchmark_id / safe_sequence
    labels_path = seq_dir / "labels.csv"
    if args.resume and labels_path.is_file():
        rows = list(csv.DictReader(labels_path.open()))
        if len(rows) >= len(rows_in) and all(row.get("status") == "ok" for row in rows):
            return {"benchmark_id": benchmark_id, "sequence_id": sequence_id, "status": "skipped", "rows": len(rows)}

    started_seq = time.time()
    bench_path = find_bench_path(benchmark_id)
    candidate_count, candidate_ids = candidate_ids_by_net(bench_path)

    baseline_dir = seq_dir / "baseline"
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
    if args.cleanup_workdir:
        cleanup_work_dir(baseline_dir)

    out_rows: list[dict[str, Any]] = []
    selected: list[SelectedPoint] = []
    for row_in in rows_in:
        selected.append(selected_point(row_in, candidate_ids))
        step = int(row_in.get("step") or len(selected))
        work_dir = seq_dir / f"step_{step:04d}"
        started = time.time()
        error = None
        try:
            report = evaluate_points(
                circuit_path=bench_path,
                benchmark_id=benchmark_id,
                selected=selected.copy(),
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
            sequence_id=sequence_id,
            step=step,
            seed=args.seed,
            patterns=args.patterns,
            candidate_count=candidate_count,
            selected=selected.copy(),
            report=report,
            baseline_report=baseline_report,
            work_dir=work_dir,
            error=error,
        )
        payload = _record_dict(record, report=report, elapsed=time.time() - started)
        payload["source_sequence_id"] = row_in.get("source_sequence_id") or row_in.get("sequence_id")
        payload["source_label_csv"] = row_in.get("source_label_csv") or str(args.labels)
        out_rows.append(payload)
        if args.cleanup_workdir:
            cleanup_work_dir(work_dir)

    write_rows(labels_path, out_rows)
    return {
        "benchmark_id": benchmark_id,
        "sequence_id": sequence_id,
        "status": "ok" if all(row.get("status") == "ok" for row in out_rows) else "partial",
        "rows": len(out_rows),
        "elapsed_sec": round(time.time() - started_seq, 3),
        "labels_csv": str(labels_path),
    }


def merge_labels(out_dir: Path) -> Path:
    files = sorted(out_dir.glob("sequences/*/*/labels.csv"))
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    for path in files:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for field in reader.fieldnames or []:
                if field not in fieldnames:
                    fieldnames.append(field)
            rows.extend(dict(row) for row in reader)
    merged = out_dir / "labels.csv"
    with merged.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=["tmax", "atalanta-bist"], default="atalanta-bist")
    parser.add_argument("--patterns", type=int, default=300000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--timeout-sec", type=int, default=14400)
    parser.add_argument("--parallel-jobs", type=int, default=4)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--tmax-bin", default="/data3/pengqingsong/synopsys/txs/O-2018.06-SP1/bin/tmax")
    parser.add_argument(
        "--atalanta-bin",
        default="/data3/pengqingsong/DFT/DeepTPI-project/external/DeepTPI/src/external/Atalanta_BIST/atalanta",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cleanup-workdir", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sequences = load_sequences(args.labels, args.max_sequences)
    manifest = {
        "labels": str(args.labels),
        "out_dir": str(args.out_dir),
        "backend": args.backend,
        "patterns": args.patterns,
        "sequence_count": len(sequences),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    summaries: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.parallel_jobs)) as executor:
        futures = {executor.submit(relabel_one, args, key, rows): key for key, rows in sequences.items()}
        for future in as_completed(futures):
            key = futures[future]
            try:
                summary = future.result()
            except Exception as exc:
                benchmark_id, sequence_id = key.split("::", 1)
                summary = {
                    "benchmark_id": benchmark_id,
                    "sequence_id": sequence_id,
                    "status": "error",
                    "error": str(exc),
                }
            summaries.append(summary)
            print(json.dumps(summary, sort_keys=True), flush=True)
            (args.out_dir / "collection_status.json").write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")

    merged = merge_labels(args.out_dir)
    manifest["merged_labels"] = str(merged)
    manifest["summaries"] = summaries
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if any(row.get("status") == "error" for row in summaries):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
