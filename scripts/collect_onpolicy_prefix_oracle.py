#!/usr/bin/env python3
"""Collect real one-step action labels at frozen on-policy plan prefixes.

Each candidate is evaluated as ``prefix + candidate`` and compared with the
same prefix report.  Consequently ``oracle_delta_tc`` is a true one-step
marginal label rather than coverage relative to the empty plan.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import asdict, dataclass
import hashlib
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

from tpi_eval.random_labels import cleanup_work_dir, evaluate_points, make_record  # noqa: E402

from scripts.relabel_sequences_with_backend import (  # noqa: E402
    candidate_ids_by_net,
    selected_point,
)
from tpi_jepa.bench import parse_bench  # noqa: E402
from tpi_jepa.graph import build_graph  # noqa: E402
from tpi_jepa.labels import find_bench_path  # noqa: E402
from tpi_jepa.plan import clear_planner_caches, enumerate_candidates  # noqa: E402
from tpi_jepa.protocol import eval_benchmarks_from_protocol  # noqa: E402


ACTION_TO_CANONICAL = {
    "CP0": "control0",
    "CP1": "control1",
    "OP": "observe",
    "control0": "control0",
    "control1": "control1",
    "observe": "observe",
}

ORACLE_FIELDS = [
    "benchmark_id",
    "state_id",
    "sequence_id",
    "prefix_step",
    "state_actions",
    "prefix_actions",
    "candidate_strategy",
    "candidate_rank",
    "candidate_pool_rank",
    "candidate_pool_size",
    "node",
    "type",
    "action_key",
    "onpolicy_chosen",
    "onpolicy_in_pool",
    "status",
    "oracle_delta_tc",
    "oracle_delta_fault_coverage",
    "oracle_delta_pattern_count",
    "oracle_test_coverage",
    "oracle_fault_coverage",
    "oracle_pattern_count",
    "oracle_hard_fault_count",
    "oracle_undetected_fault_count",
    "oracle_undetected_sa0_count",
    "oracle_undetected_sa1_count",
    "oracle_hard_reduction_total",
    "oracle_hard_reduction_sa0",
    "oracle_hard_reduction_sa1",
    "prefix_test_coverage",
    "prefix_fault_coverage",
    "prefix_pattern_count",
    "prefix_hard_fault_count",
    "prefix_undetected_fault_count",
    "prefix_undetected_sa0_count",
    "prefix_undetected_sa1_count",
    "oracle_error",
    "eval_dir",
    "source_plan_csv",
    "source_plan_sha256",
    "task_signature",
]


@dataclass(frozen=True)
class PrefixTask:
    benchmark_id: str
    state_id: str
    sequence_id: str
    prefix_step: int
    prefix_actions: tuple[tuple[str, str], ...]
    candidates: tuple[tuple[str, str], ...]
    candidate_pool_ranks: tuple[int, ...]
    candidate_pool_size: int
    onpolicy_action: tuple[str, str]
    onpolicy_in_pool: bool
    candidate_strategy: str
    source_plan_csv: str
    source_plan_sha256: str


def parse_int_list(value: str) -> list[int]:
    values = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not values or values[0] < 0:
        raise ValueError("--prefix-steps must contain non-negative integers")
    return values


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_action(action_type: str) -> str:
    try:
        return ACTION_TO_CANONICAL[action_type.strip()]
    except KeyError as exc:
        raise ValueError(f"unsupported TPI action type {action_type!r}") from exc


def read_plan(path: Path) -> list[tuple[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    ordered = sorted(rows, key=lambda row: int(row.get("step") or 0))
    actions: list[tuple[str, str]] = []
    for expected_step, row in enumerate(ordered, start=1):
        step = int(row.get("step") or 0)
        if step != expected_step:
            raise ValueError(f"non-contiguous plan steps in {path}: expected {expected_step}, got {step}")
        node = str(row.get("node") or row.get("net") or "").strip()
        if not node:
            raise ValueError(f"plan row missing node/net in {path}: {row}")
        actions.append((node, canonical_action(str(row.get("type") or ""))))
    if not actions:
        raise ValueError(f"empty plan: {path}")
    return actions


def load_plan_paths(
    plans_dir: Path,
    training_manifest: Path,
    eval_protocol: Path,
    max_benchmarks: int | None = None,
) -> list[tuple[str, Path]]:
    payload = json.loads(training_manifest.read_text())
    accepted = {str(item) for item in payload.get("accepted_benchmarks", []) if str(item)}
    if not accepted:
        raise ValueError(f"training manifest has no accepted_benchmarks: {training_manifest}")
    paths = sorted(plans_dir.glob("*.csv"))
    if not paths:
        raise ValueError(f"no plan CSV files under {plans_dir}")
    by_benchmark = {path.stem: path for path in paths}
    unexpected = set(by_benchmark) - accepted
    if unexpected:
        raise ValueError(f"plans not admitted by training manifest: {sorted(unexpected)}")
    forbidden = eval_benchmarks_from_protocol(eval_protocol)
    leaked = set(by_benchmark) & forbidden
    if leaked:
        raise ValueError(f"refusing evaluation-protocol circuits in oracle collection: {sorted(leaked)}")
    jobs = sorted(by_benchmark.items())
    if max_benchmarks is not None and max_benchmarks > 0:
        jobs = jobs[:max_benchmarks]
    return jobs


def choose_candidates(
    pool: list[tuple[str, str]],
    onpolicy_action: tuple[str, str],
    limit: int,
) -> tuple[list[tuple[str, str]], list[int], bool]:
    """Stratify the pool across planner choice, types, and pool-tail negatives."""

    if limit < 2:
        raise ValueError("--actions-per-prefix must be at least 2")
    if len(pool) < 2:
        raise ValueError("candidate pool must contain at least two actions")
    pool_rank = {candidate: rank for rank, candidate in enumerate(pool, start=1)}
    onpolicy_in_pool = onpolicy_action in pool_rank
    chosen: list[tuple[str, str]] = []

    def add(candidate: tuple[str, str]) -> None:
        if candidate not in chosen and len(chosen) < limit:
            chosen.append(candidate)

    if onpolicy_in_pool:
        add(onpolicy_action)

    # The previous selected-only trajectories over-represented control0 even
    # though its measured marginal TC was often negative.  Allocate the
    # counterfactual budget evenly by action type, drawing from both the head
    # and tail of the exact same recall pool so every group contains useful
    # alternatives and hard negatives.
    action_types = ("control0", "control1", "observe")
    quotas = {action_type: limit // len(action_types) for action_type in action_types}
    for action_type in action_types[: limit % len(action_types)]:
        quotas[action_type] += 1
    for action_type in action_types:
        typed = [candidate for candidate in pool if candidate[1] == action_type]
        typed_order: list[tuple[str, str]] = []
        left, right = 0, len(typed) - 1
        while left <= right:
            typed_order.append(typed[left])
            left += 1
            if left <= right:
                typed_order.append(typed[right])
                right -= 1
        while sum(candidate[1] == action_type for candidate in chosen) < quotas[action_type] and typed_order:
            add(typed_order.pop(0))

    # Fill any unallocated slots deterministically if one action type did not
    # have enough pool members.  Preserve both high-recall and tail examples.
    add(pool[-1])
    for candidate in pool:
        add(candidate)
    for candidate in reversed(pool):
        add(candidate)
    chosen.sort(key=pool_rank.__getitem__)
    ranks = [pool_rank[candidate] for candidate in chosen]
    return chosen, ranks, onpolicy_in_pool


def task_signature(task: PrefixTask) -> str:
    payload = asdict(task)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def build_tasks(args: argparse.Namespace) -> list[PrefixTask]:
    tasks: list[PrefixTask] = []
    skipped_states: list[dict[str, Any]] = []
    prefix_steps = parse_int_list(args.prefix_steps)
    for benchmark_id, plan_path in load_plan_paths(
        args.plans_dir,
        args.training_manifest,
        args.eval_protocol,
        args.max_benchmarks,
    ):
        plan = read_plan(plan_path)
        graph = build_graph(parse_bench(find_bench_path(benchmark_id)))
        clear_planner_caches()
        for prefix_step in prefix_steps:
            if prefix_step >= len(plan):
                skipped_states.append(
                    {
                        "benchmark_id": benchmark_id,
                        "prefix_step": prefix_step,
                        "reason": "prefix_not_available",
                        "plan_steps": len(plan),
                    }
                )
                continue
            prefix = plan[:prefix_step]
            onpolicy_action = plan[prefix_step]
            pool = enumerate_candidates(
                graph,
                prefix,
                args.candidate_pool_size,
                args.candidate_strategy,
                real_fault_benchmark_id=benchmark_id,
                real_fault_prior_path=args.real_fault_priors,
                activation_prior_path=args.activation_priors,
                candidate_cache_dir=args.candidate_cache_dir,
                candidate_sample_seed=args.candidate_sample_seed,
            )
            if len(pool) < args.actions_per_prefix:
                skipped_states.append(
                    {
                        "benchmark_id": benchmark_id,
                        "prefix_step": prefix_step,
                        "reason": "insufficient_candidate_pool",
                        "candidate_pool_size": len(pool),
                        "required_candidates": args.actions_per_prefix,
                        "onpolicy_action": list(onpolicy_action),
                        "plan_steps": len(plan),
                    }
                )
                continue
            candidates, ranks, in_pool = choose_candidates(pool, onpolicy_action, args.actions_per_prefix)
            if not in_pool and not args.allow_onpolicy_outside_pool:
                raise ValueError(
                    f"on-policy action {onpolicy_action} missing from recomputed pool "
                    f"for {benchmark_id} prefix={prefix_step}; candidate settings drifted"
                )
            if not in_pool:
                candidates[-1] = onpolicy_action
                ranks[-1] = 0
            tasks.append(
                PrefixTask(
                    benchmark_id=benchmark_id,
                    state_id=f"prefix_{prefix_step:04d}",
                    sequence_id=f"onpolicy:{benchmark_id}",
                    prefix_step=prefix_step,
                    prefix_actions=tuple(prefix),
                    candidates=tuple(candidates),
                    candidate_pool_ranks=tuple(ranks),
                    candidate_pool_size=len(pool),
                    onpolicy_action=onpolicy_action,
                    onpolicy_in_pool=in_pool,
                    candidate_strategy=args.candidate_strategy,
                    source_plan_csv=str(plan_path),
                    source_plan_sha256=file_sha256(plan_path),
                )
            )
    if not tasks:
        raise ValueError("no prefix tasks remained after applying --prefix-steps")
    args.skipped_states = skipped_states
    return tasks


def _json_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if value is None:
        return ""
    return value


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ORACLE_FIELDS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _json_cell(row.get(field)) for field in ORACLE_FIELDS})


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _fault(report: dict[str, Any] | None) -> dict[str, Any]:
    fault = (report or {}).get("fault_report") or {}
    return fault if isinstance(fault, dict) else {}


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in value)


def _reduction(before: Any, after: Any) -> int | None:
    if before is None or after is None:
        return None
    try:
        return int(before) - int(after)
    except (TypeError, ValueError):
        return None


def evaluate_state(
    args: argparse.Namespace,
    task: PrefixTask,
    bench_path: Path,
    candidate_count: int,
    candidate_ids: dict[str, int],
) -> dict[str, Any]:
    signature = task_signature(task)
    state_dir = args.out_dir / "states" / task.benchmark_id / task.state_id
    rows_path = state_dir / "oracle_actions.tsv"
    if args.resume and not args.force and rows_path.is_file():
        existing = read_tsv(rows_path)
        expected_keys = {f"{node}::{action_type}" for node, action_type in task.candidates}
        if (
            len(existing) == len(task.candidates)
            and {row.get("action_key") for row in existing} == expected_keys
            and all(row.get("task_signature") == signature and row.get("status") == "ok" for row in existing)
        ):
            return {"state_id": task.state_id, "status": "skipped", "rows": len(existing)}

    print(
        json.dumps(
            {
                "benchmark_id": task.benchmark_id,
                "state_id": task.state_id,
                "prefix_step": task.prefix_step,
                "candidate_count": len(task.candidates),
                "status": "started",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    state_dir.mkdir(parents=True, exist_ok=True)
    prefix_points = [
        selected_point({"net": node, "type": action_type}, candidate_ids)
        for node, action_type in task.prefix_actions
    ]
    prefix_dir = state_dir / "prefix"
    prefix_report = evaluate_points(
        circuit_path=bench_path,
        benchmark_id=task.benchmark_id,
        selected=prefix_points,
        work_dir=prefix_dir,
        patterns=args.patterns,
        backend=args.backend,
        tmax_bin=args.tmax_bin,
        atalanta_bin=args.atalanta_bin,
        timeout_sec=args.timeout_sec,
        seed=args.seed,
        force=args.force,
        dry_run=args.dry_run,
    )
    prefix_fault = _fault(prefix_report)
    rows: list[dict[str, Any]] = []
    for candidate_rank, ((node, action_type), pool_rank) in enumerate(
        zip(task.candidates, task.candidate_pool_ranks), start=1
    ):
        action_key = f"{node}::{action_type}"
        action_dir = state_dir / "actions" / f"{candidate_rank:03d}_{_safe_name(action_key)}"
        error = None
        try:
            point = selected_point({"net": node, "type": action_type}, candidate_ids)
            report = evaluate_points(
                circuit_path=bench_path,
                benchmark_id=task.benchmark_id,
                selected=[*prefix_points, point],
                work_dir=action_dir,
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
            benchmark_id=task.benchmark_id,
            sequence_id=0,
            step=task.prefix_step + 1,
            seed=args.seed,
            patterns=args.patterns,
            candidate_count=candidate_count,
            selected=[*prefix_points, selected_point({"net": node, "type": action_type}, candidate_ids)],
            report=report,
            baseline_report=prefix_report,
            work_dir=action_dir,
            error=error,
        )
        fault = _fault(report)
        rows.append(
            {
                "benchmark_id": task.benchmark_id,
                "state_id": task.state_id,
                "sequence_id": task.sequence_id,
                "prefix_step": task.prefix_step,
                "state_actions": [list(action) for action in task.prefix_actions],
                "prefix_actions": [list(action) for action in task.prefix_actions],
                "candidate_strategy": task.candidate_strategy,
                "candidate_rank": candidate_rank,
                "candidate_pool_rank": pool_rank,
                "candidate_pool_size": task.candidate_pool_size,
                "node": node,
                "type": action_type,
                "action_key": action_key,
                "onpolicy_chosen": int((node, action_type) == task.onpolicy_action),
                "onpolicy_in_pool": int(task.onpolicy_in_pool),
                "status": record.status,
                "oracle_delta_tc": record.delta_test_coverage,
                "oracle_delta_fault_coverage": record.delta_fault_coverage,
                "oracle_delta_pattern_count": record.delta_pattern_count,
                "oracle_test_coverage": record.test_coverage,
                "oracle_fault_coverage": record.fault_coverage,
                "oracle_pattern_count": record.pattern_count,
                "oracle_hard_fault_count": fault.get("hard_fault_count"),
                "oracle_undetected_fault_count": fault.get("undetected_fault_count"),
                "oracle_undetected_sa0_count": fault.get("undetected_sa0_count"),
                "oracle_undetected_sa1_count": fault.get("undetected_sa1_count"),
                "oracle_hard_reduction_total": _reduction(
                    prefix_fault.get("hard_fault_count"), fault.get("hard_fault_count")
                ),
                "oracle_hard_reduction_sa0": _reduction(
                    prefix_fault.get("undetected_sa0_count"), fault.get("undetected_sa0_count")
                ),
                "oracle_hard_reduction_sa1": _reduction(
                    prefix_fault.get("undetected_sa1_count"), fault.get("undetected_sa1_count")
                ),
                "prefix_test_coverage": prefix_fault.get("test_coverage"),
                "prefix_fault_coverage": prefix_fault.get("fault_coverage"),
                "prefix_pattern_count": prefix_fault.get("pattern_count"),
                "prefix_hard_fault_count": prefix_fault.get("hard_fault_count"),
                "prefix_undetected_fault_count": prefix_fault.get("undetected_fault_count"),
                "prefix_undetected_sa0_count": prefix_fault.get("undetected_sa0_count"),
                "prefix_undetected_sa1_count": prefix_fault.get("undetected_sa1_count"),
                "oracle_error": error,
                "eval_dir": str(action_dir),
                "source_plan_csv": task.source_plan_csv,
                "source_plan_sha256": task.source_plan_sha256,
                "task_signature": signature,
            }
        )
        if args.cleanup_workdir:
            cleanup_work_dir(action_dir)
    if args.cleanup_workdir:
        cleanup_work_dir(prefix_dir)
    write_tsv(rows_path, rows)
    status = "ok" if all(row["status"] == "ok" for row in rows) else "dry_run" if args.dry_run else "partial"
    state_manifest = {
        "task": asdict(task),
        "task_signature": signature,
        "status": status,
        "rows": len(rows),
        "oracle_actions": str(rows_path),
    }
    (state_dir / "manifest.json").write_text(json.dumps(state_manifest, indent=2, sort_keys=True) + "\n")
    summary = {"state_id": task.state_id, "status": status, "rows": len(rows)}
    print(json.dumps({"benchmark_id": task.benchmark_id, **summary}, sort_keys=True), flush=True)
    return summary


def evaluate_benchmark(args: argparse.Namespace, benchmark_id: str, tasks: list[PrefixTask]) -> dict[str, Any]:
    started = time.time()
    bench_path = find_bench_path(benchmark_id)
    candidate_count, candidate_ids = candidate_ids_by_net(bench_path)
    state_summaries = []
    try:
        for task in tasks:
            state_summaries.append(evaluate_state(args, task, bench_path, candidate_count, candidate_ids))
    except Exception as exc:
        summary = {
            "benchmark_id": benchmark_id,
            "status": "error",
            "error": str(exc),
            "state_summaries": state_summaries,
        }
    else:
        bad = [row for row in state_summaries if row["status"] not in {"ok", "skipped", "dry_run"}]
        summary = {
            "benchmark_id": benchmark_id,
            "status": "partial" if bad else "dry_run" if args.dry_run else "ok",
            "states": len(state_summaries),
            "rows": sum(int(row["rows"]) for row in state_summaries),
            "elapsed_sec": round(time.time() - started, 3),
            "state_summaries": state_summaries,
        }
    log = args.out_dir / "logs" / f"{_safe_name(benchmark_id)}.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def merge_state_rows(out_dir: Path) -> Path:
    rows = []
    for path in sorted(out_dir.glob("states/*/*/oracle_actions.tsv")):
        rows.extend(read_tsv(path))
    merged = out_dir / "oracle_actions.tsv"
    write_tsv(merged, rows)
    return merged


def write_task_manifest(args: argparse.Namespace, tasks: list[PrefixTask]) -> None:
    tasks_path = args.out_dir / "tasks.jsonl"
    with tasks_path.open("w") as handle:
        for task in tasks:
            payload = asdict(task)
            payload["task_signature"] = task_signature(task)
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    skipped_states = list(getattr(args, "skipped_states", []))
    skipped_path = args.out_dir / "skipped_states.json"
    skipped_path.write_text(json.dumps(skipped_states, indent=2, sort_keys=True) + "\n")
    manifest = {
        "plans_dir": str(args.plans_dir),
        "training_manifest": str(args.training_manifest),
        "eval_protocol": str(args.eval_protocol),
        "benchmarks": sorted({task.benchmark_id for task in tasks}),
        "benchmark_count": len({task.benchmark_id for task in tasks}),
        "state_count": len(tasks),
        "candidate_evaluations": sum(len(task.candidates) for task in tasks),
        "prefix_evaluations": len(tasks),
        "candidate_strategy": args.candidate_strategy,
        "candidate_pool_size": args.candidate_pool_size,
        "actions_per_prefix": args.actions_per_prefix,
        "prefix_steps": parse_int_list(args.prefix_steps),
        "skipped_state_count": len(skipped_states),
        "skipped_states_json": str(skipped_path),
        "patterns": args.patterns,
        "seed": args.seed,
        "backend": args.backend,
        "tasks_jsonl": str(tasks_path),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans-dir", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--eval-protocol", type=Path, default=Path("configs/eval_protocol_coverage_only.json"))
    parser.add_argument("--prefix-steps", default="4,12,20,28")
    parser.add_argument("--candidate-strategy", default="hard_fault_cluster")
    parser.add_argument("--candidate-pool-size", type=int, default=48)
    parser.add_argument("--actions-per-prefix", type=int, default=8)
    parser.add_argument("--candidate-sample-seed", type=int, default=0)
    parser.add_argument("--candidate-cache-dir", default=None)
    parser.add_argument("--real-fault-priors", default=None)
    parser.add_argument("--activation-priors", default=None)
    parser.add_argument("--patterns", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--backend", choices=["tmax", "atalanta-bist"], default="atalanta-bist")
    parser.add_argument("--timeout-sec", type=int, default=14400)
    parser.add_argument("--parallel-jobs", type=int, default=12)
    parser.add_argument("--max-benchmarks", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tmax-bin", default="/data3/pengqingsong/synopsys/txs/O-2018.06-SP1/bin/tmax")
    parser.add_argument(
        "--atalanta-bin",
        default=str(Path(os.environ.get("DFT_ROOT", "/data4/pengqingsong/DFT")) / "tool/atalanta_bist_with_ufaults/atalanta"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--cleanup-workdir", action="store_true")
    parser.add_argument("--allow-onpolicy-outside-pool", action="store_true")
    args = parser.parse_args()

    if args.candidate_pool_size < args.actions_per_prefix:
        parser.error("--candidate-pool-size must be >= --actions-per-prefix")
    if args.actions_per_prefix < 2:
        parser.error("--actions-per-prefix must be at least 2")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(args)
    write_task_manifest(args, tasks)
    print(
        json.dumps(
            {
                "status": "prepared",
                "benchmarks": len({task.benchmark_id for task in tasks}),
                "states": len(tasks),
                "skipped_states": len(getattr(args, "skipped_states", [])),
                "prefix_evaluations": len(tasks),
                "candidate_evaluations": sum(len(task.candidates) for task in tasks),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if args.prepare_only:
        return

    by_benchmark: dict[str, list[PrefixTask]] = {}
    for task in tasks:
        by_benchmark.setdefault(task.benchmark_id, []).append(task)
    summaries = []
    with ThreadPoolExecutor(max_workers=max(1, args.parallel_jobs)) as executor:
        futures = {
            executor.submit(evaluate_benchmark, args, benchmark_id, benchmark_tasks): benchmark_id
            for benchmark_id, benchmark_tasks in sorted(by_benchmark.items())
        }
        for future in as_completed(futures):
            benchmark_id = futures[future]
            try:
                summary = future.result()
            except Exception as exc:
                summary = {"benchmark_id": benchmark_id, "status": "error", "error": str(exc)}
            summaries.append(summary)
            print(json.dumps(summary, sort_keys=True), flush=True)
            (args.out_dir / "collection_status.json").write_text(
                json.dumps(summaries, indent=2, sort_keys=True) + "\n"
            )

    merged = merge_state_rows(args.out_dir)
    manifest_path = args.out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["summaries"] = summaries
    manifest["oracle_actions"] = str(merged)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if any(summary.get("status") in {"error", "partial"} for summary in summaries):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
