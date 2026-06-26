"""Run multi-benchmark rollout sweeps and rank variants by macro mean TC gain."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import itertools
import json
from math import isnan
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tpi_jepa.bench import parse_bench
from tpi_jepa.labels import find_bench_path


DEFAULT_BENCHMARKS = [
    "iscas89__s838",
    "iscas89__s838a",
    "epfl__random_control__priority__priority",
    "epfl__random_control__router__router",
]
ROUTER_BENCHMARK = "epfl__random_control__router__router"

RESULT_FIELDS = [
    "timestamp",
    "variant_id",
    "status",
    "benchmark_id",
    "logic_gates",
    "budget_mode",
    "budget",
    "planner",
    "beam_objective",
    "score_field",
    "beam_width",
    "lookahead_depth",
    "max_candidates",
    "discount_gamma",
    "candidate_strategy",
    "candidate_diversity_penalty",
    "candidate_diversity_depth",
    "candidate_sample_seed",
    "patterns",
    "seed",
    "plan_score_sum",
    "plan_reward_sum",
    "plan_fc_sum",
    "plan_return_sum",
    "plan_sequence_sum",
    "plan_objective_sum",
    "delta_test_coverage",
    "delta_fault_coverage",
    "delta_pattern_count",
    "plan_csv",
    "eval_dir",
    "elapsed_sec",
    "error",
]

GROUPED_FIELDS = [
    "timestamp",
    "variant_id",
    "status",
    "planner",
    "beam_objective",
    "score_field",
    "beam_width",
    "lookahead_depth",
    "max_candidates",
    "discount_gamma",
    "candidate_strategy",
    "candidate_diversity_penalty",
    "candidate_diversity_depth",
    "candidate_sample_seed",
    "patterns",
    "seed",
    "benchmark_count",
    "macro_mean_delta_tc",
    "min_delta_tc",
    "router_delta_tc",
    "positive_count",
    "negative_count",
    "safe",
    "elapsed_sec",
]


def parse_csv_values(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_int_values(text: str) -> list[int]:
    return [int(item) for item in parse_csv_values(text)]


def parse_float_values(text: str) -> list[float]:
    return [float(item) for item in parse_csv_values(text)]


def sanitize(value: str) -> str:
    return value.replace("/", "_").replace(":", "_").replace(".", "p")


def logic_gate_count(benchmark_id: str) -> int:
    circuit = parse_bench(find_bench_path(benchmark_id))
    inputs = set(circuit.inputs)
    return sum(1 for node, gate_type in circuit.gate_types.items() if node not in inputs and gate_type != "WIRE")


def budget_for(logic_gates: int, mode: str, fixed_budget: int | None) -> int:
    if mode == "fixed":
        if fixed_budget is None:
            raise ValueError("--fixed-budget is required when --budget-mode=fixed")
        return max(1, int(fixed_budget))
    if mode == "floor1pct":
        return max(1, logic_gates // 100)
    if mode == "ceil1pct":
        return max(1, -(-logic_gates // 100))
    if mode == "round1pct":
        return max(1, round(logic_gates * 0.01))
    raise ValueError(f"unsupported budget mode: {mode}")


def run_command(cmd: list[str], log_path: Path) -> tuple[bool, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)
    if result.returncode == 0:
        return True, ""
    try:
        tail = "".join(log_path.read_text(errors="replace").splitlines(True)[-40:])
    except OSError:
        tail = ""
    return False, tail.strip()


def _float_cell(row: dict[str, str], field: str) -> float:
    value = row.get(field)
    if value in (None, ""):
        return 0.0
    return float(value)


def plan_sums(path: Path) -> dict[str, float]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        "plan_score_sum": sum(_float_cell(row, "score_pred") for row in rows),
        "plan_reward_sum": sum(_float_cell(row, "reward_pred") for row in rows),
        "plan_fc_sum": sum(_float_cell(row, "fc_pred") for row in rows),
        "plan_return_sum": sum(_float_cell(row, "return_pred") for row in rows),
        "plan_sequence_sum": sum(_float_cell(row, "sequence_score") for row in rows),
        "plan_objective_sum": sum(_float_cell(row, "objective_score") for row in rows),
    }


def final_eval_metrics(labels_csv: Path) -> dict[str, str]:
    with labels_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    return rows[-1]


def append_tsv(path: Path, row: dict[str, Any], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def apply_eval_protocol(args: argparse.Namespace) -> dict[str, Any] | None:
    """Load and apply a fixed evaluation protocol."""

    if not args.eval_protocol:
        return None
    protocol_path = Path(args.eval_protocol)
    protocol = json.loads(protocol_path.read_text())
    if "benchmarks" in protocol:
        benchmarks = protocol["benchmarks"]
        args.benchmarks = ",".join(benchmarks) if isinstance(benchmarks, list) else str(benchmarks)
    for key in [
        "budget_mode",
        "fixed_budget",
        "patterns",
        "seed",
        "timeout_sec",
        "safety_benchmark",
        "safety_min_delta",
    ]:
        if key in protocol:
            setattr(args, key, protocol[key])
    return {"path": str(protocol_path), **protocol}


def numeric(value: Any, default: float = float("nan")) -> float:
    if value in (None, "", "NA"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def aggregate_variant(
    *,
    variant_id: str,
    variant: dict[str, Any],
    rows: list[dict[str, Any]],
    required_benchmarks: int,
    router_benchmark: str,
    safety_min_delta: float,
    started: float,
) -> dict[str, Any]:
    completed_rows = [row for row in rows if row.get("status") in {"ok", "dry_run"}]
    metric_rows = [
        row
        for row in completed_rows
        if row.get("delta_test_coverage") not in ("", "NA", None)
        and not isnan(numeric(row.get("delta_test_coverage")))
    ]
    deltas = [numeric(row.get("delta_test_coverage")) for row in metric_rows]
    router_rows = [row for row in metric_rows if row.get("benchmark_id") == router_benchmark]
    router_delta = numeric(router_rows[-1].get("delta_test_coverage")) if router_rows else float("nan")
    complete = len(completed_rows) == required_benchmarks
    safe = bool(complete and router_rows and router_delta >= safety_min_delta)
    status = "ok" if complete and len(metric_rows) == required_benchmarks else "dry_run" if complete else "incomplete"
    if rows and any(str(row.get("status", "")).endswith("error") for row in rows):
        status = "error"
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "variant_id": variant_id,
        "status": status,
        **variant,
        "benchmark_count": len(completed_rows),
        "macro_mean_delta_tc": sum(deltas) / len(deltas) if deltas else "NA",
        "min_delta_tc": min(deltas) if deltas else "NA",
        "router_delta_tc": router_delta if router_rows else "NA",
        "positive_count": sum(1 for value in deltas if value > 0.0),
        "negative_count": sum(1 for value in deltas if value < 0.0),
        "safe": safe,
        "elapsed_sec": round(time.time() - started, 3),
    }


def best_group(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    safe_rows = [row for row in rows if row.get("safe") is True or row.get("safe") == "True"]
    candidates = safe_rows or rows

    def key(row: dict[str, Any]) -> tuple[float, float, float, float]:
        return (
            numeric(row.get("macro_mean_delta_tc"), -1e9),
            numeric(row.get("min_delta_tc"), -1e9),
            numeric(row.get("router_delta_tc"), -1e9),
            -numeric(row.get("negative_count"), 1e9),
        )

    return max(candidates, key=key)


def copy_best_artifacts(out_dir: Path, best: dict[str, Any], result_rows: list[dict[str, Any]]) -> None:
    write_json(out_dir / "best.json", best)
    variant_rows = [row for row in result_rows if row.get("variant_id") == best.get("variant_id")]
    manifest = {
        "variant": best,
        "benchmarks": [
            {
                "benchmark_id": row.get("benchmark_id"),
                "budget": row.get("budget"),
                "logic_gates": row.get("logic_gates"),
                "delta_test_coverage": row.get("delta_test_coverage"),
                "plan_csv": row.get("plan_csv"),
                "eval_dir": row.get("eval_dir"),
            }
            for row in variant_rows
        ],
    }
    write_json(out_dir / "best_plan_manifest.json", manifest)
    best_plans = out_dir / "best_plans"
    best_plans.mkdir(parents=True, exist_ok=True)
    for row in variant_rows:
        plan_csv = Path(str(row.get("plan_csv", "")))
        if plan_csv.exists():
            shutil.copy2(plan_csv, best_plans / f"{sanitize(str(row.get('benchmark_id')))}.csv")


def variant_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    variants = []
    for planner in parse_csv_values(args.planners):
        depth_values = [0] if planner == "beam_full" else parse_int_values(args.lookahead_depths)
        for (
            beam_objective,
            score_field,
            beam_width,
            depth,
            max_candidates,
            candidate_strategy,
            diversity_penalty,
            diversity_depth,
            candidate_sample_seed,
        ) in itertools.product(
            parse_csv_values(args.beam_objectives),
            parse_csv_values(args.score_fields),
            parse_int_values(args.beam_widths),
            depth_values,
            parse_int_values(args.max_candidates),
            parse_csv_values(args.candidate_strategies),
            parse_float_values(args.candidate_diversity_penalties),
            parse_int_values(args.candidate_diversity_depths),
            parse_int_values(args.candidate_sample_seeds),
        ):
            gammas = parse_float_values(args.discount_gammas) if beam_objective == "discounted" else [1.0]
            for discount_gamma in gammas:
                variant = {
                    "planner": planner,
                    "beam_objective": beam_objective,
                    "score_field": score_field,
                    "beam_width": beam_width,
                    "lookahead_depth": depth,
                    "max_candidates": max_candidates,
                    "discount_gamma": discount_gamma,
                    "candidate_strategy": candidate_strategy,
                    "candidate_diversity_penalty": diversity_penalty,
                    "candidate_diversity_depth": diversity_depth,
                    "candidate_sample_seed": candidate_sample_seed,
                    "patterns": args.patterns,
                    "seed": args.seed,
                }
                variant_id = (
                    f"{planner}__{beam_objective}__{score_field}__bw{beam_width}"
                    f"__d{depth}__c{max_candidates}__g{sanitize(str(discount_gamma))}"
                    f"__cand{sanitize(candidate_strategy)}__div{sanitize(str(diversity_penalty))}"
                    f"__s{candidate_sample_seed}"
                )
                variants.append({"variant_id": variant_id, **variant})
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(description="Generalization-mean TPI-JEPA rollout sweep.")
    parser.add_argument("--eval-protocol", default=None, help="JSON file defining fixed benchmarks and evaluation settings.")
    parser.add_argument("--checkpoint", default="autoresearch/framework-260610-014420/best.pt")
    parser.add_argument("--benchmarks", default=",".join(DEFAULT_BENCHMARKS))
    parser.add_argument("--budget-mode", choices=["floor1pct", "ceil1pct", "round1pct", "fixed"], default="floor1pct")
    parser.add_argument("--fixed-budget", type=int, default=None)
    parser.add_argument("--planners", default="beam,beam_full")
    parser.add_argument("--score-fields", default="reward_pred")
    parser.add_argument("--beam-objectives", default="cumulative,terminal,discounted")
    parser.add_argument("--beam-widths", default="4,8")
    parser.add_argument("--lookahead-depths", default="3,5")
    parser.add_argument("--max-candidates", default="32,64")
    parser.add_argument("--discount-gammas", default="0.9")
    parser.add_argument("--candidate-strategies", default="checkpoint")
    parser.add_argument("--candidate-diversity-penalties", default="0.0")
    parser.add_argument("--candidate-diversity-depths", default="4")
    parser.add_argument("--candidate-cache-dir", default=None)
    parser.add_argument("--candidate-sample-seeds", default="0")
    parser.add_argument("--real-fault-priors", default=None)
    parser.add_argument("--activation-priors", default=None)
    parser.add_argument("--plan-device", default="cpu")
    parser.add_argument("--eval-backend", choices=["tmax", "atalanta-bist"], default="tmax")
    parser.add_argument(
        "--atalanta-bin",
        default="/data3/pengqingsong/DFT/DeepTPI-project/external/DeepTPI/src/external/Atalanta_BIST/atalanta",
    )
    parser.add_argument("--patterns", type=int, default=300000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--timeout-sec", type=int, default=14400)
    parser.add_argument("--time-limit-hours", type=float, default=12.0)
    parser.add_argument("--safety-benchmark", default=ROUTER_BENCHMARK)
    parser.add_argument("--safety-min-delta", type=float, default=-0.005)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cleanup-workdir", action="store_true", default=True)
    args = parser.parse_args()
    protocol = apply_eval_protocol(args)

    started = time.time()
    stamp = datetime.now().strftime("%y%m%d-%H%M%S")
    out_dir = Path(args.out_dir or f"autoresearch/gmean-300k-{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.tsv"
    grouped_path = out_dir / "grouped_results.tsv"
    config = vars(args) | {"out_dir": str(out_dir), "started": stamp, "eval_protocol_loaded": protocol}
    write_json(out_dir / "config.json", config)
    if protocol is not None:
        write_json(out_dir / "eval_protocol.json", protocol)

    benchmarks = parse_csv_values(args.benchmarks)
    budgets = {}
    for benchmark in benchmarks:
        gates = logic_gate_count(benchmark)
        budgets[benchmark] = {
            "logic_gates": gates,
            "budget": budget_for(gates, args.budget_mode, args.fixed_budget),
        }
    write_json(out_dir / "budgets.json", budgets)

    deadline = started + args.time_limit_hours * 3600.0
    all_result_rows: list[dict[str, Any]] = []
    grouped_rows: list[dict[str, Any]] = []

    for variant in variant_grid(args):
        if time.time() >= deadline:
            break
        variant_started = time.time()
        variant_id = variant["variant_id"]
        variant_rows: list[dict[str, Any]] = []
        core_variant = {field: variant[field] for field in GROUPED_FIELDS if field in variant}
        for benchmark in benchmarks:
            if time.time() >= deadline:
                break
            iter_started = time.time()
            budget_info = budgets[benchmark]
            run_id = f"{sanitize(benchmark)}__{variant_id}"
            plan_csv = out_dir / "plans" / f"{run_id}.csv"
            eval_dir = out_dir / "evals" / run_id
            plan_log = out_dir / "logs" / f"{run_id}.plan.log"
            eval_log = out_dir / "logs" / f"{run_id}.eval.log"
            row: dict[str, Any] = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "variant_id": variant_id,
                "benchmark_id": benchmark,
                "logic_gates": budget_info["logic_gates"],
                "budget_mode": args.budget_mode,
                "budget": budget_info["budget"],
                "plan_csv": str(plan_csv),
                "eval_dir": str(eval_dir),
                **core_variant,
            }
            plan_cmd = [
                "python",
                "-m",
                "tpi_jepa.plan",
                "--checkpoint",
                args.checkpoint,
                "--benchmark-id",
                benchmark,
                "--budget",
                str(budget_info["budget"]),
                "--max-candidates",
                str(variant["max_candidates"]),
                "--device",
                args.plan_device,
                "--planner",
                variant["planner"],
                "--beam-width",
                str(variant["beam_width"]),
                "--lookahead-depth",
                str(max(1, int(variant["lookahead_depth"]))),
                "--score-field",
                variant["score_field"],
                "--beam-objective",
                variant["beam_objective"],
                "--discount-gamma",
                str(variant["discount_gamma"]),
                "--out",
                str(plan_csv),
            ]
            if variant["candidate_strategy"] != "checkpoint":
                plan_cmd.extend(["--candidate-strategy", variant["candidate_strategy"]])
            if args.candidate_cache_dir:
                plan_cmd.extend(["--candidate-cache-dir", args.candidate_cache_dir])
            plan_cmd.extend(["--candidate-sample-seed", str(variant["candidate_sample_seed"])])
            if args.real_fault_priors:
                plan_cmd.extend(["--real-fault-priors", args.real_fault_priors])
            if args.activation_priors:
                plan_cmd.extend(["--activation-priors", args.activation_priors])
            plan_cmd.extend(
                [
                    "--candidate-diversity-penalty",
                    str(variant["candidate_diversity_penalty"]),
                    "--candidate-diversity-depth",
                    str(variant["candidate_diversity_depth"]),
                ]
            )
            ok, error = run_command(plan_cmd, plan_log)
            if not ok:
                row.update({"status": "plan_error", "error": error, "elapsed_sec": round(time.time() - iter_started, 3)})
                append_tsv(results_path, row, RESULT_FIELDS)
                variant_rows.append(row)
                all_result_rows.append(row)
                continue

            row.update(plan_sums(plan_csv))
            eval_cmd = [
                "python",
                "-m",
                "tpi_jepa.evaluate_plan_tmax",
                "--benchmark-id",
                benchmark,
                "--plan-csv",
                str(plan_csv),
                "--out-dir",
                str(eval_dir),
                "--patterns",
                str(args.patterns),
                "--seed",
                str(args.seed),
                "--backend",
                args.eval_backend,
                "--timeout-sec",
                str(args.timeout_sec),
                "--force",
            ]
            if args.eval_backend == "atalanta-bist":
                eval_cmd.extend(["--atalanta-bin", args.atalanta_bin])
            if args.dry_run:
                eval_cmd.append("--dry-run")
            if args.cleanup_workdir:
                eval_cmd.append("--cleanup-workdir")
            ok, error = run_command(eval_cmd, eval_log)
            if not ok:
                row.update({"status": "eval_error", "error": error, "elapsed_sec": round(time.time() - iter_started, 3)})
                append_tsv(results_path, row, RESULT_FIELDS)
                variant_rows.append(row)
                all_result_rows.append(row)
                continue

            metrics = final_eval_metrics(eval_dir / "labels.csv")
            row.update(
                {
                    "status": metrics.get("status") or "ok",
                    "delta_test_coverage": metrics.get("delta_test_coverage") or "NA",
                    "delta_fault_coverage": metrics.get("delta_fault_coverage") or "NA",
                    "delta_pattern_count": metrics.get("delta_pattern_count") or "NA",
                    "elapsed_sec": round(time.time() - iter_started, 3),
                }
            )
            append_tsv(results_path, row, RESULT_FIELDS)
            variant_rows.append(row)
            all_result_rows.append(row)

        group = aggregate_variant(
            variant_id=variant_id,
            variant=core_variant,
            rows=variant_rows,
            required_benchmarks=len(benchmarks),
            router_benchmark=args.safety_benchmark,
            safety_min_delta=args.safety_min_delta,
            started=variant_started,
        )
        append_tsv(grouped_path, group, GROUPED_FIELDS)
        grouped_rows.append(group)
        best = best_group(grouped_rows)
        if best is not None:
            copy_best_artifacts(out_dir, best, all_result_rows)


if __name__ == "__main__":
    main()
