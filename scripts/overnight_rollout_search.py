"""Run unattended rollout-planner sweeps and record TMAX outcomes."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import time


DEFAULT_BENCHMARKS = ["iscas89__s838", "iscas89__s838a"]
RESULT_FIELDS = [
    "timestamp",
    "status",
    "benchmark_id",
    "planner",
    "budget",
    "max_candidates",
    "beam_width",
    "lookahead_depth",
    "score_field",
    "beam_objective",
    "discount_gamma",
    "patterns",
    "seed",
    "plan_score_sum",
    "plan_fc_sum",
    "delta_test_coverage",
    "delta_fault_coverage",
    "delta_pattern_count",
    "plan_csv",
    "eval_dir",
    "elapsed_sec",
    "error",
]


def parse_csv_values(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_int_values(text: str) -> list[int]:
    return [int(item) for item in parse_csv_values(text)]


def sanitize(value: str) -> str:
    return value.replace("/", "_").replace(":", "_")


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


def plan_sums(path: Path) -> tuple[float, float]:
    rows = list(csv.DictReader(path.open()))
    return (
        sum(float(row.get("score_pred") or 0.0) for row in rows),
        sum(float(row.get("fc_pred") or 0.0) for row in rows),
    )


def final_eval_metrics(labels_csv: Path) -> dict[str, str]:
    rows = list(csv.DictReader(labels_csv.open()))
    if not rows:
        return {}
    return rows[-1]


def append_result(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})


def update_best(out_dir: Path, row: dict) -> None:
    best_path = out_dir / "best.json"
    current = None
    if best_path.exists():
        current = json.loads(best_path.read_text())
    value = row.get("delta_test_coverage")
    if value in ("", None, "NA"):
        return
    value_float = float(value)
    if current is not None and value_float <= float(current.get("delta_test_coverage", "-inf")):
        return
    best_path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    plan_csv = Path(row["plan_csv"])
    if plan_csv.exists():
        shutil.copy2(plan_csv, out_dir / "best_plan.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Overnight rollout planner search.")
    parser.add_argument("--checkpoint", default="runs/tmax50k_curriculum_full5/best.pt")
    parser.add_argument("--benchmarks", default=",".join(DEFAULT_BENCHMARKS))
    parser.add_argument("--planners", default="beam")
    parser.add_argument("--budgets", default="5,8,10")
    parser.add_argument("--max-candidates", default="16,32")
    parser.add_argument("--beam-widths", default="2,4,8")
    parser.add_argument("--lookahead-depths", default="1,2,3,4")
    parser.add_argument("--score-fields", default="reward_pred")
    parser.add_argument("--beam-objectives", default="cumulative")
    parser.add_argument("--discount-gammas", default="0.9")
    parser.add_argument("--patterns", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--timeout-sec", type=int, default=600)
    parser.add_argument("--time-limit-hours", type=float, default=8.0)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cleanup-workdir", action="store_true", default=True)
    args = parser.parse_args()

    started = time.time()
    stamp = datetime.now().strftime("%y%m%d-%H%M%S")
    out_dir = Path(args.out_dir or f"autoresearch/overnight-{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.tsv"
    config = vars(args) | {"out_dir": str(out_dir), "started": stamp}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    benchmarks = parse_csv_values(args.benchmarks)
    planners = parse_csv_values(args.planners)
    budgets = parse_int_values(args.budgets)
    max_candidates_values = parse_int_values(args.max_candidates)
    beam_widths = parse_int_values(args.beam_widths)
    lookahead_depths = parse_int_values(args.lookahead_depths)
    score_fields = parse_csv_values(args.score_fields)
    beam_objectives = parse_csv_values(args.beam_objectives)
    discount_gammas = [float(value) for value in parse_csv_values(args.discount_gammas)]
    deadline = started + args.time_limit_hours * 3600.0

    for benchmark in benchmarks:
        for planner in planners:
            depth_values = [0] if planner == "beam_full" else lookahead_depths
            for budget in budgets:
                for max_candidates in max_candidates_values:
                    for beam_width in beam_widths:
                        for lookahead_depth in depth_values:
                            for score_field in score_fields:
                                for beam_objective in beam_objectives:
                                    gamma_values = discount_gammas if beam_objective == "discounted" else [1.0]
                                    for discount_gamma in gamma_values:
                                        if time.time() >= deadline:
                                            return
                                        iter_started = time.time()
                                        variant = (
                                            f"{sanitize(benchmark)}__{planner}__b{budget}__c{max_candidates}"
                                            f"__bw{beam_width}__d{lookahead_depth}__{score_field}"
                                            f"__{beam_objective}__g{sanitize(str(discount_gamma))}"
                                        )
                                        plan_csv = out_dir / "plans" / f"{variant}.csv"
                                        eval_dir = out_dir / "evals" / variant
                                        plan_log = out_dir / "logs" / f"{variant}.plan.log"
                                        eval_log = out_dir / "logs" / f"{variant}.eval.log"
                                        row = {
                                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                                            "benchmark_id": benchmark,
                                            "planner": planner,
                                            "budget": budget,
                                            "max_candidates": max_candidates,
                                            "beam_width": beam_width,
                                            "lookahead_depth": lookahead_depth,
                                            "score_field": score_field,
                                            "beam_objective": beam_objective,
                                            "discount_gamma": discount_gamma,
                                            "patterns": args.patterns,
                                            "seed": args.seed,
                                            "plan_csv": str(plan_csv),
                                            "eval_dir": str(eval_dir),
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
                                            str(budget),
                                            "--max-candidates",
                                            str(max_candidates),
                                            "--device",
                                            "cpu",
                                            "--planner",
                                            planner,
                                            "--beam-width",
                                            str(beam_width),
                                            "--lookahead-depth",
                                            str(max(1, lookahead_depth)),
                                            "--score-field",
                                            score_field,
                                            "--beam-objective",
                                            beam_objective,
                                            "--discount-gamma",
                                            str(discount_gamma),
                                            "--out",
                                            str(plan_csv),
                                        ]
                                        ok, error = run_command(plan_cmd, plan_log)
                                        if not ok:
                                            row.update({"status": "plan_error", "error": error})
                                            row["elapsed_sec"] = round(time.time() - iter_started, 3)
                                            append_result(results_path, row)
                                            continue

                                        score_sum, fc_sum = plan_sums(plan_csv)
                                        row.update({"plan_score_sum": score_sum, "plan_fc_sum": fc_sum})
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
                                            "--timeout-sec",
                                            str(args.timeout_sec),
                                            "--force",
                                        ]
                                        if args.dry_run:
                                            eval_cmd.append("--dry-run")
                                        if args.cleanup_workdir:
                                            eval_cmd.append("--cleanup-workdir")
                                        ok, error = run_command(eval_cmd, eval_log)
                                        if not ok:
                                            row.update({"status": "eval_error", "error": error})
                                            row["elapsed_sec"] = round(time.time() - iter_started, 3)
                                            append_result(results_path, row)
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
                                        append_result(results_path, row)
                                        update_best(out_dir, row)


if __name__ == "__main__":
    main()
