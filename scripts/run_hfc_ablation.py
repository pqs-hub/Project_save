"""Run hard-fault-cone candidate-source ablations.

The goal is to separate candidate-generator value from world-model scorer value:

* hfc_model: hard-fault-cone candidates scored by the TPI-JEPA model.
* hfc_heuristic: hard-fault-cone candidates selected by the heuristic rank only.
* hfc_random: random selection inside the same top-K hard-fault-cone pool.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tpi_jepa.bench import parse_bench
from tpi_jepa.graph import build_graph
from tpi_jepa.labels import find_bench_path
from tpi_jepa.plan import (
    _ranked_candidates,
    enumerate_candidates,
    load_checkpoint,
    set_real_fault_context,
    write_plan_csv,
)


RESULT_FIELDS = [
    "method",
    "selector",
    "benchmark_id",
    "status",
    "budget",
    "max_candidates",
    "patterns",
    "seed",
    "baseline_test_coverage",
    "final_test_coverage",
    "delta_test_coverage",
    "baseline_fault_coverage",
    "final_fault_coverage",
    "delta_fault_coverage",
    "plan_csv",
    "eval_dir",
    "elapsed_sec",
    "error",
]


def append_tsv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})


def load_completed_results(path: Path) -> set[tuple[str, str]]:
    if not path.is_file():
        return set()
    completed: set[tuple[str, str]] = set()
    with path.open(newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("status") == "ok":
                completed.add((row.get("method", ""), row.get("benchmark_id", "")))
    return completed


def read_final_metrics(labels_csv: Path) -> dict[str, Any]:
    with labels_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"status": "missing"}
    final = rows[-1]
    return {
        "status": final.get("status") or "ok",
        "baseline_test_coverage": final.get("baseline_test_coverage"),
        "final_test_coverage": final.get("test_coverage"),
        "delta_test_coverage": final.get("delta_test_coverage"),
        "baseline_fault_coverage": final.get("baseline_fault_coverage"),
        "final_fault_coverage": final.get("fault_coverage"),
        "delta_fault_coverage": final.get("delta_fault_coverage"),
    }


def run_command(cmd: list[str], log_path: Path) -> tuple[bool, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.write_text(completed.stdout)
    if completed.returncode != 0:
        return False, completed.stdout[-4000:]
    return True, ""


def heuristic_plan(
    *,
    benchmark_id: str,
    budget: int,
    max_candidates: int,
    real_fault_priors: str | None,
    activation_priors: str | None,
    selector: str,
    seed: int,
) -> list[dict[str, Any]]:
    graph = build_graph(parse_bench(find_bench_path(benchmark_id)))
    set_real_fault_context(benchmark_id, real_fault_priors, activation_priors)
    rng = random.Random(seed)
    selected: list[tuple[str, str]] = []
    rows: list[dict[str, Any]] = []
    for step in range(1, budget + 1):
        pool = enumerate_candidates(
            graph,
            selected,
            max_candidates=max_candidates,
            strategy="hard_fault_cone",
            real_fault_benchmark_id=benchmark_id,
            real_fault_prior_path=real_fault_priors,
            activation_prior_path=activation_priors,
        )
        if not pool:
            break
        ranked = _ranked_candidates(
            graph,
            "hard_fault_cone",
            real_fault_benchmark_id=benchmark_id,
            real_fault_prior_path=real_fault_priors,
            activation_prior_path=activation_priors,
        )
        score_by_candidate = {(node, action): score for node, action, score in ranked}
        if selector == "hfc_random":
            candidate = rng.choice(pool)
        else:
            candidate = max(pool, key=lambda item: score_by_candidate.get(item, float("-inf")))
        selected.append(candidate)
        score = float(score_by_candidate.get(candidate, 0.0))
        rows.append(
            {
                "step": step,
                "node": candidate[0],
                "type": candidate[1],
                "score_pred": score,
                "reward_pred": score,
                "fc_pred": score,
                "step_value": score,
                "sequence_score": sum(float(row["step_value"]) for row in rows) + score,
                "objective_score": score,
                "objective": "heuristic" if selector == "hfc_heuristic" else "random",
                "planner": selector,
                "candidate_strategy": "hard_fault_cone",
            }
        )
    return rows


def plan_with_model(args: argparse.Namespace, benchmark: str, plan_csv: Path, log_path: Path) -> tuple[bool, str]:
    if plan_csv.is_file() and not args.force:
        return True, ""
    cmd = [
        sys.executable,
        "-m",
        "tpi_jepa.plan",
        "--checkpoint",
        str(args.checkpoint),
        "--benchmark-id",
        benchmark,
        "--budget",
        str(args.budget),
        "--max-candidates",
        str(args.max_candidates),
        "--planner",
        args.model_planner,
        "--beam-width",
        str(args.beam_width),
        "--lookahead-depth",
        str(args.lookahead_depth),
        "--score-field",
        args.score_field,
        "--beam-objective",
        args.beam_objective,
        "--discount-gamma",
        str(args.discount_gamma),
        "--candidate-strategy",
        "hard_fault_cone",
        "--candidate-diversity-penalty",
        "0.0",
        "--candidate-diversity-depth",
        "4",
        "--out",
        str(plan_csv),
        "--device",
        args.device,
    ]
    if args.real_fault_priors:
        cmd.extend(["--real-fault-priors", str(args.real_fault_priors)])
    if args.activation_priors:
        cmd.extend(["--activation-priors", str(args.activation_priors)])
    return run_command(cmd, log_path)


def evaluate(args: argparse.Namespace, benchmark: str, plan_csv: Path, eval_dir: Path, log_path: Path) -> tuple[bool, str]:
    labels_csv = eval_dir / "labels.csv"
    if labels_csv.is_file() and not args.force:
        return True, ""
    cmd = [
        sys.executable,
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
        cmd.extend(["--atalanta-bin", args.atalanta_bin])
    if args.cleanup_workdir:
        cmd.append("--cleanup-workdir")
    return run_command(cmd, log_path)


def summarize(results_path: Path, out_md: Path, eval_protocol: Path) -> None:
    with results_path.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    protocol = json.loads(eval_protocol.read_text())
    aliases = {row["benchmark_id"]: (row["id"], row["circuit"]) for row in protocol.get("table_rows", [])}
    order = {bench: idx for idx, bench in enumerate(protocol.get("benchmarks", []))}
    methods = []
    for row in rows:
        if row["method"] not in methods:
            methods.append(row["method"])

    def pct(value: str | float | None) -> str:
        if value in (None, "", "NA"):
            return "NA"
        return f"{100.0 * float(value):.2f}%"

    lines = ["# Hard-Fault-Cone Ablation Results", ""]
    for method in methods:
        items = [row for row in rows if row["method"] == method]
        items.sort(key=lambda row: order.get(row["benchmark_id"], 999))
        deltas = [float(row["delta_test_coverage"]) for row in items if row.get("delta_test_coverage") not in ("", "NA")]
        neg = sum(1 for value in deltas if value < 0)
        lines.extend(
            [
                f"## {method}",
                "",
                "| ID | Circuit | Benchmark | #TPs | Baseline TC | After TPI TC | Imp. |",
                "|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in items:
            bid = row["benchmark_id"]
            ident, circuit = aliases.get(bid, ("", bid))
            lines.append(
                f"| {ident} | {circuit} | `{bid}` | {row['budget']} | "
                f"{pct(row['baseline_test_coverage'])} | {pct(row['final_test_coverage'])} | "
                f"{pct(row['delta_test_coverage'])} |"
            )
        avg = sum(deltas) / len(deltas) if deltas else 0.0
        lines.append(f"| Avg. |  |  |  |  |  | {pct(avg)} |")
        lines.append("")
        lines.append(f"`positive_count={sum(1 for value in deltas if value > 0)}`, `negative_count={neg}`, `min_delta={pct(min(deltas) if deltas else 0.0)}`")
        lines.append("")
    out_md.write_text("\n".join(lines) + "\n")


def protocol_benchmarks(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    return list(data.get("benchmarks") or [row["benchmark_id"] for row in data.get("table_rows", [])])


def stable_seed(base_seed: int, *parts: str) -> int:
    digest = hashlib.sha256("::".join(parts).encode()).hexdigest()
    return int(base_seed) + int(digest[:8], 16) % 1_000_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-protocol", type=Path, default=Path("configs/eval_protocol_coverage_only.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("autoresearch/hfc-ablation-budget5-atalanta300k-v1"))
    parser.add_argument("--checkpoint", type=Path, default=Path("autoresearch/hard-fault-cone-distill-train-smallmid-v2/best.pt"))
    parser.add_argument("--real-fault-priors", default="autoresearch/eval8-real-priors-budget5-v1/real_fault_priors.csv")
    parser.add_argument("--activation-priors", default="autoresearch/eval8-real-priors-budget5-v1/activation_priors_30k.csv")
    parser.add_argument("--methods", default="hfc_model,hfc_heuristic,hfc_random")
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=96)
    parser.add_argument("--patterns", type=int, default=300000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--eval-backend", choices=["tmax", "atalanta-bist"], default="atalanta-bist")
    parser.add_argument("--timeout-sec", type=int, default=14400)
    parser.add_argument("--atalanta-bin", default="/data3/pengqingsong/DFT/DeepTPI-project/external/DeepTPI/src/external/Atalanta_BIST/atalanta")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-planner", choices=["greedy", "beam", "beam_full"], default="beam")
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--lookahead-depth", type=int, default=3)
    parser.add_argument("--score-field", default="reward_pred")
    parser.add_argument("--beam-objective", default="cumulative")
    parser.add_argument("--discount-gamma", type=float, default=1.0)
    parser.add_argument("--cleanup-workdir", action="store_true", default=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str, sort_keys=True) + "\n")
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    benchmarks = protocol_benchmarks(args.eval_protocol)
    results_path = args.out_dir / "results.tsv"
    if args.force and results_path.exists():
        results_path.unlink()
    completed = load_completed_results(results_path) if not args.force else set()

    # Reuse the already-computed model plans/evals when available. This keeps the
    # ablation centered on the new heuristic/random controls.
    existing_model_root = Path("autoresearch/eval8-existing-plans-atalanta300k-v1")
    existing_model_results = existing_model_root / "results.tsv"
    existing_by_bench: dict[str, dict[str, str]] = {}
    if existing_model_results.is_file() and not args.force:
        with existing_model_results.open(newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                existing_by_bench[row["benchmark_id"]] = row

    for method in methods:
        for benchmark in benchmarks:
            if (method, benchmark) in completed:
                continue
            started = time.time()
            plan_csv = args.out_dir / "plans" / method / f"{benchmark}.csv"
            eval_dir = args.out_dir / "evals" / method / benchmark
            plan_log = args.out_dir / "logs" / method / f"{benchmark}.plan.log"
            eval_log = args.out_dir / "logs" / method / f"{benchmark}.eval.log"
            row = {
                "method": method,
                "selector": method,
                "benchmark_id": benchmark,
                "budget": args.budget,
                "max_candidates": args.max_candidates,
                "patterns": args.patterns,
                "seed": args.seed,
                "plan_csv": str(plan_csv),
                "eval_dir": str(eval_dir),
            }
            if method == "hfc_model" and benchmark in existing_by_bench and not args.force:
                source = existing_by_bench[benchmark]
                source_plan = Path(source["plan_csv"])
                plan_csv.parent.mkdir(parents=True, exist_ok=True)
                if source_plan.is_file() and not plan_csv.is_file():
                    shutil.copy2(source_plan, plan_csv)
                row.update(
                    {
                        "status": source.get("status", "ok"),
                        "baseline_test_coverage": source.get("baseline_test_coverage"),
                        "final_test_coverage": source.get("final_test_coverage"),
                        "delta_test_coverage": source.get("delta_test_coverage"),
                        "baseline_fault_coverage": source.get("baseline_fault_coverage"),
                        "final_fault_coverage": source.get("final_fault_coverage"),
                        "delta_fault_coverage": source.get("delta_fault_coverage"),
                        "elapsed_sec": round(time.time() - started, 3),
                    }
                )
                append_tsv(results_path, row)
                continue

            if method == "hfc_model":
                ok, error = plan_with_model(args, benchmark, plan_csv, plan_log)
            elif method in {"hfc_heuristic", "hfc_random"}:
                if not plan_csv.is_file() or args.force:
                    rows = heuristic_plan(
                        benchmark_id=benchmark,
                        budget=args.budget,
                        max_candidates=args.max_candidates,
                        real_fault_priors=args.real_fault_priors,
                        activation_priors=args.activation_priors,
                        selector=method,
                        seed=stable_seed(args.seed, method, benchmark),
                    )
                    write_plan_csv(plan_csv, rows)
                ok, error = True, ""
            else:
                raise ValueError(f"unknown method: {method}")
            if not ok:
                row.update({"status": "plan_error", "error": error, "elapsed_sec": round(time.time() - started, 3)})
                append_tsv(results_path, row)
                continue

            ok, error = evaluate(args, benchmark, plan_csv, eval_dir, eval_log)
            if not ok:
                row.update({"status": "eval_error", "error": error, "elapsed_sec": round(time.time() - started, 3)})
                append_tsv(results_path, row)
                continue
            row.update(read_final_metrics(eval_dir / "labels.csv"))
            row["elapsed_sec"] = round(time.time() - started, 3)
            append_tsv(results_path, row)

    summarize(results_path, args.out_dir / "ablation_report.md", args.eval_protocol)
    print(json.dumps({"results": str(results_path), "report": str(args.out_dir / "ablation_report.md")}, indent=2))


if __name__ == "__main__":
    main()
