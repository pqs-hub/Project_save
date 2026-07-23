#!/usr/bin/env python3
"""Materialize a direct five-circuit residual-refresh run as an audited sweep."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
from typing import Any


CIRCUITS = (
    ("b15_C", "iscas99__b15_1", 278, 27353),
    ("b20_C", "iscas99__b20", 616, 61145),
    ("b21_C", "iscas99__b21", 628, 62295),
    ("b22_C", "iscas99__b22", 915, 90738),
    ("b17_C", "iscas99__b17", 994, 98014),
)

PLANNER_ENV_KEYS = (
    "TPI_CANDIDATE_PRIOR_ALPHA",
    "TPI_ADAPTIVE_BASE_CANDIDATES",
    "TPI_ADAPTIVE_EXPANSION_MARGIN",
    "TPI_ADAPTIVE_MARGIN_MODE",
    "TPI_HARD_CLUSTER_MAX_HARD_NODES",
    "TPI_LATENT_NORM_CLIP_RATIO",
    "TPI_PLAN_THREADS",
    "TPI_Q_CONTEXT_DISAGREEMENT_BETA",
    "TPI_Q_CONTEXT_SUPPORT_ALPHA",
    "TPI_SCORE_QUANTIZATION",
    "TPI_TORCH_DETERMINISTIC",
    "TPI_TYPED_RELIABLE_CP0_MIN_HEADS",
    "TPI_TYPED_RELIABLE_MARGINAL_WEIGHT",
    "TPI_TYPED_RELIABLE_MIN_HEADS",
    "TPI_TYPED_RESIDUAL_ALPHA",
    "TPI_TYPED_RESIDUAL_CLIP",
    "TPI_TYPED_RESIDUAL_DECAY_START",
    "TPI_TYPED_RESIDUAL_DECAY_RESET_ON_PREFIX",
    "TPI_TYPED_RESIDUAL_DECAY_STEPS",
    "TPI_TYPED_RESIDUAL_DISAGREEMENT_BETA",
    "TPI_TYPED_TRUST_ADVANTAGE_MARGIN",
    "TPI_TYPED_TRUST_CP0_MIN_HEADS",
    "TPI_TYPED_TRUST_HEAD_MARGIN",
    "TPI_TYPED_TRUST_MIN_HEADS",
)

RESULT_FIELDS = (
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
    "k_recall",
    "k_model",
    "k_plan",
    "discount_gamma",
    "candidate_strategy",
    "candidate_diversity_penalty",
    "candidate_diversity_depth",
    "candidate_sample_seed",
    "patterns",
    "seed",
    "candidate_prior_alpha",
    "refresh_ratio_numerator",
    "refresh_ratio_denominator",
    "refresh_rounding",
    "refresh_steps",
    "prefix_state_mode",
    "residual_prior_source",
    "source_bench_sha256",
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
    "prior_setup_elapsed_sec",
    "plan_elapsed_sec",
    "eval_elapsed_sec",
    "elapsed_sec",
    "error",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def elapsed(path: Path) -> float:
    return float(path.read_text().strip()) if path.is_file() else 0.0


def total(rows: list[dict[str, str]], field: str) -> float:
    return sum(float(row.get(field) or 0.0) for row in rows)


def relative_to_repo(path: Path, repo: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repo.resolve()))
    except ValueError:
        return str(resolved)


def write_tsv(path: Path, row: dict[str, Any]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--incumbent-root", type=Path, required=True)
    parser.add_argument("--mapping-root", type=Path, required=True)
    parser.add_argument("--ratio-numerator", type=int, required=True)
    parser.add_argument("--ratio-denominator", type=int, required=True)
    parser.add_argument("--rounding", choices=("floor",), default="floor")
    parser.add_argument("--prefix-state-mode", choices=("replay", "reencode"), required=True)
    parser.add_argument("--patterns", type=int, default=300000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--candidate-prior-alpha", type=float, default=0.0)
    parser.add_argument("--score-field", default="q_typed_reliable_context")
    parser.add_argument("--max-candidates", type=int, default=64)
    args = parser.parse_args()

    if args.ratio_numerator <= 0 or args.ratio_denominator <= 0:
        raise SystemExit("refresh ratio must be positive")
    if args.max_candidates <= 0:
        raise SystemExit("max candidates must be positive")
    repo = Path(__file__).resolve().parents[1]
    root = args.root if args.root.is_absolute() else repo / args.root
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else repo / args.checkpoint
    incumbent_root = args.incumbent_root if args.incumbent_root.is_absolute() else repo / args.incumbent_root
    mapping_root = args.mapping_root if args.mapping_root.is_absolute() else repo / args.mapping_root
    planner_environment = {key: os.environ.get(key) for key in PLANNER_ENV_KEYS}
    method = {
        "checkpoint": relative_to_repo(checkpoint, repo),
        "score_field": args.score_field,
        "candidate_strategy": "hard_fault_cluster",
        "max_candidates": args.max_candidates,
        "refresh_ratio_numerator": args.ratio_numerator,
        "refresh_ratio_denominator": args.ratio_denominator,
        "refresh_rounding": args.rounding,
        "prefix_state_mode": args.prefix_state_mode,
        "residual_prior_source": "intermediate_atpg_last_nonbaseline_row",
        "patterns": args.patterns,
        "seed": args.seed,
        "candidate_prior_alpha": args.candidate_prior_alpha,
        "extra_intermediate_fault_simulation": True,
    }
    manifests: list[dict[str, Any]] = []

    for circuit, benchmark, budget, logic_gates in CIRCUITS:
        circuit_root = root / circuit
        plan_path = circuit_root / "plans" / f"{benchmark}.csv"
        eval_dir = circuit_root / "evals" / "final"
        labels_path = eval_dir / "labels.csv"
        prior_path = circuit_root / "residual_priors" / "real_fault_priors.csv"
        prefix_eval = circuit_root / "prefix_eval" / "labels.csv"
        prefix_plan = incumbent_root / circuit / "best_plans" / f"{benchmark}.csv"
        allowlist_path = mapping_root / circuit / "exact_candidate_nodes.txt"
        required = (plan_path, labels_path, prior_path, prefix_eval, prefix_plan, allowlist_path)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(f"{circuit}: missing required artifacts: {missing}")

        refresh_steps = budget * args.ratio_numerator // args.ratio_denominator
        plan = read_csv(plan_path)
        if len(plan) != budget:
            raise RuntimeError(f"{circuit}: plan length {len(plan)} != budget {budget}")
        if [int(row["step"]) for row in plan] != list(range(1, budget + 1)):
            raise RuntimeError(f"{circuit}: plan steps are not contiguous 1..{budget}")
        allowed = {line.strip() for line in allowlist_path.read_text().splitlines() if line.strip()}
        illegal = sorted({row["node"] for row in plan} - allowed)
        if illegal:
            raise RuntimeError(f"{circuit}: illegal original-netlist nodes: {illegal[:10]}")

        labels = read_csv(labels_path)
        successful = [row for row in labels if row.get("status") == "ok"]
        baseline = next(row for row in successful if int(row["step"]) == 0)
        final = max(successful, key=lambda row: int(row["step"]))
        if int(final["step"]) != budget:
            raise RuntimeError(f"{circuit}: final evaluated step {final['step']} != budget {budget}")
        source_shas = {row["source_bench_sha256"] for row in successful}
        if len(source_shas) != 1:
            raise RuntimeError(f"{circuit}: inconsistent source BENCH hashes: {sorted(source_shas)}")
        source_sha = next(iter(source_shas))
        prefix_labels = read_csv(prefix_eval)
        prefix_final = max(
            (row for row in prefix_labels if row.get("status") == "ok"),
            key=lambda row: int(row["step"]),
        )
        if int(prefix_final["step"]) != refresh_steps:
            raise RuntimeError(
                f"{circuit}: prefix evaluated step {prefix_final['step']} != fixed-ratio step {refresh_steps}"
            )

        best_dir = circuit_root / "best_plans"
        best_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(plan_path, best_dir / f"{benchmark}.csv")
        prior_elapsed = elapsed(circuit_root / "timing" / "prefix_eval_sec.txt")
        plan_elapsed = elapsed(circuit_root / "timing" / "plan_sec.txt")
        eval_elapsed = elapsed(circuit_root / "timing" / "final_eval_sec.txt")
        variant = (
            "residual_refresh__replay__"
            f"ratio{args.ratio_numerator}of{args.ratio_denominator}__"
            f"prioralpha{args.candidate_prior_alpha:g}__{args.score_field}__"
            f"c{args.max_candidates}"
        )
        row: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "variant_id": variant,
            "status": "ok",
            "benchmark_id": benchmark,
            "logic_gates": logic_gates,
            "budget_mode": "deeptpi_table2_fixed",
            "budget": budget,
            "planner": "greedy",
            "beam_objective": "cumulative",
            "score_field": method["score_field"],
            "beam_width": 1,
            "lookahead_depth": 1,
            "max_candidates": method["max_candidates"],
            "k_recall": "",
            "k_model": "",
            "k_plan": "",
            "discount_gamma": 1.0,
            "candidate_strategy": method["candidate_strategy"],
            "candidate_diversity_penalty": 0.0,
            "candidate_diversity_depth": 4,
            "candidate_sample_seed": 0,
            "patterns": args.patterns,
            "seed": args.seed,
            "candidate_prior_alpha": args.candidate_prior_alpha,
            "refresh_ratio_numerator": args.ratio_numerator,
            "refresh_ratio_denominator": args.ratio_denominator,
            "refresh_rounding": args.rounding,
            "refresh_steps": refresh_steps,
            "prefix_state_mode": args.prefix_state_mode,
            "residual_prior_source": method["residual_prior_source"],
            "source_bench_sha256": source_sha,
            "plan_score_sum": total(plan, "score_pred"),
            "plan_reward_sum": total(plan, "reward_pred"),
            "plan_fc_sum": total(plan, "fc_pred"),
            "plan_return_sum": total(plan, "return_pred"),
            "plan_sequence_sum": total(plan, "sequence_score"),
            "plan_objective_sum": total(plan, args.score_field),
            "delta_test_coverage": float(final["delta_test_coverage"]),
            "delta_fault_coverage": float(final["delta_fault_coverage"]),
            "delta_pattern_count": int(final["delta_pattern_count"]),
            "plan_csv": relative_to_repo(plan_path, repo),
            "eval_dir": relative_to_repo(eval_dir, repo),
            "prior_setup_elapsed_sec": prior_elapsed,
            "plan_elapsed_sec": plan_elapsed,
            "eval_elapsed_sec": eval_elapsed,
            "elapsed_sec": prior_elapsed + plan_elapsed + eval_elapsed,
            "error": "",
        }
        write_tsv(circuit_root / "results.tsv", row)
        config = {
            "checkpoint": method["checkpoint"],
            "ensemble_checkpoints": None,
            "ensemble_lcb_alpha": 1.0,
            "eval_backend": "atalanta-bist",
            "eval_step_mode": "final",
            "patterns": args.patterns,
            "seed": args.seed,
            "planners": "greedy",
            "score_fields": method["score_field"],
            "beam_objectives": "cumulative",
            "beam_widths": "1",
            "lookahead_depths": "1",
            "max_candidates": str(method["max_candidates"]),
            "k_recalls": "",
            "k_models": "",
            "k_plans": "",
            "candidate_strategies": method["candidate_strategy"],
            "candidate_diversity_penalties": "0.0",
            "candidate_diversity_depths": "4",
            "candidate_sample_seeds": "0",
            "planner_environment": planner_environment,
            "refresh_ratio_numerator": args.ratio_numerator,
            "refresh_ratio_denominator": args.ratio_denominator,
            "refresh_rounding": args.rounding,
            "prefix_state_mode": args.prefix_state_mode,
            "residual_prior_source": method["residual_prior_source"],
            "extra_intermediate_fault_simulation": True,
            "candidate_prior_alpha": args.candidate_prior_alpha,
            "benchmark": benchmark,
            "budget": budget,
            "refresh_steps": refresh_steps,
            "prefix_plan": relative_to_repo(prefix_plan, repo),
            "candidate_allowlist": relative_to_repo(allowlist_path, repo),
            "candidate_real_fault_priors": relative_to_repo(prior_path, repo),
            "source_bench_sha256": source_sha,
        }
        (circuit_root / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        manifests.append(
            {
                "circuit": circuit,
                "benchmark_id": benchmark,
                "budget": budget,
                "refresh_steps": refresh_steps,
                "source_bench_sha256": source_sha,
                "baseline_test_coverage": float(baseline["test_coverage"]),
                "final_test_coverage": float(final["test_coverage"]),
                "delta_test_coverage": float(final["delta_test_coverage"]),
                "legal_plan_nodes": len(plan),
                "plan_csv": row["plan_csv"],
                "eval_dir": row["eval_dir"],
            }
        )

    payload = {"method": method, "per_circuit": manifests}
    (root / "refresh_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
