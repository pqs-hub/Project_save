#!/usr/bin/env python3
"""Verify that an exact-legal five-circuit result uses one shared method."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CIRCUITS = ("b15_C", "b20_C", "b21_C", "b22_C", "b17_C")
RESULT_METHOD_FIELDS = (
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
    "prefix_state_mode",
    "residual_prior_source",
)
CONFIG_METHOD_FIELDS = (
    "checkpoint",
    "ensemble_checkpoints",
    "ensemble_lcb_alpha",
    "eval_backend",
    "eval_step_mode",
    "patterns",
    "seed",
    "planners",
    "score_fields",
    "beam_objectives",
    "beam_widths",
    "lookahead_depths",
    "max_candidates",
    "k_recalls",
    "k_models",
    "k_plans",
    "candidate_strategies",
    "candidate_diversity_penalties",
    "candidate_diversity_depths",
    "candidate_sample_seeds",
    "planner_environment",
    "refresh_ratio_numerator",
    "refresh_ratio_denominator",
    "refresh_rounding",
    "prefix_state_mode",
    "residual_prior_source",
    "extra_intermediate_fault_simulation",
    "candidate_prior_alpha",
)


def table(path: Path, delimiter: str) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def resolve(repo: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo / path


def identical_fingerprint(records: dict[str, dict], fields: tuple[str, ...], label: str) -> dict:
    fingerprints = {
        circuit: {field: record.get(field) for field in fields}
        for circuit, record in records.items()
    }
    serialized = {json.dumps(value, sort_keys=True) for value in fingerprints.values()}
    if len(serialized) != 1:
        raise RuntimeError(f"{label} differs by circuit: {json.dumps(fingerprints, indent=2)}")
    return next(iter(fingerprints.values()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--mapping-root",
        type=Path,
        default=Path("autoresearch/original-netlist-recovery-260712/exact_itc99"),
    )
    parser.add_argument("--require-beats-deeptpi", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = resolve(repo, str(args.root))
    mapping_root = resolve(repo, str(args.mapping_root))

    results: dict[str, dict[str, str]] = {}
    configs: dict[str, dict] = {}
    legal_nodes = 0
    plan_nodes = 0
    for circuit in CIRCUITS:
        attempts = table(root / circuit / "results.tsv", "\t")
        successful = [row for row in attempts if row.get("status") == "ok"]
        if len(successful) != 1:
            raise RuntimeError(f"{circuit}: expected exactly one successful attempt, got {len(successful)}")
        result = successful[0]
        results[circuit] = result
        configs[circuit] = json.loads((root / circuit / "config.json").read_text())
        allowed = {
            line.strip()
            for line in (mapping_root / circuit / "exact_candidate_nodes.txt").read_text().splitlines()
            if line.strip()
        }
        plan = table(resolve(repo, result["plan_csv"]), ",")
        illegal = sorted({row["node"] for row in plan} - allowed)
        if illegal:
            raise RuntimeError(f"{circuit}: illegal original-netlist nodes: {illegal[:10]}")
        if len(plan) != int(result["budget"]):
            raise RuntimeError(f"{circuit}: plan length {len(plan)} != budget {result['budget']}")
        ratio_numerator = result.get("refresh_ratio_numerator") or ""
        ratio_denominator = result.get("refresh_ratio_denominator") or ""
        if ratio_numerator or ratio_denominator:
            if not ratio_numerator or not ratio_denominator:
                raise RuntimeError(f"{circuit}: incomplete residual-refresh ratio")
            if result.get("refresh_rounding") != "floor":
                raise RuntimeError(f"{circuit}: unsupported refresh rounding {result.get('refresh_rounding')}")
            expected_refresh = (
                int(result["budget"]) * int(ratio_numerator) // int(ratio_denominator)
            )
            if int(result.get("refresh_steps") or -1) != expected_refresh:
                raise RuntimeError(
                    f"{circuit}: refresh step {result.get('refresh_steps')} != fixed-ratio floor {expected_refresh}"
                )
        plan_nodes += len(plan)
        legal_nodes += len(plan) - len(illegal)

    result_fingerprint = identical_fingerprint(results, RESULT_METHOD_FIELDS, "result method")
    config_fingerprint = identical_fingerprint(configs, CONFIG_METHOD_FIELDS, "config method")
    if result_fingerprint["patterns"] != "300000" or result_fingerprint["seed"] != "2026":
        raise RuntimeError("non-standard ATPG protocol: expected patterns=300000 and seed=2026")

    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    beats = bool(summary and summary["aggregate"].get("all_beat_deeptpi"))
    if args.require_beats_deeptpi and not beats:
        raise SystemExit("FAIL: uniform and exact-legal, but not every circuit beats DeepTPI")
    print(
        json.dumps(
            {
                "status": "PASS",
                "uniform_method": True,
                "exact_legal": legal_nodes == plan_nodes,
                "legal_plan_nodes": legal_nodes,
                "plan_nodes": plan_nodes,
                "all_beat_deeptpi": beats,
                "result_method": result_fingerprint,
                "config_method": config_fingerprint,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
