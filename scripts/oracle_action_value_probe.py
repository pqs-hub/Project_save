"""Probe whether model action scores align with backend-measured TC gains."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tpi_jepa.bench import parse_bench  # noqa: E402
from tpi_jepa.evaluate_plan_tmax import evaluate_plan  # noqa: E402
from tpi_jepa.features import make_base_node_features, make_state_features  # noqa: E402
from tpi_jepa.graph import build_graph  # noqa: E402
from tpi_jepa.labels import find_bench_path  # noqa: E402
from tpi_jepa.plan import (  # noqa: E402
    PLAN_FIELDNAMES,
    enumerate_candidates,
    load_checkpoint,
    score_candidate_from_latent,
    set_real_fault_context,
)


ACTION_TO_CANONICAL = {
    "CP0": "control0",
    "CP1": "control1",
    "OP": "observe",
    "control0": "control0",
    "control1": "control1",
    "observe": "observe",
}

DEFAULT_ATALANTA = "/data4/pengqingsong/DFT/tool/atalanta_bist_with_ufaults/atalanta"
DEFAULT_TMAX = "/data3/pengqingsong/synopsys/txs/O-2018.06-SP1/bin/tmax"
SCORE_FIELDS = [
    "reward_pred",
    "fc_pred",
    "guarded_reward",
    "return_pred",
    "hard_reduction_total_pred",
    "hybrid_pred",
]
ORACLE_FIELDS = [
    "benchmark_id",
    "state_id",
    "candidate_strategy",
    "candidate_rank",
    "node",
    "type",
    "action_key",
    "status",
    "oracle_delta_tc",
    "oracle_delta_fault_coverage",
    "oracle_delta_pattern_count",
    "oracle_test_coverage",
    "oracle_fault_coverage",
    "oracle_hard_fault_count",
    "oracle_undetected_fault_count",
    "oracle_error",
    "eval_dir",
    *SCORE_FIELDS,
    "score_pred",
    "pattern_pred",
    "hard_reduction_sa0_pred",
    "hard_reduction_sa1_pred",
]
RANK_FIELDS = [
    "benchmark_id",
    "state_id",
    "candidate_strategy",
    "score_field",
    "k",
    "oracle_top_m",
    "candidate_count",
    "oracle_action_recall",
    "oracle_node_recall",
    "regret",
    "best_in_topk_delta_tc",
    "oracle_best_delta_tc",
    "top1_real_delta_tc",
    "top1_regret",
    "negative_top1",
]
PRED_FIELDS = [
    "benchmark_id",
    "state_id",
    "candidate_strategy",
    "score_field",
    "candidate_count",
    "spearman",
    "kendall_tau",
    "pearson",
    "topk",
    "topk_real_gain_by_pred",
    "topk_hit_rate",
    "top1_real_delta_tc",
    "top1_regret",
    "sign_accuracy",
    "negative_top1",
    "calibration_slope",
    "calibration_intercept",
]
SUMMARY_FIELDS = [
    "benchmark_id",
    "state_id",
    "candidate_strategy",
    "score_field",
    "oracle_best_action",
    "oracle_best_delta_tc",
    "model_top1_action",
    "model_top1_pred",
    "model_top1_real_delta_tc",
    "top1_regret",
    "negative_top1",
]
GROUP_FIELDS = [
    "benchmark_id",
    "state_id",
    "candidate_strategy",
    "candidate_count",
    "finite_count",
    "ok_count",
    "positive_count",
    "zero_count",
    "negative_count",
    "oracle_best_action",
    "oracle_best_delta_tc",
    "oracle_worst_action",
    "oracle_worst_delta_tc",
    "mean_delta_tc",
    "base_test_coverage",
    "base_fault_coverage",
]
ORACLE_BACKEND_FIELDS = [
    "status",
    "oracle_delta_tc",
    "oracle_delta_fault_coverage",
    "oracle_delta_pattern_count",
    "oracle_test_coverage",
    "oracle_fault_coverage",
    "oracle_hard_fault_count",
    "oracle_undetected_fault_count",
    "oracle_error",
    "eval_dir",
]


def parse_csv_values(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_int_values(text: str) -> list[int]:
    return [int(item) for item in parse_csv_values(text)]


def canonical_actions(text: str) -> list[str]:
    actions = []
    for item in parse_csv_values(text):
        if item not in ACTION_TO_CANONICAL:
            raise ValueError(f"Unsupported action type: {item!r}")
        actions.append(ACTION_TO_CANONICAL[item])
    return actions


def action_key(node: str, action_type: str) -> str:
    return f"{node}::{action_type}"


def safe_float(value: Any, default: float = float("nan")) -> float:
    if value in (None, "", "NA"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def finite_pairs(xs: list[float], ys: list[float]) -> tuple[list[float], list[float]]:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if not pairs:
        return [], []
    left, right = zip(*pairs)
    return list(left), list(right)


def mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else float("nan")


def pearson(xs: list[float], ys: list[float]) -> float:
    xs, ys = finite_pairs(xs, ys)
    if len(xs) < 2:
        return float("nan")
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    out = [0.0 for _ in values]
    pos = 0
    while pos < len(order):
        end = pos + 1
        while end < len(order) and values[order[end]] == values[order[pos]]:
            end += 1
        avg_rank = (pos + 1 + end) / 2.0
        for idx in order[pos:end]:
            out[idx] = avg_rank
        pos = end
    return out


def spearman(xs: list[float], ys: list[float]) -> float:
    xs, ys = finite_pairs(xs, ys)
    if len(xs) < 2:
        return float("nan")
    return pearson(ranks(xs), ranks(ys))


def kendall_tau(xs: list[float], ys: list[float]) -> float:
    xs, ys = finite_pairs(xs, ys)
    if len(xs) < 2:
        return float("nan")
    concordant = 0
    discordant = 0
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            prod = dx * dy
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else float("nan")


def linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    xs, ys = finite_pairs(xs, ys)
    if len(xs) < 2:
        return float("nan"), float("nan")
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 0:
        return float("nan"), my
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return slope, my - slope * mx


def sign(value: float) -> int:
    if not math.isfinite(value) or value == 0.0:
        return 0
    return 1 if value > 0 else -1


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def checkpoint_id(path: str | Path) -> str:
    path = Path(path)
    try:
        stat = path.stat()
        payload = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    except OSError:
        payload = str(path)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def resume_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("benchmark_id", "")),
        str(row.get("state_id", "")),
        str(row.get("candidate_strategy", "")),
        str(row.get("action_key", "")),
    )


def load_resume_rows(path: Path) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    return {resume_key(row): row for row in read_tsv(path)}


def plan_csv_for(path: Path, selected: list[tuple[str, str]], scored_row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for step, (node, action_type) in enumerate(selected, start=1):
        rows.append({"step": step, "node": node, "type": action_type})
    row = {key: value for key, value in scored_row.items() if not key.startswith("_")}
    row["step"] = len(rows) + 1
    rows.append(row)
    fieldnames = [field for field in PLAN_FIELDNAMES if any(field in item for item in rows)]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: item.get(field, "") for field in fieldnames} for item in rows)


def candidate_actions_for_strategy(
    *,
    graph,
    selected: list[tuple[str, str]],
    strategy: str,
    max_nets: int,
    action_types: list[str],
    benchmark_id: str,
    real_fault_priors: str | None,
    activation_priors: str | None,
    candidate_cache_dir: str | None,
    candidate_sample_seed: int,
) -> list[tuple[str, str]]:
    raw = enumerate_candidates(
        graph,
        selected,
        max(max_nets * max(1, len(action_types)), max_nets),
        strategy,
        real_fault_benchmark_id=benchmark_id,
        real_fault_prior_path=real_fault_priors,
        activation_prior_path=activation_priors,
        candidate_cache_dir=candidate_cache_dir,
        candidate_sample_seed=candidate_sample_seed,
    )
    nets: list[str] = []
    seen_nets: set[str] = set()
    for node, _ in raw:
        if node in seen_nets:
            continue
        nets.append(node)
        seen_nets.add(node)
        if len(nets) >= max_nets:
            break
    used = set(selected)
    actions = []
    for node in nets:
        for action_type in action_types:
            candidate = (node, action_type)
            if candidate not in used:
                actions.append(candidate)
    return actions


@torch.no_grad()
def score_actions(
    *,
    model,
    graph,
    selected: list[tuple[str, str]],
    candidates: list[tuple[str, str]],
    config: dict[str, Any],
    device: torch.device,
    candidate_diversity_penalty: float,
    candidate_diversity_depth: int,
) -> dict[str, dict[str, Any]]:
    feature_mode = str(config.get("feature_mode", "basic"))
    relation_mode = str(config.get("relation_mode", "basic"))
    relation_depth = int(config.get("relation_depth", 8))
    base_features = make_base_node_features(
        graph,
        feature_mode,
        benchmark_id=None,
        real_fault_prior_path=config.get("real_fault_priors") or config.get("real_fault_prior_path"),
        activation_prior_path=config.get("activation_priors") or config.get("activation_prior_path"),
    )
    x_state = make_state_features(graph, selected, base_features).to(device)
    z_state = model.online_encoder(
        x_state,
        graph.edge_src.to(device),
        graph.edge_dst.to(device),
        graph.gate_type_ids.to(device),
    )
    rows = {}
    for candidate in candidates:
        row = score_candidate_from_latent(
            model,
            graph,
            z_state,
            candidate,
            device,
            relation_mode,
            relation_depth,
            selected,
            candidate_diversity_penalty,
            candidate_diversity_depth,
        )
        row.pop("_z_pred", None)
        rows[action_key(row["node"], row["type"])] = row
    return rows


def oracle_row_from_eval(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    return rows[-1]


def group_summary_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    base = {
        "benchmark_id": rows[0].get("benchmark_id", "") if rows else "",
        "state_id": rows[0].get("state_id", "") if rows else "",
        "candidate_strategy": rows[0].get("candidate_strategy", "") if rows else "",
    }
    finite = [row for row in rows if math.isfinite(safe_float(row.get("oracle_delta_tc")))]
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    positives = [row for row in finite if safe_float(row.get("oracle_delta_tc")) > 0.0]
    zeros = [row for row in finite if safe_float(row.get("oracle_delta_tc")) == 0.0]
    negatives = [row for row in finite if safe_float(row.get("oracle_delta_tc")) < 0.0]
    best = max(finite, key=lambda row: safe_float(row.get("oracle_delta_tc")), default={})
    worst = min(finite, key=lambda row: safe_float(row.get("oracle_delta_tc")), default={})
    base_tc = float("nan")
    base_fc = float("nan")
    for row in finite:
        tc = safe_float(row.get("oracle_test_coverage"))
        dtc = safe_float(row.get("oracle_delta_tc"))
        fc = safe_float(row.get("oracle_fault_coverage"))
        dfc = safe_float(row.get("oracle_delta_fault_coverage"))
        if math.isfinite(tc) and math.isfinite(dtc):
            base_tc = tc - dtc
        if math.isfinite(fc) and math.isfinite(dfc):
            base_fc = fc - dfc
        if math.isfinite(base_tc) or math.isfinite(base_fc):
            break
    return {
        **base,
        "candidate_count": len(rows),
        "finite_count": len(finite),
        "ok_count": len(ok_rows),
        "positive_count": len(positives),
        "zero_count": len(zeros),
        "negative_count": len(negatives),
        "oracle_best_action": best.get("action_key", ""),
        "oracle_best_delta_tc": best.get("oracle_delta_tc", ""),
        "oracle_worst_action": worst.get("action_key", ""),
        "oracle_worst_delta_tc": worst.get("oracle_delta_tc", ""),
        "mean_delta_tc": mean([safe_float(row.get("oracle_delta_tc")) for row in finite]),
        "base_test_coverage": base_tc,
        "base_fault_coverage": base_fc,
    }


def evaluate_candidate(
    *,
    benchmark_id: str,
    state_id: str,
    selected: list[tuple[str, str]],
    scored_row: dict[str, Any],
    out_dir: Path,
    patterns: int,
    seed: int,
    backend: str,
    tmax_bin: str,
    atalanta_bin: str,
    timeout_sec: int,
    dry_run: bool,
    cleanup_workdir: bool,
) -> dict[str, Any]:
    key = action_key(scored_row["node"], scored_row["type"])
    action_dir = out_dir / "evals" / benchmark_id / state_id / key.replace("/", "_").replace("::", "__")
    plan_csv = action_dir / "plan.csv"
    plan_csv_for(plan_csv, selected, scored_row)
    rows = evaluate_plan(
        benchmark_id=benchmark_id,
        plan_csv=plan_csv,
        out_dir=action_dir,
        patterns=patterns,
        seed=seed,
        backend=backend,
        tmax_bin=tmax_bin,
        atalanta_bin=atalanta_bin,
        timeout_sec=timeout_sec,
        force=True,
        dry_run=dry_run,
        cleanup_workdir=cleanup_workdir,
    )
    return oracle_row_from_eval(rows) | {"eval_dir": str(action_dir)}


def metric_rows_for_group(
    *,
    rows: list[dict[str, Any]],
    score_fields: list[str],
    ks: list[int],
    oracle_top_m: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows = []
    pred_rows = []
    summary_rows = []
    valid = [row for row in rows if math.isfinite(safe_float(row.get("oracle_delta_tc")))]
    if not valid:
        return metric_rows, pred_rows, summary_rows
    oracle_sorted = sorted(valid, key=lambda row: safe_float(row.get("oracle_delta_tc")), reverse=True)
    oracle_best = oracle_sorted[0]
    oracle_best_delta = safe_float(oracle_best.get("oracle_delta_tc"))
    oracle_top = oracle_sorted[: max(1, min(oracle_top_m, len(oracle_sorted)))]
    oracle_top_actions = {row["action_key"] for row in oracle_top}
    oracle_top_nodes = {row["node"] for row in oracle_top}
    base = {
        "benchmark_id": valid[0]["benchmark_id"],
        "state_id": valid[0]["state_id"],
        "candidate_strategy": valid[0]["candidate_strategy"],
    }
    for score_field in score_fields:
        scored = [row for row in valid if math.isfinite(safe_float(row.get(score_field)))]
        if not scored:
            continue
        pred_sorted = sorted(scored, key=lambda row: safe_float(row.get(score_field)), reverse=True)
        top1 = pred_sorted[0]
        top1_delta = safe_float(top1.get("oracle_delta_tc"))
        top1_regret = oracle_best_delta - top1_delta if math.isfinite(top1_delta) else float("nan")
        summary_rows.append(
            {
                **base,
                "score_field": score_field,
                "oracle_best_action": oracle_best["action_key"],
                "oracle_best_delta_tc": oracle_best_delta,
                "model_top1_action": top1["action_key"],
                "model_top1_pred": safe_float(top1.get(score_field)),
                "model_top1_real_delta_tc": top1_delta,
                "top1_regret": top1_regret,
                "negative_top1": int(math.isfinite(top1_delta) and top1_delta < 0.0),
            }
        )
        preds = [safe_float(row.get(score_field)) for row in scored]
        deltas = [safe_float(row.get("oracle_delta_tc")) for row in scored]
        slope, intercept = linear_fit(preds, deltas)
        topk = min(max(ks) if ks else 1, len(pred_sorted))
        pred_topk = pred_sorted[:topk]
        pred_topk_actions = {row["action_key"] for row in pred_topk}
        sign_pairs = finite_pairs(preds, deltas)
        sign_acc = (
            sum(1 for pred, delta in zip(*sign_pairs) if sign(pred) == sign(delta)) / len(sign_pairs[0])
            if sign_pairs[0]
            else float("nan")
        )
        pred_rows.append(
            {
                **base,
                "score_field": score_field,
                "candidate_count": len(scored),
                "spearman": spearman(preds, deltas),
                "kendall_tau": kendall_tau(preds, deltas),
                "pearson": pearson(preds, deltas),
                "topk": topk,
                "topk_real_gain_by_pred": mean([safe_float(row.get("oracle_delta_tc")) for row in pred_topk]),
                "topk_hit_rate": len(pred_topk_actions & oracle_top_actions) / max(1, len(oracle_top_actions)),
                "top1_real_delta_tc": top1_delta,
                "top1_regret": top1_regret,
                "sign_accuracy": sign_acc,
                "negative_top1": int(math.isfinite(top1_delta) and top1_delta < 0.0),
                "calibration_slope": slope,
                "calibration_intercept": intercept,
            }
        )
        for k in ks:
            topk_rows = pred_sorted[: min(k, len(pred_sorted))]
            topk_actions = {row["action_key"] for row in topk_rows}
            topk_nodes = {row["node"] for row in topk_rows}
            best_topk_delta = max((safe_float(row.get("oracle_delta_tc")) for row in topk_rows), default=float("nan"))
            metric_rows.append(
                {
                    **base,
                    "score_field": score_field,
                    "k": k,
                    "oracle_top_m": len(oracle_top),
                    "candidate_count": len(scored),
                    "oracle_action_recall": len(topk_actions & oracle_top_actions) / max(1, len(oracle_top_actions)),
                    "oracle_node_recall": len(topk_nodes & oracle_top_nodes) / max(1, len(oracle_top_nodes)),
                    "regret": oracle_best_delta - best_topk_delta if math.isfinite(best_topk_delta) else float("nan"),
                    "best_in_topk_delta_tc": best_topk_delta,
                    "oracle_best_delta_tc": oracle_best_delta,
                    "top1_real_delta_tc": top1_delta,
                    "top1_regret": top1_regret,
                    "negative_top1": int(math.isfinite(top1_delta) and top1_delta < 0.0),
                }
            )
    return metric_rows, pred_rows, summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--benchmarks", required=True)
    parser.add_argument("--candidate-cache-dir", default=None)
    parser.add_argument("--candidate-strategies", default="cached_stride")
    parser.add_argument("--score-fields", default=",".join(SCORE_FIELDS))
    parser.add_argument("--states", default="initial")
    parser.add_argument("--max-nets", type=int, default=16)
    parser.add_argument("--action-types", default="CP0,CP1,OP")
    parser.add_argument("--top-ks", default="8,16,32")
    parser.add_argument("--oracle-top-m", type=int, default=5)
    parser.add_argument("--patterns", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--backend", choices=["tmax", "atalanta-bist"], default="atalanta-bist")
    parser.add_argument("--plan-device", default=None)
    parser.add_argument("--tmax-bin", default=DEFAULT_TMAX)
    parser.add_argument("--atalanta-bin", default=DEFAULT_ATALANTA)
    parser.add_argument("--timeout-sec", type=int, default=14400)
    parser.add_argument("--real-fault-priors", default=None)
    parser.add_argument("--activation-priors", default=None)
    parser.add_argument("--candidate-sample-seed", type=int, default=0)
    parser.add_argument("--candidate-diversity-penalty", type=float, default=0.0)
    parser.add_argument("--candidate-diversity-depth", type=int, default=4)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cleanup-workdir", action="store_true")
    args = parser.parse_args()

    states = parse_csv_values(args.states)
    if states != ["initial"]:
        raise ValueError("Only --states initial is currently supported")
    score_fields = parse_csv_values(args.score_fields)
    unsupported = [field for field in score_fields if field not in SCORE_FIELDS]
    if unsupported:
        raise ValueError(f"Unsupported score fields: {unsupported}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest) if args.manifest else out_dir / "manifest.json"
    resume_rows = load_resume_rows(out_dir / "oracle_actions.tsv") if args.resume else {}
    device_name = args.plan_device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    model, config = load_checkpoint(args.checkpoint, device)
    action_types = canonical_actions(args.action_types)
    ks = parse_int_values(args.top_ks)

    oracle_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    group_rows_out: list[dict[str, Any]] = []
    reused_oracle_rows = 0
    evaluated_oracle_rows = 0

    for benchmark_id in parse_csv_values(args.benchmarks):
        set_real_fault_context(benchmark_id, args.real_fault_priors, args.activation_priors)
        graph = build_graph(parse_bench(find_bench_path(benchmark_id)))
        selected: list[tuple[str, str]] = []
        state_id = "initial"
        for strategy in parse_csv_values(args.candidate_strategies):
            candidates = candidate_actions_for_strategy(
                graph=graph,
                selected=selected,
                strategy=strategy,
                max_nets=args.max_nets,
                action_types=action_types,
                benchmark_id=benchmark_id,
                real_fault_priors=args.real_fault_priors,
                activation_priors=args.activation_priors,
                candidate_cache_dir=args.candidate_cache_dir,
                candidate_sample_seed=args.candidate_sample_seed,
            )
            scored = score_actions(
                model=model,
                graph=graph,
                selected=selected,
                candidates=candidates,
                config=config,
                device=device,
                candidate_diversity_penalty=args.candidate_diversity_penalty,
                candidate_diversity_depth=args.candidate_diversity_depth,
            )
            group_rows = []
            for rank, candidate in enumerate(candidates, start=1):
                key = action_key(*candidate)
                scored_row = scored[key]
                previous = resume_rows.get((benchmark_id, state_id, strategy, key))
                if previous is not None:
                    oracle = {field: previous.get(field, "") for field in ORACLE_BACKEND_FIELDS}
                    reused_oracle_rows += 1
                else:
                    raw_oracle = evaluate_candidate(
                        benchmark_id=benchmark_id,
                        state_id=state_id,
                        selected=selected,
                        scored_row=scored_row,
                        out_dir=out_dir,
                        patterns=args.patterns,
                        seed=args.seed,
                        backend=args.backend,
                        tmax_bin=args.tmax_bin,
                        atalanta_bin=args.atalanta_bin,
                        timeout_sec=args.timeout_sec,
                        dry_run=args.dry_run,
                        cleanup_workdir=args.cleanup_workdir,
                    )
                    oracle = {
                        "status": raw_oracle.get("status"),
                        "oracle_delta_tc": raw_oracle.get("delta_test_coverage"),
                        "oracle_delta_fault_coverage": raw_oracle.get("delta_fault_coverage"),
                        "oracle_delta_pattern_count": raw_oracle.get("delta_pattern_count"),
                        "oracle_test_coverage": raw_oracle.get("test_coverage"),
                        "oracle_fault_coverage": raw_oracle.get("fault_coverage"),
                        "oracle_hard_fault_count": raw_oracle.get("hard_fault_count"),
                        "oracle_undetected_fault_count": raw_oracle.get("undetected_fault_count"),
                        "oracle_error": raw_oracle.get("error"),
                        "eval_dir": raw_oracle.get("eval_dir"),
                    }
                    evaluated_oracle_rows += 1
                row = {
                    "benchmark_id": benchmark_id,
                    "state_id": state_id,
                    "candidate_strategy": strategy,
                    "candidate_rank": rank,
                    "node": candidate[0],
                    "type": candidate[1],
                    "action_key": key,
                    **oracle,
                    **{field: scored_row.get(field) for field in ORACLE_FIELDS if field in scored_row},
                }
                oracle_rows.append(row)
                group_rows.append(row)
            group_rows_out.append(group_summary_row(group_rows))
            group_rank, group_pred, group_summary = metric_rows_for_group(
                rows=group_rows,
                score_fields=score_fields,
                ks=ks,
                oracle_top_m=args.oracle_top_m,
            )
            rank_rows.extend(group_rank)
            prediction_rows.extend(group_pred)
            summary_rows.extend(group_summary)

    write_tsv(out_dir / "oracle_actions.tsv", oracle_rows, ORACLE_FIELDS)
    write_tsv(out_dir / "oracle_groups.tsv", group_rows_out, GROUP_FIELDS)
    write_tsv(out_dir / "rank_metrics.tsv", rank_rows, RANK_FIELDS)
    write_tsv(out_dir / "prediction_metrics.tsv", prediction_rows, PRED_FIELDS)
    write_tsv(out_dir / "state_summary.tsv", summary_rows, SUMMARY_FIELDS)
    manifest = {
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(args.checkpoint),
        "checkpoint_id": checkpoint_id(args.checkpoint),
        "benchmarks": parse_csv_values(args.benchmarks),
        "candidate_strategies": parse_csv_values(args.candidate_strategies),
        "states": states,
        "max_nets": args.max_nets,
        "action_types": action_types,
        "patterns": args.patterns,
        "seed": args.seed,
        "backend": args.backend,
        "candidate_cache_dir": args.candidate_cache_dir,
        "score_fields": score_fields,
        "top_ks": ks,
        "oracle_top_m": args.oracle_top_m,
        "resume": bool(args.resume),
        "records": {
            "oracle_actions": len(oracle_rows),
            "oracle_groups": len(group_rows_out),
            "rank_metrics": len(rank_rows),
            "prediction_metrics": len(prediction_rows),
            "state_summary": len(summary_rows),
            "reused_oracle_actions": reused_oracle_rows,
            "evaluated_oracle_actions": evaluated_oracle_rows,
        },
        "outputs": {
            "oracle_actions": str(out_dir / "oracle_actions.tsv"),
            "oracle_groups": str(out_dir / "oracle_groups.tsv"),
            "rank_metrics": str(out_dir / "rank_metrics.tsv"),
            "prediction_metrics": str(out_dir / "prediction_metrics.tsv"),
            "state_summary": str(out_dir / "state_summary.tsv"),
            "manifest": str(manifest_path),
        },
    }
    write_json(manifest_path, manifest)
    handoff = {
        "mode": "fix",
        "status": "progressed",
        "objective": "oracle action-value and world-model reward-alignment probe",
        "out_dir": str(out_dir),
        "records": {
            "oracle_actions": len(oracle_rows),
            "oracle_groups": len(group_rows_out),
            "rank_metrics": len(rank_rows),
            "prediction_metrics": len(prediction_rows),
            "state_summary": len(summary_rows),
            "reused_oracle_actions": reused_oracle_rows,
            "evaluated_oracle_actions": evaluated_oracle_rows,
        },
        "outputs": {
            "oracle_actions": str(out_dir / "oracle_actions.tsv"),
            "oracle_groups": str(out_dir / "oracle_groups.tsv"),
            "rank_metrics": str(out_dir / "rank_metrics.tsv"),
            "prediction_metrics": str(out_dir / "prediction_metrics.tsv"),
            "state_summary": str(out_dir / "state_summary.tsv"),
            "manifest": str(manifest_path),
        },
        "resume_supported": True,
    }
    write_json(out_dir / "handoff.json", handoff)
    print(json.dumps(handoff, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
