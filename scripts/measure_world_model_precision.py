"""Measure world-model reward accuracy and exact candidate-rank accuracy."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.oracle_action_value_probe import (  # noqa: E402
    ORACLE_FIELDS,
    PRED_FIELDS,
    RANK_FIELDS,
    SCORE_FIELDS,
    SUMMARY_FIELDS,
    action_key,
    group_summary_row,
    metric_rows_for_group,
    safe_float,
    write_json,
    write_tsv,
)
from tpi_jepa.bench import parse_bench  # noqa: E402
from tpi_jepa.dataset import TPIDataset, split_by_benchmark  # noqa: E402
from tpi_jepa.evaluate_plan_tmax import evaluate_plan  # noqa: E402
from tpi_jepa.features import make_base_node_features, make_state_features  # noqa: E402
from tpi_jepa.graph import build_graph  # noqa: E402
from tpi_jepa.labels import find_bench_path, load_labels  # noqa: E402
from tpi_jepa.plan import (  # noqa: E402
    PLAN_FIELDNAMES,
    clear_planner_caches,
    enumerate_candidates,
    load_checkpoint,
    score_candidate_from_latent,
    set_real_fault_context,
)
from tpi_jepa.protocol import excluded_benchmarks_from_config, filter_rows_by_excluded_benchmarks  # noqa: E402


DEFAULT_ATALANTA = "/data4/pengqingsong/DFT/tool/atalanta_bist_with_ufaults/atalanta"
DEFAULT_TMAX = "/data3/pengqingsong/synopsys/txs/O-2018.06-SP1/bin/tmax"
REWARD_FIELDS = [
    "split",
    "samples",
    "reward_mae",
    "reward_rmse",
    "reward_corr",
    "reward_sign_acc",
    "reward_target_mean",
    "reward_pred_mean",
    "mean_baseline_mae",
    "hard_reduction_mae",
    "hard_reduction_sign_acc",
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


def parse_csv_values(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_int_values(text: str) -> list[int]:
    return [int(item) for item in parse_csv_values(text)]


def finite_pairs(xs: list[float], ys: list[float]) -> tuple[list[float], list[float]]:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if not pairs:
        return [], []
    left, right = zip(*pairs)
    return list(left), list(right)


def pearson(xs: list[float], ys: list[float]) -> float:
    xs, ys = finite_pairs(xs, ys)
    if len(xs) < 2:
        return float("nan")
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx > 0.0 and dy > 0.0 else float("nan")


def load_resume_rows(path: Path) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return {
        (
            row.get("benchmark_id", ""),
            row.get("state_id", ""),
            row.get("candidate_strategy", ""),
            row.get("action_key", ""),
        ): row
        for row in rows
    }


def plan_csv_for(path: Path, scored_row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {key: value for key, value in scored_row.items() if not key.startswith("_")}
    row["step"] = 1
    fieldnames = [field for field in PLAN_FIELDNAMES if field in row]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fieldnames})


def oracle_from_eval(rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw = rows[-1] if rows else {}
    return {
        "status": raw.get("status"),
        "oracle_delta_tc": raw.get("delta_test_coverage"),
        "oracle_delta_fault_coverage": raw.get("delta_fault_coverage"),
        "oracle_delta_pattern_count": raw.get("delta_pattern_count"),
        "oracle_test_coverage": raw.get("test_coverage"),
        "oracle_fault_coverage": raw.get("fault_coverage"),
        "oracle_hard_fault_count": raw.get("hard_fault_count"),
        "oracle_undetected_fault_count": raw.get("undetected_fault_count"),
        "oracle_error": raw.get("error"),
        "eval_dir": raw.get("eval_dir"),
    }


def write_all_outputs(
    out_dir: Path,
    oracle_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
    rank_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> None:
    write_tsv(out_dir / "oracle_actions.tsv", oracle_rows, ORACLE_FIELDS)
    write_tsv(out_dir / "oracle_groups.tsv", group_rows, GROUP_FIELDS)
    write_tsv(out_dir / "rank_metrics.tsv", rank_rows, RANK_FIELDS)
    write_tsv(out_dir / "prediction_metrics.tsv", prediction_rows, PRED_FIELDS)
    write_tsv(out_dir / "state_summary.tsv", summary_rows, SUMMARY_FIELDS)


@torch.no_grad()
def reward_accuracy(
    checkpoint: Path,
    config: dict[str, Any],
    device: torch.device,
    max_val_samples: int,
) -> dict[str, Any]:
    labels_path = config.get("labels")
    if not labels_path:
        return {"error": "checkpoint config has no labels path"}
    all_rows = load_labels(labels_path)
    rows = filter_rows_by_excluded_benchmarks(all_rows, excluded_benchmarks_from_config(config))
    _, val_rows, _ = split_by_benchmark(
        rows,
        int(config.get("seed", 1334)),
        train_frac=float(config.get("train_frac", 0.70)),
        val_frac=float(config.get("val_frac", 0.15)),
    )
    dataset = TPIDataset(
        val_rows,
        max_specs=max_val_samples,
        max_nodes=int(config.get("max_nodes", 0)) or None,
        feature_mode=str(config.get("feature_mode", "basic")),
        relation_mode=str(config.get("relation_mode", "basic")),
        relation_depth=int(config.get("relation_depth", 8)),
        real_fault_prior_path=config.get("real_fault_priors") or config.get("real_fault_prior_path"),
        activation_prior_path=config.get("activation_priors") or config.get("activation_prior_path"),
        cache_samples=False,
    )
    model, ckpt_config = load_checkpoint(checkpoint, device)
    coverage_scale = float(ckpt_config.get("coverage_scale", getattr(model, "coverage_scale", 100.0)))
    reward_preds: list[float] = []
    reward_targets: list[float] = []
    hard_pred: list[float] = []
    hard_target: list[float] = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        out = model(
            sample.graph,
            sample.x_pre.to(device),
            sample.x_post.to(device),
            sample.action_node_id,
            sample.action_type_id,
            sample.relation_features.to(device),
        )
        reward_preds.append(float(out["reward_pred"].detach().cpu().item()))
        reward_targets.append(float((coverage_scale * sample.delta_fault_coverage).detach().cpu().item()))
        hp = out["hard_reduction_pred"].detach().cpu().view(-1)
        ht = sample.hard_reduction_target.detach().cpu().view(-1)
        hard_pred.extend(float(x) for x in hp.tolist())
        hard_target.extend(float(x) for x in ht.tolist())
    if not reward_preds:
        return {"error": "no validation samples"}
    abs_err = [abs(p - t) for p, t in zip(reward_preds, reward_targets)]
    sq_err = [(p - t) ** 2 for p, t in zip(reward_preds, reward_targets)]
    baseline = sum(reward_targets) / len(reward_targets)
    hard_abs = [abs(p - t) for p, t in zip(hard_pred, hard_target)]
    hard_sign = [
        1.0 if (p >= 0.0) == (t >= 0.0) else 0.0
        for p, t in zip(hard_pred, hard_target)
        if math.isfinite(p) and math.isfinite(t)
    ]
    return {
        "split": "val",
        "samples": len(reward_preds),
        "reward_mae": sum(abs_err) / len(abs_err),
        "reward_rmse": math.sqrt(sum(sq_err) / len(sq_err)),
        "reward_corr": pearson(reward_preds, reward_targets),
        "reward_sign_acc": sum(
            1.0 if (p >= 0.0) == (t >= 0.0) else 0.0 for p, t in zip(reward_preds, reward_targets)
        )
        / len(reward_preds),
        "reward_target_mean": sum(reward_targets) / len(reward_targets),
        "reward_pred_mean": sum(reward_preds) / len(reward_preds),
        "mean_baseline_mae": sum(abs(baseline - t) for t in reward_targets) / len(reward_targets),
        "hard_reduction_mae": sum(hard_abs) / len(hard_abs) if hard_abs else float("nan"),
        "hard_reduction_sign_acc": sum(hard_sign) / len(hard_sign) if hard_sign else float("nan"),
    }


@torch.no_grad()
def score_exact_candidates(
    *,
    model,
    config: dict[str, Any],
    graph,
    candidates: list[tuple[str, str]],
    device: torch.device,
    diversity_penalty: float,
    diversity_depth: int,
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
    selected: list[tuple[str, str]] = []
    x_state = make_state_features(graph, selected, base_features).to(device)
    z_state = model.online_encoder(
        x_state,
        graph.edge_src.to(device),
        graph.edge_dst.to(device),
        graph.gate_type_ids.to(device),
    )
    rows: dict[str, dict[str, Any]] = {}
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
            diversity_penalty,
            diversity_depth,
        )
        row.pop("_z_pred", None)
        rows[action_key(row["node"], row["type"])] = row
    return rows


def run_exact_rank_probe(args: argparse.Namespace, model, config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    score_fields = parse_csv_values(args.score_fields)
    ks = parse_int_values(args.top_ks)
    resume_rows = load_resume_rows(out_dir / "oracle_actions.tsv") if args.resume else {}
    oracle_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    reused = 0
    evaluated = 0

    for benchmark_id in parse_csv_values(args.benchmarks):
        for strategy in parse_csv_values(args.candidate_strategies):
            clear_planner_caches()
            set_real_fault_context(benchmark_id, args.real_fault_priors, args.activation_priors)
            graph = build_graph(parse_bench(find_bench_path(benchmark_id)))
            candidates = enumerate_candidates(
                graph,
                [],
                args.max_candidates,
                strategy,
                real_fault_benchmark_id=benchmark_id,
                real_fault_prior_path=args.real_fault_priors,
                activation_prior_path=args.activation_priors,
                candidate_cache_dir=args.candidate_cache_dir,
                candidate_sample_seed=args.candidate_sample_seed,
            )
            scored = score_exact_candidates(
                model=model,
                config=config,
                graph=graph,
                candidates=candidates,
                device=device,
                diversity_penalty=args.candidate_diversity_penalty,
                diversity_depth=args.candidate_diversity_depth,
            )
            group = []
            for rank, candidate in enumerate(candidates, start=1):
                key = action_key(*candidate)
                row = scored[key]
                previous = resume_rows.get((benchmark_id, "initial", strategy, key))
                if previous is not None:
                    oracle = {field: previous.get(field, "") for field in ORACLE_FIELDS if field.startswith("oracle_")}
                    oracle["status"] = previous.get("status", "")
                    oracle["eval_dir"] = previous.get("eval_dir", "")
                    reused += 1
                else:
                    action_dir = out_dir / "evals" / benchmark_id / strategy / key.replace("/", "_").replace("::", "__")
                    plan_csv = action_dir / "plan.csv"
                    plan_csv_for(plan_csv, row)
                    raw_rows = evaluate_plan(
                        benchmark_id=benchmark_id,
                        plan_csv=plan_csv,
                        out_dir=action_dir,
                        patterns=args.patterns,
                        seed=args.seed,
                        backend=args.backend,
                        tmax_bin=args.tmax_bin,
                        atalanta_bin=args.atalanta_bin,
                        timeout_sec=args.timeout_sec,
                        force=True,
                        dry_run=args.dry_run,
                        cleanup_workdir=args.cleanup_workdir,
                        eval_step_mode="final",
                    )
                    oracle = oracle_from_eval(raw_rows)
                    evaluated += 1
                full_row = {
                    "benchmark_id": benchmark_id,
                    "state_id": "initial",
                    "candidate_strategy": strategy,
                    "candidate_rank": rank,
                    "node": candidate[0],
                    "type": candidate[1],
                    "action_key": key,
                    **oracle,
                    **{field: row.get(field) for field in ORACLE_FIELDS if field in row},
                }
                oracle_rows.append(full_row)
                group.append(full_row)
                group_rows_tmp = group_rows + [group_summary_row(group)]
                group_rank, group_pred, group_summary = metric_rows_for_group(
                    rows=group,
                    score_fields=score_fields,
                    ks=ks,
                    oracle_top_m=args.oracle_top_m,
                )
                write_all_outputs(
                    out_dir,
                    oracle_rows,
                    group_rows_tmp,
                    rank_rows + group_rank,
                    prediction_rows + group_pred,
                    summary_rows + group_summary,
                )
                print(
                    f"[rank-probe] {benchmark_id} {strategy} {rank}/{len(candidates)} "
                    f"node={candidate[0]} type={candidate[1]} reused={int(previous is not None)}"
                )
            group_rows.append(group_summary_row(group))
            group_rank, group_pred, group_summary = metric_rows_for_group(
                rows=group,
                score_fields=score_fields,
                ks=ks,
                oracle_top_m=args.oracle_top_m,
            )
            rank_rows.extend(group_rank)
            prediction_rows.extend(group_pred)
            summary_rows.extend(group_summary)
            write_all_outputs(out_dir, oracle_rows, group_rows, rank_rows, prediction_rows, summary_rows)
    return {
        "oracle_actions": len(oracle_rows),
        "oracle_groups": len(group_rows),
        "rank_metrics": len(rank_rows),
        "prediction_metrics": len(prediction_rows),
        "state_summary": len(summary_rows),
        "reused_oracle_actions": reused,
        "evaluated_oracle_actions": evaluated,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--benchmarks", required=True)
    parser.add_argument("--candidate-strategies", default="heuristic_recall_pool")
    parser.add_argument("--max-candidates", type=int, default=96)
    parser.add_argument("--score-fields", default="reward_pred")
    parser.add_argument("--top-ks", default="1,8,16,32,48,96")
    parser.add_argument("--oracle-top-m", type=int, default=5)
    parser.add_argument("--patterns", type=int, default=300000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--backend", choices=["tmax", "atalanta-bist"], default="atalanta-bist")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tmax-bin", default=DEFAULT_TMAX)
    parser.add_argument("--atalanta-bin", default=DEFAULT_ATALANTA)
    parser.add_argument("--timeout-sec", type=int, default=14400)
    parser.add_argument("--real-fault-priors", default=None)
    parser.add_argument("--activation-priors", default=None)
    parser.add_argument("--candidate-cache-dir", default=None)
    parser.add_argument("--candidate-sample-seed", type=int, default=0)
    parser.add_argument("--candidate-diversity-penalty", type=float, default=0.0)
    parser.add_argument("--candidate-diversity-depth", type=int, default=4)
    parser.add_argument("--max-val-samples", type=int, default=2048)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cleanup-workdir", action="store_true")
    parser.add_argument("--skip-reward-accuracy", action="store_true")
    parser.add_argument("--skip-rank-probe", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, config = load_checkpoint(args.checkpoint, device)

    reward_row = None
    if not args.skip_reward_accuracy:
        reward_row = reward_accuracy(args.checkpoint, config, device, args.max_val_samples)
        write_tsv(out_dir / "reward_accuracy.tsv", [reward_row], REWARD_FIELDS)

    records = {}
    if not args.skip_rank_probe:
        records = run_exact_rank_probe(args, model, config, device)

    manifest = {
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(args.checkpoint),
        "benchmarks": parse_csv_values(args.benchmarks),
        "candidate_strategies": parse_csv_values(args.candidate_strategies),
        "max_candidates": args.max_candidates,
        "score_fields": parse_csv_values(args.score_fields),
        "top_ks": parse_int_values(args.top_ks),
        "oracle_top_m": args.oracle_top_m,
        "patterns": args.patterns,
        "backend": args.backend,
        "real_fault_priors": args.real_fault_priors,
        "activation_priors": args.activation_priors,
        "reward_accuracy": reward_row,
        "records": records,
        "outputs": {
            "reward_accuracy": str(out_dir / "reward_accuracy.tsv"),
            "oracle_actions": str(out_dir / "oracle_actions.tsv"),
            "oracle_groups": str(out_dir / "oracle_groups.tsv"),
            "rank_metrics": str(out_dir / "rank_metrics.tsv"),
            "prediction_metrics": str(out_dir / "prediction_metrics.tsv"),
            "state_summary": str(out_dir / "state_summary.tsv"),
        },
    }
    write_json(out_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
