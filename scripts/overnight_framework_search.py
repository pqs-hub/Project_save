"""Run unattended framework-level training and TMAX evaluation experiments."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tpi_jepa.protocol import eval_benchmarks_from_protocol, parse_benchmark_list


RESULT_FIELDS = [
    "timestamp",
    "status",
    "variant",
    "framework_variant",
    "framework_components",
    "framework_hypothesis",
    "benchmark_id",
    "feature_mode",
    "relation_mode",
    "edge_weight_mode",
    "edge_keep_ratio",
    "residual_dynamics",
    "relation_gate",
    "candidate_strategy",
    "candidate_diversity_penalty",
    "hard_sample_weight",
    "head_context",
    "latent_dim",
    "encoder_layers",
    "dropout",
    "lambda_score",
    "lambda_return",
    "score_fc_scale",
    "score_pattern_weight",
    "rollout_max_horizon",
    "rollout_start_epoch",
    "epochs",
    "max_train_samples",
    "max_train_steps_per_epoch",
    "best_val_loss",
    "last_val_loss",
    "plan_score_sum",
    "plan_fc_sum",
    "plan_return_sum",
    "plan_score_field",
    "plan_beam_objective",
    "delta_test_coverage",
    "delta_fault_coverage",
    "delta_pattern_count",
    "config_path",
    "checkpoint",
    "plan_csv",
    "eval_dir",
    "excluded_benchmarks",
    "elapsed_sec",
    "error",
]


def parse_csv_values(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_int_values(text: str) -> list[int]:
    return [int(item) for item in parse_csv_values(text)]


def parse_float_values(text: str) -> list[float]:
    return [float(item) for item in parse_csv_values(text)]


def parse_bool_values(text: str) -> list[bool]:
    values = []
    for item in parse_csv_values(text):
        values.append(item.lower() in {"1", "true", "yes", "y"})
    return values


def framework_variant_settings(name: str) -> dict:
    """Map a named testability framework ablation to train/planner settings."""

    variants = {
        "baseline": {
            "components": ["baseline_features", "netlist_candidates"],
            "hypothesis": "Original netlist-order candidates provide a leakage-free reference point.",
            "feature_mode": "basic",
            "relation_mode": "basic",
            "edge_weight_mode": "mean",
            "edge_keep_ratio": 1.0,
            "residual_dynamics": False,
            "relation_gate": False,
            "candidate_strategy": "netlist",
            "candidate_diversity_penalty": 0.0,
            "candidate_diversity_depth": 4,
            "hard_sample_weight": 0.0,
        },
        "region": {
            "components": ["region_features", "testability_candidates"],
            "hypothesis": "Hard-region, FFR, reconvergence, and transparent-chain features improve local action valuation.",
            "feature_mode": "region",
            "relation_mode": "basic",
            "edge_weight_mode": "mean",
            "edge_keep_ratio": 1.0,
            "residual_dynamics": False,
            "relation_gate": False,
            "candidate_strategy": "testability",
            "candidate_diversity_penalty": 0.0,
            "candidate_diversity_depth": 4,
            "hard_sample_weight": 0.0,
        },
        "cone": {
            "components": ["action_conditioned_cone_relation", "testability_candidates"],
            "hypothesis": "Action-conditioned cone relation features help the world model focus on the local impact area.",
            "feature_mode": "basic",
            "relation_mode": "cone",
            "edge_weight_mode": "mean",
            "edge_keep_ratio": 1.0,
            "residual_dynamics": True,
            "relation_gate": True,
            "candidate_strategy": "testability",
            "candidate_diversity_penalty": 0.0,
            "candidate_diversity_depth": 4,
            "hard_sample_weight": 0.0,
        },
        "sparse": {
            "components": ["fault_path_sparse_message_passing", "testability_candidates"],
            "hypothesis": "Fault-path weighted sparse message passing reduces irrelevant propagation noise.",
            "feature_mode": "basic",
            "relation_mode": "basic",
            "edge_weight_mode": "fault_path",
            "edge_keep_ratio": 0.80,
            "residual_dynamics": False,
            "relation_gate": False,
            "candidate_strategy": "testability",
            "candidate_diversity_penalty": 0.0,
            "candidate_diversity_depth": 4,
            "hard_sample_weight": 0.0,
        },
        "sparse_075": {
            "components": ["fault_path_sparse_message_passing_keep075", "testability_candidates"],
            "hypothesis": "A slightly more selective sparse message-passing graph may remove more irrelevant paths.",
            "feature_mode": "basic",
            "relation_mode": "basic",
            "edge_weight_mode": "fault_path",
            "edge_keep_ratio": 0.75,
            "residual_dynamics": False,
            "relation_gate": False,
            "candidate_strategy": "testability",
            "candidate_diversity_penalty": 0.0,
            "candidate_diversity_depth": 4,
            "hard_sample_weight": 0.0,
        },
        "sparse_080": {
            "components": ["fault_path_sparse_message_passing_keep080", "testability_candidates"],
            "hypothesis": "Replicates the v3 sparse setting in the v4 sweep for same-run comparison.",
            "feature_mode": "basic",
            "relation_mode": "basic",
            "edge_weight_mode": "fault_path",
            "edge_keep_ratio": 0.80,
            "residual_dynamics": False,
            "relation_gate": False,
            "candidate_strategy": "testability",
            "candidate_diversity_penalty": 0.0,
            "candidate_diversity_depth": 4,
            "hard_sample_weight": 0.0,
        },
        "sparse_085": {
            "components": ["fault_path_sparse_message_passing_keep085", "testability_candidates"],
            "hypothesis": "A softer sparse message-passing graph may preserve useful secondary fault paths.",
            "feature_mode": "basic",
            "relation_mode": "basic",
            "edge_weight_mode": "fault_path",
            "edge_keep_ratio": 0.85,
            "residual_dynamics": False,
            "relation_gate": False,
            "candidate_strategy": "testability",
            "candidate_diversity_penalty": 0.0,
            "candidate_diversity_depth": 4,
            "hard_sample_weight": 0.0,
        },
        "hard_fault": {
            "components": ["region_features", "hard_fault_candidate_mining"],
            "hypothesis": "Prioritizing hard-fault regions improves candidate quality under a small action budget.",
            "feature_mode": "region",
            "relation_mode": "basic",
            "edge_weight_mode": "mean",
            "edge_keep_ratio": 1.0,
            "residual_dynamics": False,
            "relation_gate": False,
            "candidate_strategy": "hard_fault",
            "candidate_diversity_penalty": 0.0,
            "candidate_diversity_depth": 4,
            "hard_sample_weight": 0.0,
        },
        "hard_fault_cone": {
            "components": [
                "region_features",
                "hard_fault_activation_cones",
                "hard_fault_propagation_paths",
                "observability_bottlenecks",
                "shared_hard_fault_coverage",
            ],
            "hypothesis": "Explicit hard-fault cone and path coverage is a stronger planner prior than isolated hard-fault site scores.",
            "feature_mode": "region",
            "relation_mode": "basic",
            "edge_weight_mode": "mean",
            "edge_keep_ratio": 1.0,
            "residual_dynamics": False,
            "relation_gate": False,
            "candidate_strategy": "hard_fault_cone",
            "candidate_diversity_penalty": 0.0,
            "candidate_diversity_depth": 4,
            "hard_sample_weight": 0.0,
        },
        "reconvergence": {
            "components": ["region_features", "action_conditioned_cone_relation", "reconvergence_candidates"],
            "hypothesis": "Explicit reconvergence-aware candidates help address fault propagation masking.",
            "feature_mode": "region",
            "relation_mode": "cone",
            "edge_weight_mode": "mean",
            "edge_keep_ratio": 1.0,
            "residual_dynamics": True,
            "relation_gate": True,
            "candidate_strategy": "reconvergence",
            "candidate_diversity_penalty": 0.0,
            "candidate_diversity_depth": 4,
            "hard_sample_weight": 0.0,
        },
        "ffr": {
            "components": ["region_features", "ffr_candidates"],
            "hypothesis": "FFR-span-aware candidates capture reusable medium-scale testability structures.",
            "feature_mode": "region",
            "relation_mode": "basic",
            "edge_weight_mode": "mean",
            "edge_keep_ratio": 1.0,
            "residual_dynamics": False,
            "relation_gate": False,
            "candidate_strategy": "ffr",
            "candidate_diversity_penalty": 0.0,
            "candidate_diversity_depth": 4,
            "hard_sample_weight": 0.0,
        },
        "ffr_hier": {
            "components": ["region_features", "ffr_hierarchical_candidates"],
            "hypothesis": "Selecting FFR-like regions first and gates second reduces candidate collapse inside one chain.",
            "feature_mode": "region",
            "relation_mode": "basic",
            "edge_weight_mode": "mean",
            "edge_keep_ratio": 1.0,
            "residual_dynamics": False,
            "relation_gate": False,
            "candidate_strategy": "ffr_hier",
            "candidate_diversity_penalty": 0.0,
            "candidate_diversity_depth": 4,
            "hard_sample_weight": 0.0,
        },
        "sparse_ffr": {
            "components": ["region_features", "fault_path_sparse_message_passing", "ffr_hierarchical_candidates"],
            "hypothesis": "Combining sparse propagation with FFR-level candidate selection tests whether the two strongest v3 ideas compose.",
            "feature_mode": "region",
            "relation_mode": "basic",
            "edge_weight_mode": "fault_path",
            "edge_keep_ratio": 0.80,
            "residual_dynamics": False,
            "relation_gate": False,
            "candidate_strategy": "ffr_hier",
            "candidate_diversity_penalty": 0.0,
            "candidate_diversity_depth": 4,
            "hard_sample_weight": 0.0,
        },
        "hard_weighted_sparse": {
            "components": ["region_features", "fault_path_sparse_message_passing", "hard_region_weighted_training"],
            "hypothesis": "Hard-fault information may help as a training weight rather than as a standalone candidate ranking.",
            "feature_mode": "region",
            "relation_mode": "basic",
            "edge_weight_mode": "fault_path",
            "edge_keep_ratio": 0.80,
            "residual_dynamics": False,
            "relation_gate": False,
            "candidate_strategy": "testability",
            "candidate_diversity_penalty": 0.0,
            "candidate_diversity_depth": 4,
            "hard_sample_weight": 1.5,
        },
        "mixed": {
            "components": ["region_features", "action_conditioned_cone_relation", "mixed_candidates"],
            "hypothesis": "Combining region features, cone relation, and mixed candidate scoring gives broader coverage.",
            "feature_mode": "region",
            "relation_mode": "cone",
            "edge_weight_mode": "mean",
            "edge_keep_ratio": 1.0,
            "residual_dynamics": True,
            "relation_gate": True,
            "candidate_strategy": "mixed",
            "candidate_diversity_penalty": 0.0,
            "candidate_diversity_depth": 4,
            "hard_sample_weight": 0.0,
        },
        "diversity": {
            "components": ["region_features", "action_conditioned_cone_relation", "mixed_candidates", "region_diversity_penalty"],
            "hypothesis": "A local diversity penalty avoids redundant test points in the same cone or region.",
            "feature_mode": "region",
            "relation_mode": "cone",
            "edge_weight_mode": "mean",
            "edge_keep_ratio": 1.0,
            "residual_dynamics": True,
            "relation_gate": True,
            "candidate_strategy": "mixed",
            "candidate_diversity_penalty": 0.05,
            "candidate_diversity_depth": 8,
            "hard_sample_weight": 0.0,
        },
        "full_tac": {
            "components": [
                "region_features",
                "action_conditioned_cone_relation",
                "fault_path_sparse_message_passing",
                "mixed_candidates",
                "region_diversity_penalty",
            ],
            "hypothesis": "The full testability-aware stack should best balance scalability, relevance, and diversity.",
            "feature_mode": "region",
            "relation_mode": "cone",
            "edge_weight_mode": "fault_path",
            "edge_keep_ratio": 0.80,
            "residual_dynamics": True,
            "relation_gate": True,
            "candidate_strategy": "mixed",
            "candidate_diversity_penalty": 0.05,
            "candidate_diversity_depth": 8,
            "hard_sample_weight": 0.0,
        },
    }
    key = name.strip().lower()
    if key not in variants:
        raise ValueError(f"unknown framework variant {name!r}; choices={','.join(sorted(variants))}")
    return {"framework_variant": key, **variants[key]}


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sanitize(value: str) -> str:
    return value.replace("/", "_").replace(":", "_").replace(".", "p")


def parse_devices(text: str | None, fallback: str) -> list[str]:
    """Parse worker device assignments."""

    raw = parse_csv_values(text or fallback)
    devices = raw or [fallback]
    normalized = []
    for device in devices:
        normalized.append(f"cuda:{device}" if device.isdigit() else device)
    return normalized


def run_command(cmd: list[str], log_path: Path, env: dict[str, str] | None = None) -> tuple[bool, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True, env=env)
    if result.returncode == 0:
        return True, ""
    try:
        tail = "".join(log_path.read_text(errors="replace").splitlines(True)[-60:])
    except OSError:
        tail = ""
    return False, tail.strip()


def history_metrics(path: Path) -> tuple[str, str]:
    rows = list(csv.DictReader(path.open()))
    if not rows:
        return "NA", "NA"
    best = min(float(row["val_loss"]) for row in rows)
    return str(best), rows[-1]["val_loss"]


def plan_sums(path: Path) -> tuple[float, float, float]:
    rows = list(csv.DictReader(path.open()))
    return (
        sum(float(row.get("score_pred") or 0.0) for row in rows),
        sum(float(row.get("fc_pred") or 0.0) for row in rows),
        sum(float(row.get("return_pred") or 0.0) for row in rows),
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
    value = row.get("delta_test_coverage")
    if value in ("", None, "NA"):
        return
    best_path = out_dir / "best.json"
    current = json.loads(best_path.read_text()) if best_path.exists() else None
    value_float = float(value)
    if current is not None and value_float <= float(current.get("delta_test_coverage", "-inf")):
        return
    best_path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    plan_csv = Path(row["plan_csv"])
    if plan_csv.exists():
        shutil.copy2(plan_csv, out_dir / "best_plan.csv")
    checkpoint = Path(row["checkpoint"])
    if checkpoint.exists():
        shutil.copy2(checkpoint, out_dir / "best.pt")


def make_variant_name(parts: dict) -> str:
    tokens = [
        f"fw{parts['framework_variant']}",
        "ctx" + ("1" if parts["head_context"] else "0"),
        f"ld{parts['latent_dim']}",
        f"el{parts['encoder_layers']}",
        f"do{sanitize(str(parts['dropout']))}",
        f"lrtn{sanitize(str(parts['lambda_return']))}",
        f"h{parts['rollout_max_horizon']}",
    ]
    return "__".join(tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description="Overnight framework search for TPI-JEPA.")
    parser.add_argument("--base-config", default="configs/tmax50k_curriculum_full5.json")
    parser.add_argument("--labels", default=None)
    parser.add_argument("--eval-protocol", default="configs/eval_protocol_coverage_only.json")
    parser.add_argument("--real-fault-priors", default=None)
    parser.add_argument("--activation-priors", default=None)
    parser.add_argument(
        "--framework-variants",
        default="baseline,sparse_075,sparse_080,sparse_085,ffr_hier,sparse_ffr,hard_weighted_sparse",
    )
    parser.add_argument(
        "--allow-eval-benchmarks-in-train",
        action="store_true",
        help="Disable the default train-set exclusion for evaluation benchmarks.",
    )
    parser.add_argument("--benchmark-id", default="iscas89__s838")
    parser.add_argument("--head-context", default="false,true")
    parser.add_argument("--latent-dims", default="64")
    parser.add_argument("--encoder-layers", default="3")
    parser.add_argument("--dropouts", default="0.1")
    parser.add_argument("--lambda-returns", default="0.0")
    parser.add_argument("--rollout-horizons", default="3,5")
    parser.add_argument("--rollout-start-epochs", default="2,4")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--max-train-samples", type=int, default=12000)
    parser.add_argument("--max-train-steps-per-epoch", type=int, default=400)
    parser.add_argument("--max-val-samples", type=int, default=1024)
    parser.add_argument("--max-val-steps", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--plan-budget", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--lookahead-depth", type=int, default=3)
    parser.add_argument("--plan-score-field", default="reward_pred")
    parser.add_argument("--plan-beam-objective", default="cumulative")
    parser.add_argument("--patterns", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--timeout-sec", type=int, default=600)
    parser.add_argument("--eval-backend", choices=["tmax", "atalanta-bist"], default="tmax")
    parser.add_argument(
        "--atalanta-bin",
        default="/data3/pengqingsong/DFT/DeepTPI-project/external/DeepTPI/src/external/Atalanta_BIST/atalanta",
    )
    parser.add_argument("--time-limit-hours", type=float, default=8.0)
    parser.add_argument("--parallel-jobs", type=int, default=7)
    parser.add_argument("--devices", default="0,1,2,3,4,5,6", help="Comma-separated worker devices, e.g. cuda:0,cuda:1 or 0,1.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    started = time.time()
    stamp = datetime.now().strftime("%y%m%d-%H%M%S")
    out_dir = Path(args.out_dir or f"autoresearch/framework-{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.tsv"
    (out_dir / "runner_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n")
    base = json.loads(Path(args.base_config).read_text())
    deadline = started + args.time_limit_hours * 3600.0
    excluded_benchmarks: set[str] = set()
    if not args.allow_eval_benchmarks_in_train:
        excluded_benchmarks.update(eval_benchmarks_from_protocol(args.eval_protocol))
        excluded_benchmarks.update(parse_benchmark_list(args.benchmark_id))
    (out_dir / "excluded_benchmarks.json").write_text(
        json.dumps(sorted(excluded_benchmarks), indent=2, sort_keys=True) + "\n"
    )

    variants = itertools.product(
        parse_csv_values(args.framework_variants),
        parse_bool_values(args.head_context),
        parse_int_values(args.latent_dims),
        parse_int_values(args.encoder_layers),
        parse_float_values(args.dropouts),
        parse_float_values(args.lambda_returns),
        parse_int_values(args.rollout_horizons),
        parse_int_values(args.rollout_start_epochs),
    )

    variant_specs = list(variants)
    variant_manifest = []
    for spec in variant_specs:
        framework = framework_variant_settings(spec[0])
        variant_manifest.append(
            {
                "framework_variant": framework["framework_variant"],
                "components": framework["components"],
                "hypothesis": framework["hypothesis"],
                "head_context": spec[1],
                "latent_dim": spec[2],
                "encoder_layers": spec[3],
                "dropout": spec[4],
                "lambda_return": spec[5],
                "rollout_max_horizon": spec[6],
                "rollout_start_epoch": spec[7],
                "feature_mode": framework["feature_mode"],
                "relation_mode": framework["relation_mode"],
                "edge_weight_mode": framework["edge_weight_mode"],
                "edge_keep_ratio": framework["edge_keep_ratio"],
                "residual_dynamics": framework["residual_dynamics"],
                "relation_gate": framework["relation_gate"],
                "candidate_strategy": framework["candidate_strategy"],
                "candidate_diversity_penalty": framework["candidate_diversity_penalty"],
                "candidate_diversity_depth": framework["candidate_diversity_depth"],
                "hard_sample_weight": framework["hard_sample_weight"],
            }
        )
    write_json(out_dir / "variant_manifest.json", variant_manifest)
    result_lock = threading.Lock()
    best_lock = threading.Lock()
    devices = parse_devices(args.devices, args.device)
    max_workers = max(1, int(args.parallel_jobs))

    def run_one(index: int, spec: tuple) -> dict | None:
        (
            framework_variant,
            head_context,
            latent_dim,
            encoder_layers,
            dropout,
            lambda_return,
            rollout_max_horizon,
            rollout_start_epoch,
        ) = spec
        framework = framework_variant_settings(framework_variant)
        if time.time() >= deadline:
            return None
        iter_started = time.time()
        worker_device = devices[index % len(devices)]
        parts = {
            "framework_variant": framework["framework_variant"],
            "head_context": head_context,
            "latent_dim": latent_dim,
            "encoder_layers": encoder_layers,
            "dropout": dropout,
            "lambda_return": lambda_return,
            "rollout_max_horizon": rollout_max_horizon,
        }
        variant = make_variant_name(parts) + f"__rs{rollout_start_epoch}"
        run_dir = out_dir / "runs" / variant
        config_path = out_dir / "configs" / f"{variant}.json"
        config = dict(base)
        if args.labels:
            config["labels"] = args.labels
        if args.real_fault_priors:
            config["real_fault_priors"] = args.real_fault_priors
        if args.activation_priors:
            config["activation_priors"] = args.activation_priors
        merged_excluded = parse_benchmark_list(config.get("exclude_benchmarks")) | excluded_benchmarks
        config.update(
            {
                "run_dir": str(run_dir),
                "epochs": args.epochs,
                "max_train_samples": args.max_train_samples,
                "max_train_steps_per_epoch": args.max_train_steps_per_epoch,
                "max_val_samples": args.max_val_samples,
                "max_val_steps": args.max_val_steps,
                "latent_dim": latent_dim,
                "encoder_layers": encoder_layers,
                "dropout": dropout,
                "lambda_score": 0.0,
                "lambda_return": lambda_return,
                "lambda_pattern": 0.0,
                "score_pattern_weight": 0.0,
                "head_context": head_context,
                "rollout_max_horizon": rollout_max_horizon,
                "rollout_start_epoch": rollout_start_epoch,
                "device": worker_device,
                "exclude_benchmarks": sorted(merged_excluded),
                "exclude_eval_protocol": args.eval_protocol if args.eval_protocol else "",
                "feature_mode": framework["feature_mode"],
                "relation_mode": framework["relation_mode"],
                "edge_weight_mode": framework["edge_weight_mode"],
                "edge_keep_ratio": framework["edge_keep_ratio"],
                "residual_dynamics": framework["residual_dynamics"],
                "relation_gate": framework["relation_gate"],
                "candidate_strategy": framework["candidate_strategy"],
                "candidate_diversity_penalty": framework["candidate_diversity_penalty"],
                "candidate_diversity_depth": framework["candidate_diversity_depth"],
                "hard_sample_weight": framework["hard_sample_weight"],
            }
        )
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "status": "started",
            "variant": variant,
            "framework_variant": framework["framework_variant"],
            "framework_components": ",".join(framework["components"]),
            "framework_hypothesis": framework["hypothesis"],
            "benchmark_id": args.benchmark_id,
            "feature_mode": framework["feature_mode"],
            "relation_mode": framework["relation_mode"],
            "edge_weight_mode": framework["edge_weight_mode"],
            "edge_keep_ratio": framework["edge_keep_ratio"],
            "residual_dynamics": framework["residual_dynamics"],
            "relation_gate": framework["relation_gate"],
            "candidate_strategy": framework["candidate_strategy"],
            "candidate_diversity_penalty": framework["candidate_diversity_penalty"],
            "hard_sample_weight": framework["hard_sample_weight"],
            "head_context": head_context,
            "latent_dim": latent_dim,
            "encoder_layers": encoder_layers,
            "dropout": dropout,
            "lambda_score": 0.0,
            "lambda_return": lambda_return,
            "score_fc_scale": "",
            "score_pattern_weight": 0.0,
            "rollout_max_horizon": rollout_max_horizon,
            "rollout_start_epoch": rollout_start_epoch,
            "epochs": args.epochs,
            "max_train_samples": args.max_train_samples,
            "max_train_steps_per_epoch": args.max_train_steps_per_epoch,
            "plan_score_field": args.plan_score_field,
            "plan_beam_objective": args.plan_beam_objective,
            "config_path": str(config_path),
            "excluded_benchmarks": ",".join(sorted(merged_excluded)),
        }
        if args.dry_run:
            row.update({"status": "dry_run", "elapsed_sec": round(time.time() - iter_started, 3)})
            with result_lock:
                append_result(results_path, row)
            return row

        env = dict(os.environ)
        if worker_device.startswith("cuda:"):
            env["CUDA_VISIBLE_DEVICES"] = worker_device.split(":", 1)[1]
            config["device"] = "cuda"
            config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

        train_log = out_dir / "logs" / f"{variant}.train.log"
        ok, error = run_command(["python", "-m", "tpi_jepa.train", "--config", str(config_path)], train_log, env=env)
        if not ok:
            row.update({"status": "train_error", "error": error, "elapsed_sec": round(time.time() - iter_started, 3)})
            with result_lock:
                append_result(results_path, row)
            return row

        checkpoint = run_dir / "best.pt"
        row["checkpoint"] = str(checkpoint)
        best_val, last_val = history_metrics(run_dir / "history.csv")
        row.update({"best_val_loss": best_val, "last_val_loss": last_val})
        plan_csv = out_dir / "plans" / f"{variant}.csv"
        plan_log = out_dir / "logs" / f"{variant}.plan.log"
        plan_csv.parent.mkdir(parents=True, exist_ok=True)
        plan_cmd = [
            "python",
            "-m",
            "tpi_jepa.plan",
            "--checkpoint",
            str(checkpoint),
            "--benchmark-id",
            args.benchmark_id,
            "--budget",
            str(args.plan_budget),
            "--max-candidates",
            str(args.max_candidates),
            "--device",
            "cpu",
            "--planner",
            "beam",
            "--beam-width",
            str(args.beam_width),
            "--lookahead-depth",
            str(args.lookahead_depth),
            "--score-field",
            args.plan_score_field,
            "--beam-objective",
            args.plan_beam_objective,
            "--feature-mode",
            framework["feature_mode"],
            "--relation-mode",
            framework["relation_mode"],
            "--candidate-strategy",
            framework["candidate_strategy"],
            "--candidate-diversity-penalty",
            str(framework["candidate_diversity_penalty"]),
            "--candidate-diversity-depth",
            str(framework["candidate_diversity_depth"]),
            "--out",
            str(plan_csv),
        ]
        if args.real_fault_priors:
            plan_cmd.extend(["--real-fault-priors", args.real_fault_priors])
        if args.activation_priors:
            plan_cmd.extend(["--activation-priors", args.activation_priors])
        plan_env = dict(os.environ)
        plan_env.setdefault("OMP_NUM_THREADS", "1")
        plan_env.setdefault("MKL_NUM_THREADS", "1")
        plan_env.setdefault("OPENBLAS_NUM_THREADS", "1")
        ok, error = run_command(plan_cmd, plan_log, env=plan_env)
        if not ok:
            row.update({"status": "plan_error", "error": error, "elapsed_sec": round(time.time() - iter_started, 3)})
            with result_lock:
                append_result(results_path, row)
            return row

        score_sum, fc_sum, return_sum = plan_sums(plan_csv)
        row.update(
            {
                "plan_csv": str(plan_csv),
                "plan_score_sum": score_sum,
                "plan_fc_sum": fc_sum,
                "plan_return_sum": return_sum,
            }
        )
        eval_dir = out_dir / "evals" / variant
        eval_log = out_dir / "logs" / f"{variant}.eval.log"
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_cmd = [
            "python",
            "-m",
            "tpi_jepa.evaluate_plan_tmax",
            "--benchmark-id",
            args.benchmark_id,
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
            "--cleanup-workdir",
        ]
        if args.eval_backend == "atalanta-bist":
            eval_cmd.extend(["--atalanta-bin", args.atalanta_bin])
        if args.dry_run:
            eval_cmd.append("--dry-run")
        ok, error = run_command(eval_cmd, eval_log, env=plan_env)
        if not ok:
            row.update({"status": "eval_error", "error": error, "elapsed_sec": round(time.time() - iter_started, 3)})
            with result_lock:
                append_result(results_path, row)
            return row

        metrics = final_eval_metrics(eval_dir / "labels.csv")
        row.update(
            {
                "status": metrics.get("status") or "ok",
                "delta_test_coverage": metrics.get("delta_test_coverage") or "NA",
                "delta_fault_coverage": metrics.get("delta_fault_coverage") or "NA",
                "delta_pattern_count": metrics.get("delta_pattern_count") or "NA",
                "eval_dir": str(eval_dir),
                "elapsed_sec": round(time.time() - iter_started, 3),
            }
        )
        with result_lock:
            append_result(results_path, row)
        with best_lock:
            update_best(out_dir, row)
        return row

    if max_workers == 1:
        for index, spec in enumerate(variant_specs):
            if time.time() >= deadline:
                break
            run_one(index, spec)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_one, index, spec) for index, spec in enumerate(variant_specs)]
            for future in as_completed(futures):
                future.result()


if __name__ == "__main__":
    main()
