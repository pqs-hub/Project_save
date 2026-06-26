"""Configuration-first auto-research for hard-fault predictive pretraining."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


RESULT_FIELDS = [
    "timestamp",
    "variant_id",
    "status",
    "seed",
    "best_epoch",
    "predictive_score",
    "hard_macro_f1_tuned",
    "hard_macro_f1_at_0p5",
    "hard_recall_at_top_10pct",
    "ece_sa0",
    "ece_sa1",
    "temperature_scaled_ece_sa0",
    "temperature_scaled_ece_sa1",
    "hard_reduction_score",
    "hard_count_top10_overlap",
    "scoap_acc_at_005",
    "hard_sa0_pr_auc",
    "hard_sa1_pr_auc",
    "lambda_hard",
    "lambda_hard_count",
    "lambda_hard_reduction",
    "lambda_hard_rank",
    "lambda_hard_brier",
    "lambda_hard_soft_f1",
    "encoder_type",
    "summary_mode",
    "hard_loss",
    "hard_asl_gamma_neg",
    "hard_asl_clip",
    "hard_head_type",
    "hard_pos_weight_max",
    "hard_negative_sample_ratio",
    "hard_negative_mining",
    "train_sample_strategy",
    "feature_mode",
    "edge_weight_mode",
    "edge_keep_ratio",
    "lambda_fc",
    "run_dir",
    "config_path",
    "checkpoint",
    "metrics_csv",
    "metrics_png",
    "elapsed_sec",
    "error",
]


OBJECTIVE_FIELDS = {
    "predictive": "predictive_score",
    "hard_f1": "hard_macro_f1_tuned",
    "hard_top10": "hard_recall_at_top_10pct",
    "hard_reduction": "hard_reduction_score",
}


def parse_csv_values(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_float_values(text: str) -> list[float]:
    return [float(item) for item in parse_csv_values(text)]


def parse_int_values(text: str) -> list[int]:
    return [int(item) for item in parse_csv_values(text)]


def sanitize(value: Any) -> str:
    return str(value).replace("/", "_").replace(":", "_").replace(".", "p")


def run_command(cmd: list[str], log_path: Path, stream: bool = False) -> tuple[bool, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w") as log_file:
        if stream:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert proc.stdout is not None
            for line in proc.stdout:
                log_file.write(line)
                log_file.flush()
                print(line, end="", flush=True)
            returncode = proc.wait()
        else:
            result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)
            returncode = result.returncode
    if returncode == 0:
        return True, ""
    try:
        tail = "".join(log_path.read_text(errors="replace").splitlines(True)[-80:])
    except OSError:
        tail = ""
    return False, f"returncode={returncode} elapsed={time.time() - started:.1f}s\n{tail}".strip()


def progress(message: str) -> None:
    print(f"[autoresearch] {datetime.now().strftime('%H:%M:%S')} {message}", flush=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def append_tsv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})


def numeric(value: Any, default: float = 0.0) -> float:
    if value in (None, "", "NA"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def best_metric_row(metrics_csv: Path, objective_field: str = "predictive_score") -> dict[str, str]:
    with metrics_csv.open(newline="") as f:
        rows = [row for row in csv.DictReader(f) if int(float(row.get("epoch") or 0)) > 0]
    if not rows:
        raise RuntimeError(f"No epoch rows found in metrics CSV: {metrics_csv}")
    return max((row for row in rows), key=lambda row: numeric(row.get(objective_field), -1.0))


def make_variant_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Generate a compact deterministic search grid, capped by --max-variants."""

    raw = []
    for values in itertools.product(
        parse_int_values(args.seeds),
        parse_float_values(args.lambda_hards),
        parse_float_values(args.lambda_hard_counts),
        parse_float_values(args.lambda_hard_reductions),
        parse_float_values(args.lambda_hard_ranks),
        parse_float_values(args.lambda_hard_briers),
        parse_float_values(args.lambda_hard_soft_f1s),
        parse_csv_values(args.encoder_types),
        parse_csv_values(args.summary_modes),
        parse_csv_values(args.hard_losses),
        parse_float_values(args.hard_asl_gamma_negs),
        parse_float_values(args.hard_asl_clips),
        parse_csv_values(args.hard_head_types),
        parse_int_values(args.hard_pos_weight_maxes),
        parse_int_values(args.hard_negative_sample_ratios),
        parse_csv_values(args.hard_negative_minings),
        parse_csv_values(args.train_sample_strategies),
        parse_csv_values(args.feature_modes),
        parse_csv_values(args.edge_weight_modes),
        parse_float_values(args.edge_keep_ratios),
        parse_float_values(args.lambda_fcs),
    ):
        (
            seed,
            lambda_hard,
            lambda_hard_count,
            lambda_hard_reduction,
            lambda_hard_rank,
            lambda_hard_brier,
            lambda_hard_soft_f1,
            encoder_type,
            summary_mode,
            hard_loss,
            hard_asl_gamma_neg,
            hard_asl_clip,
            hard_head_type,
            hard_pos_weight_max,
            hard_negative_sample_ratio,
            hard_negative_mining,
            train_sample_strategy,
            feature_mode,
            edge_weight_mode,
            edge_keep_ratio,
            lambda_fc,
        ) = values
        if edge_weight_mode == "mean" and edge_keep_ratio != 1.0:
            continue
        score = 0
        score += abs(lambda_hard - float(args.center_lambda_hard))
        score += abs(lambda_hard_count - float(args.center_lambda_hard_count))
        score += abs(lambda_hard_reduction - float(args.center_lambda_hard_reduction))
        score += abs(lambda_hard_rank - float(args.center_lambda_hard_rank))
        score += abs(lambda_hard_brier - float(args.center_lambda_hard_brier))
        score += abs(lambda_hard_soft_f1 - float(args.center_lambda_hard_soft_f1))
        score += {"gate_dir": 0.0, "topo_gate": 0.05, "mean": 0.35}.get(str(encoder_type).lower(), 0.5)
        score += {"cone": 0.0, "cone_pool": 0.0, "global": 0.25}.get(str(summary_mode).lower(), 0.4)
        score += {"asl": 0.0, "focal": 0.25, "bce": 0.6}.get(str(hard_loss).lower(), 0.8)
        score += {"residual_context": 0.0, "context": 0.1, "residual": 0.2, "mlp": 0.7}.get(
            str(hard_head_type).lower(),
            0.8,
        )
        score += 0.1 * abs(hard_pos_weight_max - 20)
        score += 0.15 * abs(hard_negative_sample_ratio - 5)
        score += {"topk": 0.0, "mixed": 0.15, "random": 0.45}.get(str(hard_negative_mining).lower(), 0.6)
        score += 0.0 if train_sample_strategy == "hard_weighted" else 0.35
        score += 0.5 if feature_mode != "testability" else 0.0
        score += 0.3 if edge_weight_mode != "fault_path" else 0.0
        score += abs(edge_keep_ratio - float(args.center_edge_keep_ratio))
        score += 2.0 * abs(lambda_fc)
        raw.append(
            {
                "_score": score,
                "seed": seed,
                "lambda_hard": lambda_hard,
                "lambda_hard_count": lambda_hard_count,
                "lambda_hard_reduction": lambda_hard_reduction,
                "lambda_hard_rank": lambda_hard_rank,
                "lambda_hard_brier": lambda_hard_brier,
                "lambda_hard_soft_f1": lambda_hard_soft_f1,
                "encoder_type": encoder_type,
                "summary_mode": summary_mode,
                "hard_loss": hard_loss,
                "hard_asl_gamma_neg": hard_asl_gamma_neg,
                "hard_asl_clip": hard_asl_clip,
                "hard_head_type": hard_head_type,
                "hard_pos_weight_max": hard_pos_weight_max,
                "hard_negative_sample_ratio": hard_negative_sample_ratio,
                "hard_negative_mining": hard_negative_mining,
                "train_sample_strategy": train_sample_strategy,
                "feature_mode": feature_mode,
                "edge_weight_mode": edge_weight_mode,
                "edge_keep_ratio": edge_keep_ratio,
                "lambda_fc": lambda_fc,
            }
        )
    raw.sort(key=lambda item: item["_score"])
    variants = []
    seen: set[str] = set()
    for item in raw:
        variant = {key: value for key, value in item.items() if key != "_score"}
        variant_id = (
            f"seed{variant['seed']}"
            f"__lh{sanitize(variant['lambda_hard'])}"
            f"__lhc{sanitize(variant['lambda_hard_count'])}"
            f"__lhr{sanitize(variant['lambda_hard_reduction'])}"
            f"__lhrk{sanitize(variant['lambda_hard_rank'])}"
            f"__lhb{sanitize(variant['lambda_hard_brier'])}"
            f"__lhsf{sanitize(variant['lambda_hard_soft_f1'])}"
            f"__enc{sanitize(variant['encoder_type'])}"
            f"__sum{sanitize(variant['summary_mode'])}"
            f"__hl{sanitize(variant['hard_loss'])}"
            f"__agn{sanitize(variant['hard_asl_gamma_neg'])}"
            f"__ac{sanitize(variant['hard_asl_clip'])}"
            f"__hh{sanitize(variant['hard_head_type'])}"
            f"__pw{variant['hard_pos_weight_max']}"
            f"__ns{variant['hard_negative_sample_ratio']}"
            f"__nm{sanitize(variant['hard_negative_mining'])}"
            f"__ts{sanitize(variant['train_sample_strategy'])}"
            f"__fm{variant['feature_mode']}"
            f"__ew{variant['edge_weight_mode']}"
            f"__ek{sanitize(variant['edge_keep_ratio'])}"
            f"__fc{sanitize(variant['lambda_fc'])}"
        )
        if variant_id in seen:
            continue
        seen.add(variant_id)
        variants.append({"variant_id": variant_id, **variant})
        if len(variants) >= args.max_variants:
            break
    return variants


def write_summary(
    path: Path,
    results: list[dict[str, Any]],
    base_config: dict,
    best: dict[str, Any] | None,
    objective: str,
    objective_field: str,
) -> None:
    lines = ["# Predictive Auto-Research Summary", ""]
    lines.append(f"generated_at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"objective: `{objective}` (`{objective_field}`)")
    lines.append("")
    if best is None:
        lines.append("No successful variants completed.")
    else:
        lines.extend(
            [
                "## Best Variant",
                "",
                f"- variant_id: `{best['variant_id']}`",
                f"- objective_value: `{best.get(objective_field, '')}`",
                f"- predictive_score: `{best['predictive_score']}`",
                f"- best_epoch: `{best['best_epoch']}`",
                f"- hard_macro_f1_tuned: `{best['hard_macro_f1_tuned']}`",
                f"- hard_macro_f1_at_0p5: `{best.get('hard_macro_f1_at_0p5', '')}`",
                f"- hard_recall_at_top_10pct: `{best['hard_recall_at_top_10pct']}`",
                f"- ece_sa0: `{best.get('ece_sa0', '')}`",
                f"- ece_sa1: `{best.get('ece_sa1', '')}`",
                f"- temperature_scaled_ece_sa0: `{best.get('temperature_scaled_ece_sa0', '')}`",
                f"- temperature_scaled_ece_sa1: `{best.get('temperature_scaled_ece_sa1', '')}`",
                f"- hard_reduction_score: `{best['hard_reduction_score']}`",
                "",
                "## Compared With Base Defaults",
                "",
            ]
        )
        for key in [
            "lambda_hard",
            "lambda_hard_count",
            "lambda_hard_reduction",
            "lambda_hard_rank",
            "lambda_hard_brier",
            "lambda_hard_soft_f1",
            "encoder_type",
            "summary_mode",
            "hard_pos_weight_max",
            "hard_negative_sample_ratio",
            "hard_loss",
            "hard_asl_gamma_neg",
            "hard_asl_clip",
            "hard_head_type",
            "hard_negative_mining",
            "train_sample_strategy",
            "feature_mode",
            "edge_weight_mode",
            "edge_keep_ratio",
            "lambda_fc",
        ]:
            lines.append(f"- {key}: base=`{base_config.get(key, '')}` best=`{best.get(key, '')}`")
        lines.extend(
            [
                "",
                "## Trend Notes",
                "",
                f"- This run selected the best variant by `{objective_field}`.",
                "- Keep `predictive_score` as a secondary guardrail so F1 improvements do not destroy ranking and reduction quality.",
                "- If train loss improves but hard F1 stalls, compare ASL against focal and switch negative mining between top-k and mixed.",
                "- If hard count overlap stays low, move to the second-stage hard-count calibration change.",
                "",
                "## Suggested Next Round",
                "",
                "- Center the next grid around the best encoder, summary, hard rank weight, hard loss, and hard-negative mining mode.",
                "- Keep `lambda_fc=0.0` until hard-fault predictive metrics plateau.",
                "- Add ranking or pairwise calibration loss only after ASL/focal and top-k mining plateau.",
            ]
        )
    lines.append("")
    lines.append("## Completed Variants")
    lines.append("")
    for row in results:
        lines.append(
            f"- `{row.get('variant_id')}` status={row.get('status')} "
            f"{objective_field}={row.get(objective_field, '')} predictive_score={row.get('predictive_score', '')}"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run predictive auto-research over hard-fault pretraining configs.")
    parser.add_argument("--base-config", default="configs/aig_lowtc_100k_hard_pretrain.json")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--max-variants", type=int, default=12)
    parser.add_argument("--max-val-samples", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds. Defaults to the base config seed.")
    parser.add_argument("--train-max-steps", type=int, default=None)
    parser.add_argument("--override-epochs", type=int, default=None)
    parser.add_argument("--override-max-train-samples", type=int, default=None)
    parser.add_argument("--override-max-train-steps-per-epoch", type=int, default=None)
    parser.add_argument("--override-max-val-samples-train", type=int, default=None)
    parser.add_argument("--override-max-val-steps-train", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cache-samples", action="store_true", help="Cache tensorized train/eval samples in each dataset process.")
    parser.add_argument(
        "--sample-cache-max-entries",
        type=int,
        default=None,
        help="Maximum cached samples per dataset instance. 0 or omitted means unlimited when caching is enabled.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stream-logs", action="store_true", help="Print train/eval subprocess logs while also writing log files.")
    parser.add_argument(
        "--objective",
        choices=sorted(OBJECTIVE_FIELDS),
        default="hard_f1",
        help="Metric used to select best checkpoint and best variant.",
    )
    parser.add_argument("--lambda-hards", default="0.5,0.7,1.0,1.5")
    parser.add_argument("--lambda-hard-counts", default="0.0,0.05,0.1,0.2")
    parser.add_argument("--lambda-hard-reductions", default="0.2,0.5,0.8")
    parser.add_argument("--lambda-hard-ranks", default="0.0")
    parser.add_argument("--lambda-hard-briers", default="0.0")
    parser.add_argument("--lambda-hard-soft-f1s", default="0.0")
    parser.add_argument("--encoder-types", default="mean")
    parser.add_argument("--summary-modes", default="global")
    parser.add_argument("--center-lambda-hard", type=float, default=0.7)
    parser.add_argument("--center-lambda-hard-count", type=float, default=0.1)
    parser.add_argument("--center-lambda-hard-reduction", type=float, default=0.5)
    parser.add_argument("--center-lambda-hard-rank", type=float, default=0.0)
    parser.add_argument("--center-lambda-hard-brier", type=float, default=0.0)
    parser.add_argument("--center-lambda-hard-soft-f1", type=float, default=0.0)
    parser.add_argument("--center-edge-keep-ratio", type=float, default=0.6)
    parser.add_argument("--hard-losses", default="asl,focal,bce")
    parser.add_argument("--hard-asl-gamma-negs", default="4.0")
    parser.add_argument("--hard-asl-clips", default="0.05")
    parser.add_argument("--hard-head-types", default="residual_context,mlp")
    parser.add_argument("--hard-pos-weight-maxes", default="10,20,40")
    parser.add_argument("--hard-negative-sample-ratios", default="3,5,10")
    parser.add_argument("--hard-negative-minings", default="topk,mixed,random")
    parser.add_argument("--train-sample-strategies", default="hard_weighted,shuffle")
    parser.add_argument("--feature-modes", default="testability,region")
    parser.add_argument("--edge-weight-modes", default="fault_path,mean")
    parser.add_argument("--edge-keep-ratios", default="0.6,0.8,1.0")
    parser.add_argument("--lambda-fcs", default="0.0")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%y%m%d-%H%M%S")
    out_dir = Path(args.out_dir or f"autoresearch/predictive-{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    base_config = json.loads(Path(args.base_config).read_text())
    if args.seeds is None:
        args.seeds = str(base_config.get("seed", 2026))
    base_config["save_epoch_checkpoints"] = True
    objective_field = OBJECTIVE_FIELDS[args.objective]
    write_json(out_dir / "runner_config.json", vars(args))
    write_json(out_dir / "base_config.json", base_config)

    results_path = out_dir / "results.tsv"
    variants = make_variant_grid(args)
    write_json(out_dir / "variants.json", variants)
    progress(
        f"out_dir={out_dir} variants={len(variants)} objective={args.objective} "
        f"objective_field={objective_field} stream_logs={bool(args.stream_logs)}"
    )

    results: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for variant_index, variant in enumerate(variants, start=1):
        started = time.time()
        variant_id = variant["variant_id"]
        run_dir = out_dir / "runs" / variant_id
        config_path = out_dir / "configs" / f"{variant_id}.json"
        config = dict(base_config)
        config.update({key: value for key, value in variant.items() if key != "variant_id"})
        config["run_dir"] = str(run_dir)
        config["save_epoch_checkpoints"] = True
        if args.override_epochs is not None:
            config["epochs"] = args.override_epochs
        if args.override_max_train_samples is not None:
            config["max_train_samples"] = args.override_max_train_samples
        if args.override_max_train_steps_per_epoch is not None:
            config["max_train_steps_per_epoch"] = args.override_max_train_steps_per_epoch
        if args.override_max_val_samples_train is not None:
            config["max_val_samples"] = args.override_max_val_samples_train
        if args.override_max_val_steps_train is not None:
            config["max_val_steps"] = args.override_max_val_steps_train
        if args.cache_samples:
            config["cache_samples"] = True
            config["cache_eval_samples"] = True
            if args.sample_cache_max_entries is not None:
                config["sample_cache_max_entries"] = args.sample_cache_max_entries
        write_json(config_path, config)
        row: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "variant_id": variant_id,
            "status": "dry_run" if args.dry_run else "running",
            "run_dir": str(run_dir),
            "config_path": str(config_path),
            **{key: value for key, value in variant.items() if key != "variant_id"},
        }
        if args.dry_run:
            progress(f"[{variant_index}/{len(variants)}] dry_run variant={variant_id}")
            append_tsv(results_path, row)
            results.append(row)
            continue

        train_log = out_dir / "logs" / f"{variant_id}.train.log"
        eval_log = out_dir / "logs" / f"{variant_id}.eval.log"
        train_cmd = ["python", "-m", "tpi_jepa.train", "--config", str(config_path)]
        if args.train_max_steps is not None:
            train_cmd.extend(["--max-steps", str(args.train_max_steps)])
        progress(f"[{variant_index}/{len(variants)}] train start variant={variant_id} log={train_log}")
        ok, error = run_command(train_cmd, train_log, stream=bool(args.stream_logs))
        if not ok:
            row.update({"status": "train_error", "error": error, "elapsed_sec": round(time.time() - started, 3)})
            append_tsv(results_path, row)
            results.append(row)
            progress(f"[{variant_index}/{len(variants)}] train_error elapsed={row['elapsed_sec']}s log={train_log}")
            continue
        progress(f"[{variant_index}/{len(variants)}] train done elapsed={time.time() - started:.1f}s")

        metrics_csv = run_dir / "target_metrics.csv"
        metrics_png = run_dir / "target_metrics.png"
        eval_cmd = [
            "python",
            "scripts/evaluate_hard_checkpoints.py",
            "--config",
            str(config_path),
            "--max-val-samples",
            str(args.max_val_samples),
            "--max-steps",
            str(args.max_steps),
            "--temperature-scale-hard",
        ]
        if args.device:
            eval_cmd.extend(["--device", args.device])
        progress(f"[{variant_index}/{len(variants)}] eval start variant={variant_id} log={eval_log}")
        ok, error = run_command(eval_cmd, eval_log, stream=bool(args.stream_logs))
        if not ok:
            row.update({"status": "eval_error", "error": error, "elapsed_sec": round(time.time() - started, 3)})
            append_tsv(results_path, row)
            results.append(row)
            progress(f"[{variant_index}/{len(variants)}] eval_error elapsed={row['elapsed_sec']}s log={eval_log}")
            continue

        metric = best_metric_row(metrics_csv, objective_field)
        checkpoint = run_dir / f"epoch_{int(float(metric['epoch'])):03d}.pt"
        row.update(
            {
                "status": "ok",
                "best_epoch": metric.get("epoch"),
                "predictive_score": metric.get("predictive_score"),
                "hard_macro_f1_tuned": metric.get("hard_macro_f1_tuned"),
                "hard_macro_f1_at_0p5": metric.get("hard_macro_f1_at_0p5"),
                "hard_recall_at_top_10pct": metric.get("hard_recall_at_top_10pct"),
                "ece_sa0": metric.get("ece_sa0"),
                "ece_sa1": metric.get("ece_sa1"),
                "temperature_scaled_ece_sa0": metric.get("temperature_scaled_ece_sa0"),
                "temperature_scaled_ece_sa1": metric.get("temperature_scaled_ece_sa1"),
                "hard_reduction_score": metric.get("hard_reduction_score"),
                "hard_count_top10_overlap": metric.get("hard_count_top10_overlap"),
                "scoap_acc_at_005": metric.get("scoap_acc_at_005"),
                "hard_sa0_pr_auc": metric.get("hard_sa0_pr_auc"),
                "hard_sa1_pr_auc": metric.get("hard_sa1_pr_auc"),
                "checkpoint": str(checkpoint),
                "metrics_csv": str(metrics_csv),
                "metrics_png": str(metrics_png),
                "elapsed_sec": round(time.time() - started, 3),
            }
        )
        append_tsv(results_path, row)
        results.append(row)
        progress(
            f"[{variant_index}/{len(variants)}] ok epoch={row['best_epoch']} "
            f"{objective_field}={row.get(objective_field)} predictive={row['predictive_score']} "
            f"elapsed={row['elapsed_sec']}s"
        )
        if row["status"] == "ok" and (
            best is None or numeric(row.get(objective_field), -1.0) > numeric(best.get(objective_field), -1.0)
        ):
            best = row
            shutil.copy2(config_path, out_dir / "best_config.json")
            if checkpoint.exists():
                shutil.copy2(checkpoint, out_dir / "best.pt")
            if metrics_csv.exists():
                shutil.copy2(metrics_csv, out_dir / "best_metrics.csv")
            if metrics_png.exists():
                shutil.copy2(metrics_png, out_dir / "best_metrics.png")
            progress(f"[{variant_index}/{len(variants)}] new_best {objective_field}={best.get(objective_field)}")

    write_summary(out_dir / "summary.md", results, base_config, best, args.objective, objective_field)
    print(f"wrote_results={results_path}")
    print(f"wrote_summary={out_dir / 'summary.md'}")
    if best:
        print(
            f"best_variant={best['variant_id']} "
            f"{objective_field}={best.get(objective_field)} predictive_score={best['predictive_score']}"
        )


if __name__ == "__main__":
    main()
