"""Run Q(s,a)-centric oracle training experiments."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

INCUMBENT = (
    "autoresearch/highseed-improvement-260626-run-posweight-30/runs/"
    "seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__"
    "hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt"
)
B_ORACLE_0P05 = "autoresearch/train-ab-oracle-rank-260629/runs/B_oracle_0p05/best.pt"
B_BASE_CONFIG = "autoresearch/ablate-scoap-version-b-260629-1705/configs/B_only_delta_scoap.json"
TRAIN_ORACLE = "autoresearch/oracle-balanced-negative-rich-260629-wide/balanced_train_oracle_actions.tsv"
VAL_ORACLE = "autoresearch/oracle-balanced-negative-rich-260629-wide/balanced_val_oracle_actions.tsv"
TRANSFER_ORACLE = "autoresearch/oracle-action-probe-260629-smoke/oracle_actions.tsv"

VARIANTS: dict[str, dict[str, float]] = {
    "Q_v0_rank0p5": {"lambda_q_value": 0.5, "lambda_q_rank": 0.5, "lambda_candidate": 0.0},
    "Q_v0_rank1p0": {"lambda_q_value": 0.5, "lambda_q_rank": 1.0, "lambda_candidate": 0.0},
    "Q_v0_rank2p0": {"lambda_q_value": 0.5, "lambda_q_rank": 2.0, "lambda_candidate": 0.0},
    "Q_v0_value1_rank1": {"lambda_q_value": 1.0, "lambda_q_rank": 1.0, "lambda_candidate": 0.0},
}

SUMMARY_FIELDS = [
    "variant",
    "score_field",
    "expanded_spearman",
    "expanded_negative_top1",
    "expanded_top1_real_delta",
    "expanded_top1_regret",
    "transfer_spearman",
    "transfer_negative_top1",
    "transfer_top1_real_delta",
    "transfer_top1_regret",
    "verdict",
    "reasons",
]


def parse_csv_values(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def run(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def completed_checkpoint(variant: str, out_dir: Path) -> bool:
    checkpoint = out_dir / "runs" / variant / "best.pt"
    history = out_dir / "runs" / variant / "history.csv"
    if not checkpoint.exists() or not history.exists():
        return False
    rows = read_csv(history)
    return any(row.get("epoch") == "4" for row in rows)


def build_config(variant: str, out_dir: Path, device: str) -> Path:
    spec = VARIANTS[variant]
    config = read_json(REPO_ROOT / B_BASE_CONFIG)
    config.update(
        {
            "run_dir": str(out_dir / "runs" / variant),
            "device": device,
            "oracle_actions": TRAIN_ORACLE,
            "oracle_ranking_score_field": "q_pred",
            "lambda_jepa": 0.1,
            "lambda_scoap": 0.0,
            "lambda_delta_scoap": 0.0,
            "lambda_hard": 0.0,
            "lambda_hard_rank": 0.0,
            "lambda_hard_brier": 0.0,
            "lambda_hard_soft_f1": 0.0,
            "lambda_hard_count": 0.0,
            "lambda_hard_reduction": 0.0,
            "lambda_fc": 0.0,
            "lambda_pattern": 0.0,
            "lambda_return": 0.0,
            "lambda_oracle_rank": 0.0,
            "lambda_oracle_value": 0.0,
            "lambda_candidate": spec["lambda_candidate"],
            "lambda_q_rank": spec["lambda_q_rank"],
            "lambda_q_value": spec["lambda_q_value"],
            "oracle_batch_groups": 8,
            "oracle_every_n_steps": 1,
            "oracle_warmup_epochs": 0,
            "oracle_ramp_epochs": 1,
            "oracle_pairwise_min_delta": 0.001,
            "oracle_pairwise_temperature": 1.0,
            "oracle_max_actions_per_group": 0,
            "candidate_target_temperature": 1.0,
            "candidate_pred_temperature": 1.0,
            "epochs": 4,
            "max_train_samples": 20000,
            "max_val_samples": 4096,
            "max_train_steps_per_epoch": 500,
            "max_val_steps": 256,
            "seed": 2030,
            "q_centric": True,
        }
    )
    path = out_dir / "configs" / f"{variant}.json"
    write_json(path, config)
    return path


def train_variant(variant: str, out_dir: Path, device: str, force: bool = False) -> Path:
    config = build_config(variant, out_dir, device)
    checkpoint = out_dir / "runs" / variant / "best.pt"
    if not force and completed_checkpoint(variant, out_dir):
        return checkpoint
    run([sys.executable, "-m", "tpi_jepa.train", "--config", str(config)], out_dir / "logs" / f"{variant}.train.log")
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def train_variants(variants: list[str], out_dir: Path, devices: list[str], force: bool) -> None:
    assignments = {variant: devices[index % len(devices)] for index, variant in enumerate(variants)}
    with ThreadPoolExecutor(max_workers=min(len(devices), len(variants))) as executor:
        futures = {
            executor.submit(train_variant, variant, out_dir, device, force): variant
            for variant, device in assignments.items()
        }
        for future in as_completed(futures):
            future.result()


def checkpoint_specs(variants: list[str], out_dir: Path) -> str:
    specs = [f"incumbent={INCUMBENT}", f"B_oracle_0p05={B_ORACLE_0P05}"]
    for variant in variants:
        specs.append(f"{variant}={out_dir / 'runs' / variant / 'best.pt'}")
    return ",".join(specs)


def run_oracle_gates(variants: list[str], out_dir: Path, device: str, force: bool) -> None:
    specs = checkpoint_specs(variants, out_dir)
    score_fields = "q_pred,score_pred,hybrid_pred,derived_hard_reduction_hybrid_pred"
    for split, oracle in [("expanded_val", VAL_ORACLE), ("transfer", TRANSFER_ORACLE)]:
        summary = out_dir / "gates" / split / "oracle_action_value_summary.tsv"
        if summary.exists() and not force:
            continue
        run(
            [
                sys.executable,
                "scripts/evaluate_oracle_action_values.py",
                "--oracle-actions",
                oracle,
                "--checkpoints",
                specs,
                "--score-fields",
                score_fields,
                "--top-ks",
                "1,3,5",
                "--oracle-top-m",
                "1",
                "--plan-device",
                device,
                "--out-dir",
                str(out_dir / "gates" / split),
                "--baseline",
                "incumbent",
            ],
            out_dir / "logs" / f"gate_{split}.log",
        )


def row_for(rows: list[dict[str, str]], name: str, score_field: str) -> dict[str, str]:
    for row in rows:
        if row.get("checkpoint_name") == name and row.get("score_field") == score_field:
            return row
    return {}


def f(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def verdict(row: dict[str, Any]) -> tuple[str, str]:
    if row["variant"] in {"incumbent", "B_oracle_0p05"}:
        return "BASELINE", ""
    reasons = []
    if row["expanded_spearman"] < 0.425:
        reasons.append("expanded_spearman_below_B_oracle_0p05")
    if row["expanded_negative_top1"] > 0.162:
        reasons.append("expanded_negative_top1_worse")
    if row["transfer_negative_top1"] > 0.167:
        reasons.append("transfer_negative_top1_worse")
    if row["transfer_top1_regret"] > 0.0175:
        reasons.append("transfer_regret_worse")
    return ("PROMOTE_Q", "" if not reasons else ",".join(reasons)) if not reasons else ("REJECT", ",".join(reasons))


def summarize(variants: list[str], out_dir: Path) -> list[dict[str, Any]]:
    expanded = read_tsv(out_dir / "gates" / "expanded_val" / "oracle_action_value_summary.tsv")
    transfer = read_tsv(out_dir / "gates" / "transfer" / "oracle_action_value_summary.tsv")
    specs = [
        ("incumbent", "hybrid_pred"),
        ("B_oracle_0p05", "derived_hard_reduction_hybrid_pred"),
        *[(variant, "q_pred") for variant in variants],
    ]
    rows = []
    for name, score_field in specs:
        erow = row_for(expanded, name, score_field)
        trow = row_for(transfer, name, score_field)
        row = {
            "variant": name,
            "score_field": score_field,
            "expanded_spearman": f(erow, "mean_spearman"),
            "expanded_negative_top1": f(erow, "negative_top1_rate"),
            "expanded_top1_real_delta": f(erow, "mean_top1_real_delta_tc"),
            "expanded_top1_regret": f(erow, "mean_top1_regret"),
            "transfer_spearman": f(trow, "mean_spearman"),
            "transfer_negative_top1": f(trow, "negative_top1_rate"),
            "transfer_top1_real_delta": f(trow, "mean_top1_real_delta_tc"),
            "transfer_top1_regret": f(trow, "mean_top1_regret"),
        }
        row["verdict"], row["reasons"] = verdict(row)
        rows.append(row)
    return rows


def write_report(rows: list[dict[str, Any]], out_dir: Path) -> None:
    lines = [
        "# Q Oracle Experiment",
        "",
        f"generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "| variant | score | verdict | expanded Spearman | expanded neg top1 | transfer Spearman | transfer neg top1 | transfer regret |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {variant} | `{score_field}` | {verdict} | {expanded_spearman:.6f} | {expanded_negative_top1:.6f} | "
            "{transfer_spearman:.6f} | {transfer_negative_top1:.6f} | {transfer_top1_regret:.6f} |".format(**row)
        )
    lines.extend(["", "## Notes", "", "- `q_pred` is the Q(s,a) decision score.", "- Legacy scores are included only as baselines."])
    (out_dir / "final_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("autoresearch/q-oracle-260629"))
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--parallel-devices", default="cuda:4,cuda:5,cuda:6,cuda:7")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-gates", action="store_true")
    args = parser.parse_args()

    variants = parse_csv_values(args.variants)
    unknown = [variant for variant in variants if variant not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")
    devices = parse_csv_values(args.parallel_devices)
    if not devices:
        raise ValueError("at least one device is required")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.out_dir / "experiment_manifest.json",
        {
            "variants": variants,
            "train_oracle": TRAIN_ORACLE,
            "val_oracle": VAL_ORACLE,
            "transfer_oracle": TRANSFER_ORACLE,
            "parallel_devices": devices,
        },
    )
    if not args.skip_train:
        train_variants(variants, args.out_dir, devices, force=args.force_train)
    run_oracle_gates(variants, args.out_dir, devices[0], force=args.force_gates)
    rows = summarize(variants, args.out_dir)
    write_tsv(args.out_dir / "q_oracle_summary.tsv", rows, SUMMARY_FIELDS)
    write_report(rows, args.out_dir)
    write_json(
        args.out_dir / "handoff.json",
        {
            "mode": "fix",
            "objective": "Q(s,a)-centric oracle training experiment",
            "status": "completed",
            "out_dir": str(args.out_dir),
            "outputs": {
                "summary": str(args.out_dir / "q_oracle_summary.tsv"),
                "report": str(args.out_dir / "final_report.md"),
            },
            "promoted": [row["variant"] for row in rows if row["verdict"] == "PROMOTE_Q"],
        },
    )
    print((args.out_dir / "handoff.json").read_text())


if __name__ == "__main__":
    main()
