"""Run Scheme A/B main-model oracle-ranking experiments."""

from __future__ import annotations

import argparse
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
A_BASE_CONFIG = "autoresearch/ablate-scoap-version-a-260629-1806/configs/A_only_scoap.json"
B_BASE_CONFIG = "autoresearch/ablate-scoap-version-b-260629-1705/configs/B_only_delta_scoap.json"
A_NO_ORACLE_CKPT = "autoresearch/ablate-scoap-version-a-260629-1806/runs/A_only_scoap/best.pt"
B_NO_ORACLE_CKPT = "autoresearch/ablate-scoap-version-b-260629-1705/runs/B_only_delta_scoap/best.pt"
TRAIN_ORACLE = "autoresearch/oracle-balanced-negative-rich-260629-wide/balanced_train_oracle_actions.tsv"
VAL_ORACLE = "autoresearch/oracle-balanced-negative-rich-260629-wide/balanced_val_oracle_actions.tsv"
TRANSFER_ORACLE = "autoresearch/oracle-action-probe-260629-smoke/oracle_actions.tsv"

VARIANTS = {
    "A_oracle_0p01": {
        "scheme": "A",
        "base_config": A_BASE_CONFIG,
        "lambda_oracle_rank": 0.01,
        "oracle_ranking_score_field": "hard_reduction_total_pred",
    },
    "A_oracle_0p03": {
        "scheme": "A",
        "base_config": A_BASE_CONFIG,
        "lambda_oracle_rank": 0.03,
        "oracle_ranking_score_field": "hard_reduction_total_pred",
    },
    "A_oracle_0p05": {
        "scheme": "A",
        "base_config": A_BASE_CONFIG,
        "lambda_oracle_rank": 0.05,
        "oracle_ranking_score_field": "hard_reduction_total_pred",
    },
    "B_oracle_0p01": {
        "scheme": "B",
        "base_config": B_BASE_CONFIG,
        "lambda_oracle_rank": 0.01,
        "oracle_ranking_score_field": "derived_hard_reduction_hybrid_pred",
    },
    "B_oracle_0p03": {
        "scheme": "B",
        "base_config": B_BASE_CONFIG,
        "lambda_oracle_rank": 0.03,
        "oracle_ranking_score_field": "derived_hard_reduction_hybrid_pred",
    },
    "B_oracle_0p05": {
        "scheme": "B",
        "base_config": B_BASE_CONFIG,
        "lambda_oracle_rank": 0.05,
        "oracle_ranking_score_field": "derived_hard_reduction_hybrid_pred",
    },
}


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


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def run(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def build_config(variant: str, out_dir: Path, device: str) -> Path:
    spec = VARIANTS[variant]
    config = read_json(REPO_ROOT / spec["base_config"])
    config.update(
        {
            "run_dir": str(out_dir / "runs" / variant),
            "device": device,
            "oracle_actions": TRAIN_ORACLE,
            "lambda_oracle_rank": spec["lambda_oracle_rank"],
            "lambda_oracle_value": 0.0,
            "oracle_ranking_score_field": spec["oracle_ranking_score_field"],
            "oracle_batch_groups": 4,
            "oracle_every_n_steps": 4,
            "oracle_warmup_epochs": 1,
            "oracle_ramp_epochs": 2,
            "oracle_pairwise_min_delta": 0.001,
            "oracle_pairwise_temperature": 1.0,
            "oracle_max_actions_per_group": 0,
            "epochs": 4,
            "max_train_samples": 20000,
            "max_val_samples": 4096,
            "max_train_steps_per_epoch": 500,
            "max_val_steps": 256,
            "seed": 2030,
        }
    )
    path = out_dir / "configs" / f"{variant}.json"
    write_json(path, config)
    return path


def train_variant(variant: str, out_dir: Path, device: str) -> Path:
    config_path = build_config(variant, out_dir, device)
    run(
        [sys.executable, "-m", "tpi_jepa.train", "--config", str(config_path)],
        out_dir / "logs" / f"{variant}.train.log",
    )
    checkpoint = out_dir / "runs" / variant / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def checkpoint_specs(variants: list[str], out_dir: Path) -> str:
    specs = [
        f"incumbent={INCUMBENT}",
        f"A_no_oracle={A_NO_ORACLE_CKPT}",
        f"B_no_oracle={B_NO_ORACLE_CKPT}",
    ]
    for variant in variants:
        specs.append(f"{variant}={out_dir / 'runs' / variant / 'best.pt'}")
    return ",".join(specs)


def run_oracle_gates(variants: list[str], out_dir: Path, device: str) -> None:
    specs = checkpoint_specs(variants, out_dir)
    score_fields = "hard_reduction_total_pred,hybrid_pred,derived_hard_reduction_total_pred,derived_hard_reduction_hybrid_pred"
    for split, oracle in [("expanded_val", VAL_ORACLE), ("transfer", TRANSFER_ORACLE)]:
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


def run_hard_gates(variants: list[str], out_dir: Path, device: str) -> None:
    for variant in variants:
        run(
            [
                sys.executable,
                "scripts/evaluate_hard_checkpoints.py",
                "--config",
                str(out_dir / "configs" / f"{variant}.json"),
                "--run-dir",
                str(out_dir / "runs" / variant),
                "--out-csv",
                str(out_dir / "gates" / "hard" / f"{variant}.csv"),
                "--max-val-samples",
                "4096",
                "--max-steps",
                "256",
                "--device",
                device,
                "--temperature-scale-hard",
            ],
            out_dir / "logs" / f"gate_hard_{variant}.log",
        )


def best_row(rows: list[dict[str, str]], checkpoint: str, score_field: str) -> dict[str, str] | None:
    matches = [
        row
        for row in rows
        if row.get("checkpoint_name") == checkpoint and row.get("score_field") == score_field
    ]
    return matches[0] if matches else None


def hard_best_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    rows = read_tsv(path) if path.suffix == ".tsv" else []
    if not rows:
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
    best_rows = [row for row in rows if Path(row.get("checkpoint", "")).name == "best.pt"]
    return best_rows[0] if best_rows else (rows[0] if rows else None)


def f(row: dict[str, str] | None, key: str) -> float:
    if not row:
        return float("nan")
    try:
        return float(row.get(key, "nan"))
    except ValueError:
        return float("nan")


def summarize(variants: list[str], out_dir: Path) -> list[dict[str, Any]]:
    expanded = read_tsv(out_dir / "gates" / "expanded_val" / "oracle_action_value_summary.tsv")
    transfer = read_tsv(out_dir / "gates" / "transfer" / "oracle_action_value_summary.tsv")
    rows: list[dict[str, Any]] = []
    for name in ["incumbent", "A_no_oracle", "B_no_oracle", *variants]:
        scheme = VARIANTS.get(name, {}).get("scheme", "baseline")
        score_field = (
            VARIANTS[name]["oracle_ranking_score_field"]
            if name in VARIANTS
            else ("hybrid_pred" if name == "incumbent" else ("hard_reduction_total_pred" if name == "A_no_oracle" else "derived_hard_reduction_hybrid_pred"))
        )
        erow = best_row(expanded, name, score_field)
        trow = best_row(transfer, name, score_field)
        hrow = hard_best_row(out_dir / "gates" / "hard" / f"{name}.csv") if name in variants else None
        rows.append(
            {
                "variant": name,
                "scheme": scheme,
                "score_field": score_field,
                "expanded_spearman": f(erow, "mean_spearman"),
                "expanded_negative_top1": f(erow, "negative_top1_rate"),
                "expanded_top1_real_delta": f(erow, "mean_top1_real_delta_tc"),
                "expanded_top1_regret": f(erow, "mean_top1_regret"),
                "transfer_spearman": f(trow, "mean_spearman"),
                "transfer_negative_top1": f(trow, "negative_top1_rate"),
                "transfer_top1_real_delta": f(trow, "mean_top1_real_delta_tc"),
                "transfer_top1_regret": f(trow, "mean_top1_regret"),
                "hard_macro_f1_tuned": f(hrow, "hard_macro_f1_tuned"),
                "hard_reduction_score": f(hrow, "hard_reduction_score"),
                "derived_hard_reduction_score": f(hrow, "derived_hard_reduction_score"),
            }
        )
    return rows


def verdict(row: dict[str, Any], incumbent: dict[str, Any]) -> tuple[str, str]:
    if row["variant"] in {"incumbent", "A_no_oracle", "B_no_oracle"}:
        return "BASELINE", ""
    reasons: list[str] = []
    if row["transfer_top1_real_delta"] < 0.0:
        reasons.append("transfer_top1_real_delta_negative")
    if row["transfer_negative_top1"] > incumbent["transfer_negative_top1"] + 1e-9:
        reasons.append("transfer_negative_top1_worse")
    if row["transfer_top1_regret"] > incumbent["transfer_top1_regret"] + 0.005:
        reasons.append("transfer_regret_worse_than_slack")
    if row["hard_macro_f1_tuned"] == row["hard_macro_f1_tuned"] and row["hard_macro_f1_tuned"] < 0.7948049980095285 - 0.03:
        reasons.append("hard_f1_drop")
    if reasons:
        return "REJECT", ",".join(reasons)
    if row["transfer_top1_regret"] <= incumbent["transfer_top1_regret"] + 1e-9:
        return "PROMOTE_MAIN_SCORE", ""
    return "PROMOTE_GUARDED_RERANK", "transfer_safety_ok_but_regret_or_spearman_not_better"


def markdown_report(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Scheme A/B Oracle-Ranking Main-Model Experiment",
        "",
        f"generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "## Summary",
        "",
        "| variant | verdict | score | expanded Spearman | expanded neg top1 | transfer Spearman | transfer neg top1 | transfer top1 delta | transfer regret | hard F1 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {verdict} | `{score}` | {es:.6f} | {en:.6f} | {ts:.6f} | {tn:.6f} | {td:.6f} | {tr:.6f} | {hf:.6f} |".format(
                variant=row["variant"],
                verdict=row.get("verdict", ""),
                score=row["score_field"],
                es=row["expanded_spearman"],
                en=row["expanded_negative_top1"],
                ts=row["transfer_spearman"],
                tn=row["transfer_negative_top1"],
                td=row["transfer_top1_real_delta"],
                tr=row["transfer_top1_regret"],
                hf=row["hard_macro_f1_tuned"],
            )
        )
    lines.extend(["", "## Notes", "", "- `A_no_oracle` and `B_no_oracle` are prior no-oracle baselines.", "- `incumbent` uses `hybrid_pred`.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("autoresearch/train-ab-oracle-rank-260629"))
    parser.add_argument("--variants", default="A_oracle_0p03,B_oracle_0p03")
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-gates", action="store_true")
    args = parser.parse_args()

    variants = parse_csv_values(args.variants)
    unknown = [variant for variant in variants if variant not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown variants: {unknown}; valid={sorted(VARIANTS)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.out_dir / "experiment_manifest.json",
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "variants": variants,
            "device": args.device,
            "train_oracle": TRAIN_ORACLE,
            "val_oracle": VAL_ORACLE,
            "transfer_oracle": TRANSFER_ORACLE,
        },
    )
    if not args.skip_train:
        for variant in variants:
            train_variant(variant, args.out_dir, args.device)
    if not args.skip_gates:
        run_hard_gates(variants, args.out_dir, args.device)
        run_oracle_gates(variants, args.out_dir, args.device)
    rows = summarize(variants, args.out_dir)
    incumbent = next(row for row in rows if row["variant"] == "incumbent")
    for row in rows:
        row["verdict"], row["reasons"] = verdict(row, incumbent)
    fields = list(rows[0].keys())
    write_tsv(args.out_dir / "ab_oracle_rank_summary.tsv", rows, fields)
    (args.out_dir / "final_report.md").write_text(markdown_report(rows))
    handoff = {
        "mode": "fix",
        "status": "completed",
        "objective": "scheme A/B oracle-ranking main-model experiment",
        "out_dir": str(args.out_dir),
        "variants": variants,
        "outputs": {
            "summary": str(args.out_dir / "ab_oracle_rank_summary.tsv"),
            "report": str(args.out_dir / "final_report.md"),
            "manifest": str(args.out_dir / "experiment_manifest.json"),
        },
        "best_promoted": [row["variant"] for row in rows if str(row.get("verdict", "")).startswith("PROMOTE")],
    }
    write_json(args.out_dir / "handoff.json", handoff)
    print(json.dumps(handoff, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

