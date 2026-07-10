"""Train one Q-v1 variant, then run expanded/transfer oracle gates."""

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
B_ORACLE_0P05 = "autoresearch/train-ab-oracle-rank-260629/runs/B_oracle_0p05/best.pt"
VAL_ORACLE = "autoresearch/oracle-balanced-negative-rich-260629-wide/balanced_val_oracle_actions.tsv"
TRANSFER_ORACLE = "autoresearch/oracle-action-probe-260629-smoke/oracle_actions.tsv"
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def f(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def row_for(rows: list[dict[str, str]], checkpoint: str, score_field: str) -> dict[str, str]:
    for row in rows:
        if row.get("checkpoint_name") == checkpoint and row.get("score_field") == score_field:
            return row
    return {}


def verdict(row: dict[str, Any], promote_label: str = "PROMOTE_Q_V1") -> tuple[str, str]:
    if row["variant"] in {"incumbent", "B_oracle_0p05"}:
        return "BASELINE", ""
    reasons = []
    if row["expanded_spearman"] < 0.50:
        reasons.append("expanded_spearman_below_0p50")
    if row["expanded_negative_top1"] > 0.162:
        reasons.append("expanded_negative_top1_worse")
    if row["transfer_negative_top1"] > 0.167:
        reasons.append("transfer_negative_top1_worse")
    if row["transfer_top1_regret"] > 0.012552:
        reasons.append("transfer_regret_worse_than_incumbent")
    if row["transfer_spearman"] < 0.20:
        reasons.append("transfer_spearman_below_0p20")
    return (promote_label, "") if not reasons else ("REJECT", ",".join(reasons))


def summarize(variant: str, out_dir: Path, promote_label: str = "PROMOTE_Q_V1") -> list[dict[str, Any]]:
    expanded = read_tsv(out_dir / "gates" / variant / "expanded_val" / "oracle_action_value_summary.tsv")
    transfer = read_tsv(out_dir / "gates" / variant / "transfer" / "oracle_action_value_summary.tsv")
    specs = [
        ("incumbent", "hybrid_pred"),
        ("B_oracle_0p05", "derived_hard_reduction_hybrid_pred"),
        (variant, "q_pred"),
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
        row["verdict"], row["reasons"] = verdict(row, promote_label)
        rows.append(row)
    return rows


def write_report(variant: str, rows: list[dict[str, Any]], out_dir: Path, report_prefix: str = "Q-v1") -> None:
    lines = [
        f"# {report_prefix} Train/Eval Result: {variant}",
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
    path = out_dir / "summaries" / f"{variant}_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("autoresearch/q-v1-parallel-260630"))
    parser.add_argument("--plan-device", default="cuda")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--force-gates", action="store_true")
    parser.add_argument("--promote-label", default="PROMOTE_Q_V1")
    parser.add_argument("--report-prefix", default="Q-v1")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = read_json(config_path)
    run_dir = REPO_ROOT / str(config["run_dir"])
    checkpoint = run_dir / "best.pt"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_train:
        run([sys.executable, "-u", "-m", "tpi_jepa.train", "--config", str(config_path)])
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    checkpoint_specs = ",".join(
        [
            f"incumbent={INCUMBENT}",
            f"B_oracle_0p05={B_ORACLE_0P05}",
            f"{args.variant}={checkpoint}",
        ]
    )
    score_fields = "q_pred,score_pred,hybrid_pred,derived_hard_reduction_hybrid_pred"
    for split, oracle in [("expanded_val", VAL_ORACLE), ("transfer", TRANSFER_ORACLE)]:
        split_dir = args.out_dir / "gates" / args.variant / split
        summary = split_dir / "oracle_action_value_summary.tsv"
        if summary.exists() and not args.force_gates:
            print(f"[eval] skip existing {summary}", flush=True)
            continue
        run(
            [
                sys.executable,
                "-u",
                "scripts/evaluate_oracle_action_values.py",
                "--oracle-actions",
                oracle,
                "--checkpoints",
                checkpoint_specs,
                "--score-fields",
                score_fields,
                "--top-ks",
                "1,3,5",
                "--oracle-top-m",
                "1",
                "--plan-device",
                args.plan_device,
                "--out-dir",
                str(split_dir),
                "--baseline",
                "incumbent",
            ]
        )

    rows = summarize(args.variant, args.out_dir, args.promote_label)
    write_tsv(args.out_dir / "summaries" / f"{args.variant}_summary.tsv", rows, SUMMARY_FIELDS)
    write_report(args.variant, rows, args.out_dir, args.report_prefix)
    handoff = {
        "status": "completed",
        "variant": args.variant,
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "summary": str(args.out_dir / "summaries" / f"{args.variant}_summary.tsv"),
        "report": str(args.out_dir / "summaries" / f"{args.variant}_report.md"),
        "promoted": [row["variant"] for row in rows if row["verdict"] == args.promote_label],
    }
    write_json(args.out_dir / "summaries" / f"{args.variant}_handoff.json", handoff)
    print(json.dumps(handoff, indent=2), flush=True)


if __name__ == "__main__":
    main()
