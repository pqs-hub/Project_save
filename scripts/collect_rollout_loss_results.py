"""Collect rollout loss ablation checkpoint diagnostics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


FIELDS = [
    "variant",
    "checkpoint",
    "epoch",
    "predictive_score",
    "reward_mae",
    "hard_reduction_mae",
    "hard_reduction_score",
    "delta_scoap_mae",
    "delta_scoap_acc_at_001",
    "scoap_mae",
    "hard_macro_f1_tuned",
    "hard_recall_at_top_10pct",
    "run_dir",
    "hard_eval_csv",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def numeric(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def pick_row(rows: list[dict[str, str]]) -> dict[str, str]:
    best_rows = [row for row in rows if str(row.get("checkpoint", "")).endswith("best.pt")]
    if best_rows:
        return best_rows[-1]
    return max(rows, key=lambda row: numeric(row.get("predictive_score"), -1e9))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Rollout Loss Ablation Summary",
        "",
        "| variant | reward MAE | hard reduction MAE | delta-SCOAP MAE | hard reduction score | predictive score |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {reward_mae} | {hard_reduction_mae} | {delta_scoap_mae} | "
            "{hard_reduction_score} | {predictive_score} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("autoresearch/rollout-loss-ablation-260630"))
    args = parser.parse_args()
    rows = []
    for csv_path in sorted((args.out_dir / "hard_eval").glob("*.csv")):
        eval_rows = read_rows(csv_path)
        if not eval_rows:
            continue
        row = pick_row(eval_rows)
        variant = csv_path.stem
        rows.append(
            {
                "variant": variant,
                "run_dir": f"runs/{variant}",
                "hard_eval_csv": str(csv_path),
                **{field: row.get(field, "") for field in FIELDS},
            }
        )
    if not rows:
        raise FileNotFoundError(f"no hard eval CSV files under {args.out_dir / 'hard_eval'}")
    write_tsv(args.out_dir / "rollout_loss_summary.tsv", rows)
    write_report(args.out_dir / "rollout_loss_report.md", rows)
    print(f"wrote {args.out_dir / 'rollout_loss_summary.tsv'}")
    print(f"wrote {args.out_dir / 'rollout_loss_report.md'}")


if __name__ == "__main__":
    main()
