"""Sample subcircuits for negative-rich oracle action collection."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import random
from typing import Any


def read_oracle_benchmarks(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if "benchmark_id" not in (reader.fieldnames or []):
            raise ValueError(f"{path} has no benchmark_id column")
        return {str(row.get("benchmark_id", "")).strip() for row in reader if row.get("benchmark_id")}


def write_lines(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{value}\n" for value in values))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def bench_ids(subckt_dir: Path) -> list[str]:
    if not subckt_dir.exists():
        raise FileNotFoundError(subckt_dir)
    return sorted(path.stem for path in subckt_dir.glob("subckt_*.bench"))


def split_sample(values: list[str], count: int) -> tuple[list[str], list[str]]:
    count = max(0, min(count, len(values)))
    return values[:count], values[count:]


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Negative-Rich Oracle Subckt Sample",
        "",
        f"generated_at: `{payload['generated_at']}`",
        "",
        "## Summary",
        "",
        "| item | count |",
        "|---|---:|",
        f"| all subckt bench files | {payload['counts']['all_subckts']} |",
        f"| excluded subckts | {payload['counts']['excluded_subckts']} |",
        f"| existing train subckts | {payload['counts']['existing_train_subckts']} |",
        f"| eligible train-pool subckts | {payload['counts']['eligible_subckts']} |",
        f"| fresh eligible subckts | {payload['counts']['fresh_eligible_subckts']} |",
        f"| pilot subckts | {payload['counts']['pilot_subckts']} |",
        f"| topup subckts | {payload['counts']['topup_subckts']} |",
        f"| remaining after pilot/topup | {payload['counts']['remaining_after_topup']} |",
        "",
        "## Policy",
        "",
        "- Keep expanded validation subckts excluded from training collection.",
        "- Prefer subckts that do not already have train oracle labels.",
        "- Keep `all_eligible_subckts.txt` so collection can be expanded later instead of stopping at a small fixed target.",
        "",
        "## Files",
        "",
        "- `pilot_subckts.txt`: first backend batch.",
        "- `topup_subckts.txt`: second backend batch if pilot is insufficient.",
        "- `remaining_subckts.txt`: additional eligible subckts after pilot/topup.",
        "- `all_eligible_subckts.txt`: all non-validation subckts available for train oracle collection.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subckt-dir", type=Path, required=True)
    parser.add_argument("--exclude-oracle", type=Path, action="append", default=[])
    parser.add_argument("--existing-train-oracle", type=Path, action="append", default=[])
    parser.add_argument("--pilot-count", type=int, default=96)
    parser.add_argument("--topup-count", type=int, default=192)
    parser.add_argument("--seed", type=int, default=260629)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    all_ids = bench_ids(args.subckt_dir)
    excluded: set[str] = set()
    for path in args.exclude_oracle:
        excluded.update(read_oracle_benchmarks(path))
    existing_train: set[str] = set()
    for path in args.existing_train_oracle:
        existing_train.update(read_oracle_benchmarks(path))

    eligible = [bench for bench in all_ids if bench not in excluded]
    fresh = [bench for bench in eligible if bench not in existing_train]
    reused_train = [bench for bench in eligible if bench in existing_train]

    rng = random.Random(args.seed)
    rng.shuffle(fresh)
    rng.shuffle(reused_train)

    # Prefer fresh subckts, but keep existing-train subckts as a fallback so the
    # caller can ask for a large pool without silently underfilling it.
    ordered = [*fresh, *reused_train]
    pilot, rest = split_sample(ordered, args.pilot_count)
    topup, remaining = split_sample(rest, args.topup_count)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_lines(args.out_dir / "pilot_subckts.txt", pilot)
    write_lines(args.out_dir / "topup_subckts.txt", topup)
    write_lines(args.out_dir / "remaining_subckts.txt", remaining)
    write_lines(args.out_dir / "all_eligible_subckts.txt", ordered)
    write_lines(args.out_dir / "excluded_subckts.txt", sorted(excluded))
    write_lines(args.out_dir / "existing_train_subckts.txt", sorted(existing_train))

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "subckt_dir": str(args.subckt_dir),
            "exclude_oracle": [str(path) for path in args.exclude_oracle],
            "existing_train_oracle": [str(path) for path in args.existing_train_oracle],
            "seed": args.seed,
            "pilot_count": args.pilot_count,
            "topup_count": args.topup_count,
        },
        "counts": {
            "all_subckts": len(all_ids),
            "excluded_subckts": len(excluded),
            "existing_train_subckts": len(existing_train),
            "eligible_subckts": len(eligible),
            "fresh_eligible_subckts": len(fresh),
            "pilot_subckts": len(pilot),
            "topup_subckts": len(topup),
            "remaining_after_topup": len(remaining),
        },
        "files": {
            "pilot_subckts": str(args.out_dir / "pilot_subckts.txt"),
            "topup_subckts": str(args.out_dir / "topup_subckts.txt"),
            "remaining_subckts": str(args.out_dir / "remaining_subckts.txt"),
            "all_eligible_subckts": str(args.out_dir / "all_eligible_subckts.txt"),
            "excluded_subckts": str(args.out_dir / "excluded_subckts.txt"),
            "existing_train_subckts": str(args.out_dir / "existing_train_subckts.txt"),
        },
        "samples": {
            "pilot_head": pilot[:10],
            "topup_head": topup[:10],
            "remaining_head": remaining[:10],
        },
    }
    write_json(args.out_dir / "sample_manifest.json", payload)
    (args.out_dir / "pool_report.md").write_text(markdown_report(payload))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

