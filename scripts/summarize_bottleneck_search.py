#!/usr/bin/env python3
"""Summarize uniform two-circuit bottleneck explorations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


TARGETS = {"b22_C": 95.59, "b17_C": 91.67}


def rows(path: Path, delimiter: str) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def resolve(repo: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = resolve(repo, str(args.root))
    output: list[dict[str, object]] = []
    for variant_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if not all((variant_dir / circuit / "results.tsv").is_file() for circuit in TARGETS):
            continue
        circuit_rows: list[dict[str, object]] = []
        for circuit, target in TARGETS.items():
            attempts = rows(variant_dir / circuit / "results.tsv", "\t")
            successful = [row for row in attempts if row.get("status") == "ok"]
            if len(successful) != 1:
                raise RuntimeError(f"{variant_dir.name}/{circuit}: successful attempts={len(successful)}")
            result = successful[0]
            labels = rows(resolve(repo, result["eval_dir"]) / "labels.csv", ",")
            final = max(labels, key=lambda row: int(row["step"]))
            final_tc = 100.0 * float(final["test_coverage"])
            circuit_rows.append(
                {
                    "circuit": circuit,
                    "final_tc_pct": final_tc,
                    "target_tc_pct": target,
                    "gap_pp": final_tc - target,
                    "delta_tc_pp": 100.0 * float(final["delta_test_coverage"]),
                    "plan_elapsed_sec": float(result["plan_elapsed_sec"]),
                }
            )
        gaps = [float(row["gap_pp"]) for row in circuit_rows]
        output.append(
            {
                "variant": variant_dir.name,
                "min_gap_pp": min(gaps),
                "macro_gap_pp": sum(gaps) / len(gaps),
                "all_beat": all(gap > 0.0 for gap in gaps),
                "circuits": circuit_rows,
            }
        )
    output.sort(key=lambda row: (float(row["min_gap_pp"]), float(row["macro_gap_pp"])), reverse=True)
    payload = {"ranking": output}
    (root / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (root / "comparison.tsv").open("w", newline="") as handle:
        fieldnames = [
            "variant",
            "min_gap_pp",
            "macro_gap_pp",
            "all_beat",
            "b22_final_tc_pct",
            "b22_gap_pp",
            "b17_final_tc_pct",
            "b17_gap_pp",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for item in output:
            by_circuit = {row["circuit"]: row for row in item["circuits"]}
            writer.writerow(
                {
                    "variant": item["variant"],
                    "min_gap_pp": item["min_gap_pp"],
                    "macro_gap_pp": item["macro_gap_pp"],
                    "all_beat": item["all_beat"],
                    "b22_final_tc_pct": by_circuit["b22_C"]["final_tc_pct"],
                    "b22_gap_pp": by_circuit["b22_C"]["gap_pp"],
                    "b17_final_tc_pct": by_circuit["b17_C"]["final_tc_pct"],
                    "b17_gap_pp": by_circuit["b17_C"]["gap_pp"],
                }
            )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
