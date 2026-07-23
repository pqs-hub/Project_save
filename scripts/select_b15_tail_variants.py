#!/usr/bin/env python3
"""Select a tail-planning variant using only final b15 test coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def incumbent_delta(payload: dict) -> float:
    record = payload.get("winner", payload)
    if "b15_delta_tc" in record:
        return float(record["b15_delta_tc"])
    selected = record.get("selected")
    if selected and "delta_test_coverage" in selected:
        return float(selected["delta_test_coverage"])
    if "delta_test_coverage" in record:
        return float(record["delta_test_coverage"])
    raise KeyError("incumbent manifest has no b15 delta-TC field")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--incumbent-manifest", type=Path, required=True)
    parser.add_argument(
        "--variants",
        required=True,
        help="Comma-separated directory tags below BASE/evals and BASE/plans.",
    )
    parser.add_argument(
        "--prefer-order",
        default="",
        help="Optional comma-separated deterministic tie preference (first wins).",
    )
    args = parser.parse_args()

    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    preference = {
        tag: index
        for index, tag in enumerate(
            item.strip() for item in args.prefer_order.split(",") if item.strip()
        )
    }
    incumbent = incumbent_delta(json.loads(args.incumbent_manifest.read_text()))
    records = []
    for tag in variants:
        summary_path = args.base / "evals" / tag / "summary.json"
        rows = json.loads(summary_path.read_text())
        final = max(rows, key=lambda row: int(row["step"]))
        records.append(
            {
                "variant": tag,
                "test_coverage": float(final["test_coverage"]),
                "delta_test_coverage": float(final["delta_test_coverage"]),
                "summary_json": str(summary_path),
                "plan_csv": str(args.base / "plans" / f"{tag}.csv"),
            }
        )

    best = max(
        records,
        key=lambda row: (
            row["delta_test_coverage"],
            -preference.get(row["variant"], len(preference)),
            row["variant"],
        ),
    )
    strict_win = best["delta_test_coverage"] > incumbent + 1e-12
    payload = {
        "selection_circuit": "iscas99__b15_1",
        "selection_metric": "final_delta_test_coverage",
        "tie_break": "retain incumbent on equality; otherwise use declared preference",
        "incumbent_delta_test_coverage": incumbent,
        "incumbent_manifest": str(args.incumbent_manifest),
        "strict_win": strict_win,
        "selected": best if strict_win else None,
        "best_exploratory_variant": best,
        "records": sorted(
            records,
            key=lambda row: (-row["delta_test_coverage"], row["variant"]),
        ),
    }
    output = args.base / "b15_selection.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
