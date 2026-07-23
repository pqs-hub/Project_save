#!/usr/bin/env python3
"""Select one global trust-residual setting using b15 results only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else REPO_ROOT / args.root
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else REPO_ROOT / args.checkpoint
    candidates = []
    for path in sorted((root / "settings").glob("*/results.tsv")):
        rows = list(csv.DictReader(path.open(newline=""), delimiter="\t"))
        row = next((item for item in rows if item.get("status") == "ok"), None)
        if row is None:
            continue
        env = json.loads((path.parent / "config.json").read_text())["planner_environment"]
        encoded = re.fullmatch(
            r"a(?P<alpha>[0-9p]+)_d(?P<decay>\d+)_h(?P<heads>\d+)_c(?P<cp0>\d+)_m(?P<head_margin>[0-9p]+)_v(?P<adv_margin>[0-9p]+)",
            path.parent.name,
        )
        if encoded is None:
            raise ValueError(f"unrecognized trust setting directory: {path.parent.name}")

        def setting_float(name: str) -> float:
            return float(encoded.group(name).replace("p", "."))

        candidates.append(
            {
                "setting": path.parent.name,
                "checkpoint": str(checkpoint.relative_to(REPO_ROOT)),
                "score_field": "q_typed_trust_context",
                "typed_residual_alpha": float(env["TPI_TYPED_RESIDUAL_ALPHA"]),
                "typed_residual_decay_steps": float(env["TPI_TYPED_RESIDUAL_DECAY_STEPS"]),
                "typed_trust_min_heads": int(env.get("TPI_TYPED_TRUST_MIN_HEADS") or encoded.group("heads")),
                "typed_trust_cp0_min_heads": int(
                    env.get("TPI_TYPED_TRUST_CP0_MIN_HEADS") or encoded.group("cp0")
                ),
                "typed_trust_head_margin": float(
                    env.get("TPI_TYPED_TRUST_HEAD_MARGIN") or setting_float("head_margin")
                ),
                "typed_trust_advantage_margin": float(
                    env.get("TPI_TYPED_TRUST_ADVANTAGE_MARGIN") or setting_float("adv_margin")
                ),
                "b15_delta_tc": float(row["delta_test_coverage"]),
            }
        )
    if not candidates:
        raise SystemExit(f"no successful b15 trust candidates under {root / 'settings'}")
    # Primary selection is measured b15 TC. Exact ties prefer stronger support,
    # then a smaller/shorter residual, which is the conservative global rule.
    candidates.sort(
        key=lambda item: (
            -item["b15_delta_tc"],
            -item["typed_trust_min_heads"],
            -item["typed_trust_cp0_min_heads"],
            item["typed_residual_alpha"],
            item["typed_residual_decay_steps"] or float("inf"),
            -item["typed_trust_head_margin"],
            -item["typed_trust_advantage_margin"],
            item["setting"],
        )
    )
    payload = {
        "selection_circuit": "b15_C",
        "selection_rule": "max b15 delta TC; exact ties prefer stronger support and smaller residual",
        "validation_circuits_read": [],
        "winner": candidates[0],
        "candidates": candidates,
    }
    out = root / "winner.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["winner"], indent=2))


if __name__ == "__main__":
    main()
