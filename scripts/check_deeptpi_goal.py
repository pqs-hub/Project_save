#!/usr/bin/env python3
"""Emit a scalar distance to the uniform exact-legal DeepTPI goal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_root", type=Path)
    parser.add_argument("--require-success", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    root = args.eval_root if args.eval_root.is_absolute() else repo / args.eval_root
    summary = json.loads((root / "summary.json").read_text())
    rows = summary.get("per_circuit") or []
    if len(rows) != 5:
        raise SystemExit(f"expected five circuit rows, got {len(rows)}")

    audit = subprocess.run(
        [sys.executable, str(repo / "scripts/verify_uniform_exact_itc99.py"), str(root)],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    uniform_exact = audit.returncode == 0
    gaps = [float(row["gap_vs_deeptpi_pp"]) for row in rows]
    failures = sum(gap <= 0.0 for gap in gaps)
    deficit = sum(max(0.0, -gap) for gap in gaps)
    units = float(failures) + deficit
    if not uniform_exact:
        units += 100.0
    print(f"{units:.6f}")
    if args.require_success and units != 0.0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
