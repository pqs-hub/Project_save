#!/usr/bin/env python3
"""Prepare leak-free late-horizon source circuits for Round9.

Round7/8 real counterfactual labels stop at prefix 127 even though the five
held-out ITC99 budgets range from 278 to 994.  This script admits six source
subcircuits with enough legal candidates for a 256-action trajectory and
records their baseline coverage and structural size for provenance.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tpi_jepa.bench import parse_bench
from tpi_jepa.protocol import eval_benchmarks_from_protocol


LOOP = Path("autoresearch/loop-260720-0945")
ROOT = LOOP / "model_training_round9"
LABELS = Path(
    "/data4/pengqingsong/DFT/Dataset/atalanta_bist_lowtc_subckt_100k_labels/labels.csv"
)
SUBCIRCUITS = LABELS.parent / "subcircuits"
EVAL_PROTOCOL = Path("configs/eval_protocol_coverage_only.json")
PLAN_BUDGET = 256
PREFIX_STEPS = [144, 176, 208, 240, 255]

# Low-coverage sources are deliberately represented, while the two largest
# available graphs reduce the structural-size shift to the held-out circuits.
BENCHMARKS = [
    "subckt_0288",  # baseline TC 62.8%
    "subckt_0360",  # baseline TC 68.3%
    "subckt_0012",  # baseline TC 92.6%
    "subckt_0266",  # baseline TC 93.5%
    "subckt_0309",  # 1.5k graph nodes
    "subckt_0230",  # 3.1k graph nodes
]


def baseline_rows() -> dict[str, dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    rows: dict[str, dict[str, str]] = {}
    with LABELS.open(newline="") as handle:
        for row in csv.DictReader(handle):
            benchmark_id = row.get("benchmark_id", "")
            if benchmark_id in BENCHMARKS and row.get("step") == "0":
                rows.setdefault(benchmark_id, row)
    return rows


def main() -> None:
    forbidden = eval_benchmarks_from_protocol(EVAL_PROTOCOL)
    leaked = sorted(set(BENCHMARKS) & forbidden)
    if leaked:
        raise ValueError(f"evaluation-protocol leakage: {leaked}")

    baselines = baseline_rows()
    missing = sorted(set(BENCHMARKS) - set(baselines))
    if missing:
        raise ValueError(f"missing source baseline labels: {missing}")

    circuits = []
    for benchmark_id in BENCHMARKS:
        bench_path = SUBCIRCUITS / f"{benchmark_id}.bench"
        if not bench_path.is_file():
            raise FileNotFoundError(bench_path)
        circuit = parse_bench(bench_path)
        row = baselines[benchmark_id]
        candidate_count = int(row["candidate_count"])
        if candidate_count <= PLAN_BUDGET:
            raise ValueError(
                f"{benchmark_id} has only {candidate_count} candidates for budget {PLAN_BUDGET}"
            )
        circuits.append(
            {
                "benchmark_id": benchmark_id,
                "bench_path": str(bench_path),
                "graph_nodes": len(circuit.node_names),
                "candidate_count": candidate_count,
                "baseline_test_coverage": float(row["test_coverage"]),
                "baseline_undetected_fault_count": int(row["undetected_fault_count"]),
            }
        )

    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "round": 9,
        "purpose": "late-horizon real one-step ATPG supervision",
        "accepted_benchmarks": BENCHMARKS,
        "excluded_benchmarks": sorted(forbidden),
        "eval_protocol": str(EVAL_PROTOCOL),
        "source_labels": str(LABELS),
        "target_circuits_in_training": False,
        "plan_budget": PLAN_BUDGET,
        "prefix_steps": PREFIX_STEPS,
        "circuits": circuits,
    }
    path = ROOT / "late_source_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(path)
    for row in circuits:
        print(
            f"{row['benchmark_id']} nodes={row['graph_nodes']} "
            f"candidates={row['candidate_count']} "
            f"baseline_tc={row['baseline_test_coverage']:.5f}"
        )


if __name__ == "__main__":
    main()
