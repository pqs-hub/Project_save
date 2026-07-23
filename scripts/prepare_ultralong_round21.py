#!/usr/bin/env python3
"""Prepare leak-free ultra-long prefix sources for horizon-aware ranking."""

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
ROOT = LOOP / "model_training_round21"
LABELS = Path(
    "/data4/pengqingsong/DFT/Dataset/atalanta_bist_lowtc_subckt_100k_labels/labels.csv"
)
SUBCIRCUITS = LABELS.parent / "subcircuits"
EVAL_PROTOCOL = Path("configs/eval_protocol_coverage_only.json")
PLAN_BUDGET = 768
PREFIX_STEPS = [320, 448, 576, 704, 767]
BENCHMARKS = [
    "subckt_0360",
    "subckt_0266",
    "subckt_0289",
    "subckt_0347",
    "subckt_0292",
    "subckt_0020",
    "subckt_0197",
    "subckt_0116",
    "subckt_0298",
    "subckt_0309",
    "subckt_0230",
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
        "round": 21,
        "purpose": "ultra-long real ATPG supervision for explicit horizon-aware return ranking",
        "accepted_benchmarks": BENCHMARKS,
        "excluded_benchmarks": sorted(forbidden),
        "eval_protocol": str(EVAL_PROTOCOL),
        "source_labels": str(LABELS),
        "target_circuits_in_training": False,
        "plan_budget": PLAN_BUDGET,
        "prefix_steps": PREFIX_STEPS,
        "actions_per_prefix": 15,
        "circuits": circuits,
    }
    path = ROOT / "ultralong_source_manifest.json"
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
