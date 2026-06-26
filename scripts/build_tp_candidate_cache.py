"""Build per-benchmark caches of backend-valid insertable test-point candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TPI_EVAL_ROOT = Path("/data4/pengqingsong/DFT/Dataset")
if str(TPI_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(TPI_EVAL_ROOT))

from tpi_eval.bench import parse_bench as parse_eval_bench  # noqa: E402
from tpi_eval.candidates import candidate_file_payload, generate_candidates  # noqa: E402
from tpi_jepa.labels import find_bench_path  # noqa: E402


def parse_csv_values(text: str) -> list[str]:
    """Parse comma-separated command-line values."""

    return [item.strip() for item in text.split(",") if item.strip()]


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description="Build insertable TP candidate cache JSON files.")
    parser.add_argument("--benchmarks", required=True, help="Comma-separated benchmark ids.")
    parser.add_argument("--out-dir", required=True, help="Directory for {benchmark_id}.json cache files.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for benchmark_id in parse_csv_values(args.benchmarks):
        bench_path = find_bench_path(benchmark_id)
        circuit = parse_eval_bench(bench_path)
        candidates = generate_candidates(circuit)
        payload = candidate_file_payload(benchmark_id, candidates)
        payload["bench_path"] = str(bench_path)
        payload["source"] = "tpi_eval.candidates.generate_candidates"
        path = out_dir / f"{benchmark_id}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"{benchmark_id}\tcandidates={len(candidates)}\tpath={path}")


if __name__ == "__main__":
    main()
