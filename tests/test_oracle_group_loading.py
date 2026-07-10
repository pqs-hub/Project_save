import csv

import pytest

from tpi_jepa.train import load_oracle_groups


def _write_oracle(path, rows):
    fields = [
        "benchmark_id",
        "state_id",
        "candidate_strategy",
        "candidate_rank",
        "node",
        "type",
        "oracle_delta_tc",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def test_load_oracle_groups_skips_forbidden_benchmarks(tmp_path) -> None:
    path = tmp_path / "oracle.tsv"
    _write_oracle(
        path,
        [
            {
                "benchmark_id": "subckt_0001",
                "state_id": "initial",
                "candidate_strategy": "cached_random",
                "candidate_rank": "1",
                "node": "forbidden",
                "type": "control0",
                "oracle_delta_tc": "0.1",
            },
            {
                "benchmark_id": "subckt_0002",
                "state_id": "initial",
                "candidate_strategy": "cached_random",
                "candidate_rank": "1",
                "node": "kept",
                "type": "control1",
                "oracle_delta_tc": "0.2",
            },
        ],
    )

    groups = load_oracle_groups(path, forbidden_benchmarks={"subckt_0001"})

    assert len(groups) == 1
    assert groups[0][0]["benchmark_id"] == "subckt_0002"


def test_load_oracle_groups_errors_when_filtering_removes_all_groups(tmp_path) -> None:
    path = tmp_path / "oracle.tsv"
    _write_oracle(
        path,
        [
            {
                "benchmark_id": "subckt_0001",
                "state_id": "initial",
                "candidate_strategy": "cached_random",
                "candidate_rank": "1",
                "node": "forbidden",
                "type": "control0",
                "oracle_delta_tc": "0.1",
            }
        ],
    )

    with pytest.raises(ValueError, match="after forbidden-benchmark filtering"):
        load_oracle_groups(path, forbidden_benchmarks={"subckt_0001"})
