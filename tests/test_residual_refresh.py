from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch

from scripts.build_real_fault_priors import _selected_label_rows, build_priors
from tpi_jepa.features import _load_typed_real_fault_priors
from tpi_jepa.plan import _fault_polarity_strength, _typed_control_factors, read_prefix_plan_csv


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_read_prefix_plan_canonicalizes_and_limits(tmp_path: Path) -> None:
    path = tmp_path / "plan.csv"
    _write_csv(
        path,
        ["step", "net", "type", "sequence_score"],
        [
            {"step": 1, "net": "n1", "type": "CP0", "sequence_score": 1.0},
            {"step": 2, "net": "n2", "type": "OP", "sequence_score": 2.0},
            {"step": 3, "net": "n3", "type": "control1", "sequence_score": 3.0},
        ],
    )

    rows = read_prefix_plan_csv(path, max_steps=2)

    assert [(row["step"], row["node"], row["type"]) for row in rows] == [
        (1, "n1", "control0"),
        (2, "n2", "observe"),
    ]


def test_read_prefix_plan_rejects_duplicate_nodes(tmp_path: Path) -> None:
    path = tmp_path / "plan.csv"
    _write_csv(
        path,
        ["node", "type"],
        [
            {"node": "n1", "type": "control0"},
            {"node": "n1", "type": "observe"},
        ],
    )

    with pytest.raises(ValueError, match="more than once"):
        read_prefix_plan_csv(path)


def test_last_row_only_uses_last_nonbaseline_state(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    _write_csv(
        path,
        ["benchmark_id", "step", "fault_csv_path"],
        [
            {"benchmark_id": "b", "step": 0, "fault_csv_path": "baseline.csv"},
            {"benchmark_id": "b", "step": 64, "fault_csv_path": "step64.csv"},
            {"benchmark_id": "b", "step": 128, "fault_csv_path": "step128.csv"},
        ],
    )

    rows = _selected_label_rows(path, last_row_only=True)

    assert len(rows) == 1
    assert rows[0]["step"] == "128"
    assert rows[0]["fault_csv_path"] == "step128.csv"


def test_residual_prior_preserves_stuck_at_polarity(tmp_path: Path) -> None:
    faults = tmp_path / "faults.csv"
    _write_csv(
        faults,
        ["net", "fault_type", "sa_value", "status", "is_hard"],
        [
            {"net": "n1", "fault_type": "sa0", "sa_value": "0", "status": "undetected", "is_hard": 1},
            {"net": "n1", "fault_type": "sa1", "sa_value": "1", "status": "detected", "is_hard": 0},
            {"net": "n2", "fault_type": "", "sa_value": "1", "status": "undetected", "is_hard": 1},
        ],
    )
    labels = tmp_path / "labels.csv"
    _write_csv(
        labels,
        ["benchmark_id", "step", "fault_csv_path"],
        [{"benchmark_id": "bench", "step": 64, "fault_csv_path": str(faults)}],
    )

    rows = {row["net"]: row for row in build_priors([labels], last_row_only=True)}

    assert rows["n1"]["sa0_fault_count"] == 1
    assert rows["n1"]["sa1_fault_count"] == 1
    assert rows["n1"]["sa0_hard_fault_count"] == 1
    assert rows["n1"]["sa1_hard_fault_count"] == 0
    assert rows["n2"]["sa1_hard_fault_count"] == 1


def test_typed_prior_loader_reads_sa_counts(tmp_path: Path) -> None:
    path = tmp_path / "priors.csv"
    _write_csv(
        path,
        ["benchmark_id", "net", "sa0_hard_fault_count", "sa1_hard_fault_count"],
        [{"benchmark_id": "bench", "net": "n1", "sa0_hard_fault_count": 2, "sa1_hard_fault_count": 1}],
    )
    _load_typed_real_fault_priors.cache_clear()

    priors = _load_typed_real_fault_priors(str(path))

    assert priors["bench"]["n1"] == (2.0, 1.0)


def test_fault_polarity_maps_sa1_to_cp0_and_sa0_to_cp1() -> None:
    control0, control1 = _typed_control_factors(
        {
            "real_sa0": torch.tensor([1.0, 0.0, 1.0]),
            "real_sa1": torch.tensor([0.0, 1.0, 1.0]),
        },
        0.75,
    )

    assert control0.tolist() == pytest.approx([0.25, 1.75, 1.0])
    assert control1.tolist() == pytest.approx([1.75, 0.25, 1.0])


def test_fault_polarity_strength_rejects_invalid_value(monkeypatch) -> None:
    monkeypatch.setenv("TPI_HARD_CLUSTER_FAULT_POLARITY_ALPHA", "1.1")
    with pytest.raises(ValueError, match="between 0 and 1"):
        _fault_polarity_strength()
