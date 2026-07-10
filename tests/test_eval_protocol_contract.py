import argparse
import csv
from pathlib import Path

from scripts.run_gmean_sweep import benchmark_budget_overrides
from scripts.validate_eval_protocol import load_json, validate_protocol, validate_results


def test_table2_protocol_budget_contract_is_valid():
    protocol = load_json(Path("configs/eval_protocol_coverage_only.json"))

    errors = validate_protocol(protocol)

    assert errors == []
    assert protocol["strict_benchmark_budgets"] is True


def test_budget_override_preserves_table2_metadata():
    args = argparse.Namespace(benchmark_budgets=None)
    protocol = load_json(Path("configs/eval_protocol_coverage_only.json"))

    overrides = benchmark_budget_overrides(args, protocol)

    assert overrides["iscas99__b21"]["budget"] == 628
    assert overrides["iscas99__b21"]["logic_gates"] == 62295
    assert overrides["iscas99__b21"]["logic_gates"] + 522 == 62817


def test_result_budget_drift_is_reported(tmp_path):
    protocol = load_json(Path("configs/eval_protocol_coverage_only.json"))
    result_path = tmp_path / "results.tsv"
    fields = ["benchmark_id", "logic_gates", "budget", "patterns", "seed"]
    with result_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerow(
            {
                "benchmark_id": "iscas99__b21",
                "logic_gates": "85858",
                "budget": "858",
                "patterns": "300000",
                "seed": "2026",
            }
        )

    errors = validate_results(protocol, [result_path])

    assert any("budget 858 != 628" in error for error in errors)
    assert any("logic_gates 85858 != 62295" in error for error in errors)
