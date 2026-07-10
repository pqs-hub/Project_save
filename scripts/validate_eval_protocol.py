"""Validate fixed evaluation protocols and result files against their budget contract."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def normalize_budget_entry(value: Any) -> dict[str, int]:
    if isinstance(value, dict):
        if "budget" not in value:
            raise ValueError(f"budget object is missing budget: {value}")
        entry = {"budget": int(value["budget"])}
        for key in ("logic_gates", "table_gates", "pis"):
            if value.get(key) not in (None, ""):
                entry[key] = int(value[key])
        return entry
    return {"budget": int(value)}


def protocol_contract(protocol: dict[str, Any]) -> dict[str, dict[str, int]]:
    raw = protocol.get("benchmark_budgets")
    if not isinstance(raw, dict) or not raw:
        raise ValueError("protocol must define non-empty benchmark_budgets")
    return {str(key): normalize_budget_entry(value) for key, value in raw.items()}


def validate_protocol(protocol: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    benchmarks = [str(item) for item in protocol.get("benchmarks", [])]
    if not benchmarks:
        errors.append("protocol.benchmarks is empty")
    contract = protocol_contract(protocol)

    for benchmark in benchmarks:
        if benchmark not in contract:
            errors.append(f"{benchmark}: missing benchmark_budgets entry")

    for benchmark, item in contract.items():
        budget = item["budget"]
        if budget <= 0:
            errors.append(f"{benchmark}: budget must be positive, got {budget}")
        if "table_gates" in item:
            expected = item["table_gates"] // 100
            if budget != expected:
                errors.append(f"{benchmark}: budget {budget} != floor(table_gates/100) {expected}")
        if "logic_gates" in item and "pis" in item and "table_gates" in item:
            expected_table_gates = item["logic_gates"] + item["pis"]
            if item["table_gates"] != expected_table_gates:
                errors.append(
                    f"{benchmark}: table_gates {item['table_gates']} != "
                    f"logic_gates + pis {expected_table_gates}"
                )

    if protocol.get("strict_benchmark_budgets") and not contract:
        errors.append("strict_benchmark_budgets is true but no contract was found")
    return errors


def read_result_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="") as f:
            rows.extend(dict(row) for row in csv.DictReader(f, delimiter="\t"))
    return rows


def validate_results(protocol: dict[str, Any], paths: list[Path]) -> list[str]:
    errors: list[str] = []
    contract = protocol_contract(protocol)
    expected_patterns = str(protocol.get("patterns", ""))
    expected_seed = str(protocol.get("seed", ""))

    for path in paths:
        rows = read_result_rows([path])
        if not rows:
            errors.append(f"{path}: no result rows")
            continue
        for row_index, row in enumerate(rows, start=2):
            benchmark = row.get("benchmark_id", "")
            if benchmark not in contract:
                errors.append(f"{path}:{row_index}: unexpected benchmark_id {benchmark!r}")
                continue
            expected = contract[benchmark]
            if row.get("budget") not in ("", None) and int(float(row["budget"])) != expected["budget"]:
                errors.append(
                    f"{path}:{row_index}: {benchmark} budget {row['budget']} != {expected['budget']}"
                )
            if "logic_gates" in expected and row.get("logic_gates") not in ("", None):
                if int(float(row["logic_gates"])) != expected["logic_gates"]:
                    errors.append(
                        f"{path}:{row_index}: {benchmark} logic_gates "
                        f"{row['logic_gates']} != {expected['logic_gates']}"
                    )
            if expected_patterns and row.get("patterns") not in ("", None, expected_patterns):
                errors.append(
                    f"{path}:{row_index}: {benchmark} patterns {row.get('patterns')} != {expected_patterns}"
                )
            if expected_seed and row.get("seed") not in ("", None, expected_seed):
                errors.append(f"{path}:{row_index}: {benchmark} seed {row.get('seed')} != {expected_seed}")
    return errors


def expand_result_paths(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(path.glob("**/results.tsv")))
            summary = path / "summary.tsv"
            if summary.exists():
                expanded.append(summary)
        else:
            expanded.append(path)
    return list(dict.fromkeys(expanded))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a fixed eval protocol and optional result files.")
    parser.add_argument("--protocol", type=Path, default=Path("configs/eval_protocol_coverage_only.json"))
    parser.add_argument("--results", type=Path, nargs="*", default=[])
    args = parser.parse_args()

    protocol = load_json(args.protocol)
    errors = validate_protocol(protocol)
    result_paths = expand_result_paths(args.results)
    if result_paths:
        errors.extend(validate_results(protocol, result_paths))

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    print(f"ok protocol={args.protocol} results={len(result_paths)}")


if __name__ == "__main__":
    main()
