"""Helpers for fixed evaluation protocols and train/eval circuit isolation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Any


def parse_benchmark_list(value: Any) -> set[str]:
    """Parse a benchmark list from JSON values, CLI strings, or iterables."""

    if value in (None, ""):
        return set()
    if isinstance(value, str):
        return {item.strip() for item in value.split(",") if item.strip()}
    if isinstance(value, dict):
        ids: set[str] = set()
        for item in value.values():
            ids.update(parse_benchmark_list(item))
        return ids
    if isinstance(value, Iterable):
        return {str(item).strip() for item in value if str(item).strip()}
    return {str(value).strip()}


def eval_benchmarks_from_protocol(path: str | Path | None, include_auxiliary: bool = True) -> set[str]:
    """Load benchmark ids that must stay out of training for a protocol."""

    if not path:
        return set()
    protocol_path = Path(path)
    if not protocol_path.exists():
        raise FileNotFoundError(f"Evaluation protocol not found: {protocol_path}")
    protocol = json.loads(protocol_path.read_text())
    excluded = parse_benchmark_list(protocol.get("benchmarks"))
    aliases = protocol.get("benchmark_aliases")
    excluded.update(parse_benchmark_list(aliases))
    if isinstance(aliases, dict):
        excluded.update(parse_benchmark_list(aliases.keys()))
    for row in protocol.get("table_rows", []):
        if isinstance(row, dict) and row.get("benchmark_id"):
            excluded.add(str(row["benchmark_id"]).strip())
        if isinstance(row, dict) and row.get("circuit"):
            excluded.add(str(row["circuit"]).strip())
    if include_auxiliary:
        excluded.update(parse_benchmark_list(protocol.get("safety_benchmark")))
        excluded.update(parse_benchmark_list(protocol.get("development_benchmark")))
    return {item for item in excluded if item}


def excluded_benchmarks_from_config(config: dict) -> set[str]:
    """Return benchmark ids requested for train-set exclusion."""

    excluded = parse_benchmark_list(config.get("exclude_benchmarks"))
    excluded.update(
        eval_benchmarks_from_protocol(
            config.get("exclude_eval_protocol"),
            include_auxiliary=bool(config.get("exclude_protocol_auxiliary", True)),
        )
    )
    return excluded


def filter_rows_by_excluded_benchmarks(rows: list, excluded: set[str]) -> list:
    """Drop rows whose benchmark id is in `excluded`."""

    if not excluded:
        return rows
    return [row for row in rows if getattr(row, "benchmark_id", None) not in excluded]
