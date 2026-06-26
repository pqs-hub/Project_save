"""Build a benchmark-isolated subset of cumulative TPI sequence labels."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tpi_jepa.labels import DEFAULT_LABELS, RAW_TO_ACTION  # noqa: E402
from tpi_jepa.protocol import eval_benchmarks_from_protocol, parse_benchmark_list  # noqa: E402


def usable_action(row: dict[str, str]) -> bool:
    if row.get("status") != "ok":
        return False
    if (row.get("type") or "").strip() not in RAW_TO_ACTION:
        return False
    if not (row.get("net") or "").strip():
        return False
    try:
        return int((row.get("step") or "0").strip()) >= 1
    except ValueError:
        return False


def load_grouped_rows(labels_csv: Path, excluded: set[str]) -> tuple[list[str], dict[tuple[str, str], list[dict[str, str]]]]:
    with labels_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in reader:
            benchmark_id = (row.get("benchmark_id") or "").strip()
            if benchmark_id in excluded or not usable_action(row):
                continue
            sequence_id = (row.get("sequence_id") or "").strip()
            groups[(benchmark_id, sequence_id)].append(dict(row))

    clean_groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for key, rows in groups.items():
        rows.sort(key=lambda row: int(row.get("step") or 0))
        steps = [int(row.get("step") or 0) for row in rows]
        if steps == list(range(1, len(rows) + 1)):
            clean_groups[key] = rows
    return fieldnames, clean_groups


def sample_groups(
    groups: dict[tuple[str, str], list[dict[str, str]]],
    *,
    target_rows: int,
    seed: int,
) -> list[tuple[tuple[str, str], list[dict[str, str]]]]:
    by_bench: dict[str, list[tuple[tuple[str, str], list[dict[str, str]]]]] = defaultdict(list)
    for key, rows in groups.items():
        by_bench[key[0]].append((key, rows))

    rng = random.Random(seed)
    for items in by_bench.values():
        rng.shuffle(items)

    selected: list[tuple[tuple[str, str], list[dict[str, str]]]] = []
    selected_rows = 0
    bench_ids = sorted(by_bench)
    cursor = {bench: 0 for bench in bench_ids}
    while selected_rows < target_rows:
        progressed = False
        for bench in bench_ids:
            idx = cursor[bench]
            items = by_bench[bench]
            if idx >= len(items):
                continue
            key, rows = items[idx]
            if selected_rows + len(rows) > target_rows:
                continue
            selected.append((key, rows))
            selected_rows += len(rows)
            cursor[bench] += 1
            progressed = True
            if selected_rows >= target_rows:
                break
        if not progressed:
            break
    return selected


def remap_and_flatten(
    selected: list[tuple[tuple[str, str], list[dict[str, str]]]],
    *,
    source_label_csv: Path,
) -> list[dict[str, str]]:
    rows_out: list[dict[str, str]] = []
    for seq_index, ((benchmark_id, sequence_id), rows) in enumerate(selected):
        new_sequence_id = f"seq10k:{seq_index:05d}"
        for row in rows:
            new_row = dict(row)
            new_row["sequence_id"] = new_sequence_id
            new_row["source_sequence_id"] = sequence_id
            new_row["source_label_csv"] = str(source_label_csv)
            rows_out.append(new_row)
    rows_out.sort(key=lambda row: (row["benchmark_id"], row["sequence_id"], int(row.get("step") or 0)))
    return rows_out


def counts_by(rows: list[dict[str, str]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = row.get(field, "")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--target-rows", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--eval-protocol", default="configs/eval_protocol_coverage_only.json")
    parser.add_argument("--allow-eval-benchmarks", action="store_true")
    parser.add_argument("--extra-exclude", default="")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    excluded = parse_benchmark_list(args.extra_exclude)
    if not args.allow_eval_benchmarks:
        excluded.update(eval_benchmarks_from_protocol(args.eval_protocol))

    fieldnames, groups = load_grouped_rows(args.labels, excluded)
    selected = sample_groups(groups, target_rows=args.target_rows, seed=args.seed)
    rows_out = remap_and_flatten(selected, source_label_csv=args.labels)
    if len(rows_out) != args.target_rows:
        raise SystemExit(f"selected {len(rows_out)} rows, expected exactly {args.target_rows}")

    for field in ("source_sequence_id", "source_label_csv"):
        if field not in fieldnames:
            fieldnames.append(field)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_out:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    manifest: dict[str, Any] = {
        "source_labels": str(args.labels),
        "target_rows": args.target_rows,
        "selected_rows": len(rows_out),
        "selected_sequences": len(selected),
        "excluded_benchmarks": sorted(excluded),
        "bench_counts": counts_by(rows_out, "benchmark_id"),
        "type_counts": counts_by(rows_out, "type"),
        "out": str(args.out),
    }
    manifest_path = args.manifest or args.out.with_name("manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
