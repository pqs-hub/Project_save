"""Load TPI labels and convert rows into transition specifications."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from statistics import mean


DEFAULT_LABELS = Path(
    "/data4/pengqingsong/DFT/Dataset/atalanta_bist_lowtc_subckt_100k_labels/labels.csv"
)
LEGACY_DFT_ROOT = Path("/data3/pengqingsong/DFT")
DEFAULT_DFT_ROOT = Path(os.environ.get("DFT_ROOT", "/data4/pengqingsong/DFT"))
LOWTC_LABEL_ROOT = DEFAULT_DFT_ROOT / "Dataset/atalanta_bist_lowtc_subckt_100k_labels"
BENCH_ROOTS = [
    *(Path(root) for root in os.environ.get("TPI_BENCH_ROOT", "").split(os.pathsep) if root),
    LOWTC_LABEL_ROOT / "subcircuits",
    DEFAULT_DFT_ROOT / "Dataset/deeptpi_official_aig_bench_standard",
]
RAW_TO_ACTION = {"CP0": "control0", "CP1": "control1", "OP": "observe"}


@dataclass
class LabelRow:
    """One valid CSV label row after basic filtering and type conversion."""

    benchmark_id: str
    sequence_id: str
    step: int
    net: str
    raw_type: str
    insertion_sequence: str
    delta_fault_coverage: float
    delta_pattern: float | None
    undetected_node_csv_path: Path | None
    hard_fault_summary_path: Path | None
    undetected_fault_count: float | None
    undetected_sa0_count: float | None
    undetected_sa1_count: float | None


@dataclass
class TransitionSpec:
    """A lightweight transition before graph tensors are built."""

    benchmark_id: str
    bench_path: Path
    pre_actions: list[tuple[str, str]]
    post_actions: list[tuple[str, str]]
    action_node: str
    action_type: str
    delta_fault_coverage: float
    delta_pattern: float | None
    step: int
    sequence_id: str
    pre_undetected_node_csv_path: Path | None
    post_undetected_node_csv_path: Path | None
    pre_hard_fault_summary_path: Path | None
    post_hard_fault_summary_path: Path | None
    pre_undetected_fault_count: float | None
    post_undetected_fault_count: float | None
    pre_undetected_sa0_count: float | None
    post_undetected_sa0_count: float | None
    pre_undetected_sa1_count: float | None
    post_undetected_sa1_count: float | None


def _to_float(value: str | None) -> float | None:
    """Convert a CSV number field to float, preserving missing values."""

    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    return float(value)


def _canonical_action(raw_type: str) -> str:
    """Map raw label action names to canonical model action names."""

    if raw_type not in RAW_TO_ACTION:
        raise ValueError(f"Unsupported raw action type: {raw_type!r}")
    return RAW_TO_ACTION[raw_type]


def load_labels(labels_csv: str | Path = DEFAULT_LABELS) -> list[LabelRow]:
    """Read `labels.csv` and keep valid non-baseline action rows."""

    rows: list[LabelRow] = []
    with Path(labels_csv).open(newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            if raw.get("status") != "ok":
                continue
            raw_type = (raw.get("type") or "").strip()
            if raw_type not in RAW_TO_ACTION:
                continue
            step_text = (raw.get("step") or "").strip()
            if not step_text:
                continue
            step = int(step_text)
            if step < 1:
                continue
            net = (raw.get("net") or "").strip()
            if not net:
                continue
            delta_fc = _to_float(raw.get("delta_test_coverage"))
            if delta_fc is None:
                delta_fc = _to_float(raw.get("delta_fault_coverage"))
            if delta_fc is None:
                continue
            delta_eff = _to_float(raw.get("delta_effective_pattern_count"))
            delta_pat = _to_float(raw.get("delta_pattern_count"))
            rows.append(
                LabelRow(
                    benchmark_id=(raw.get("benchmark_id") or "").strip(),
                    sequence_id=(raw.get("sequence_id") or "").strip(),
                    step=step,
                    net=net,
                    raw_type=raw_type,
                    insertion_sequence=raw.get("insertion_sequence") or "[]",
                    delta_fault_coverage=delta_fc,
                    delta_pattern=delta_eff if delta_eff is not None else delta_pat,
                    undetected_node_csv_path=_to_path(raw.get("undetected_node_csv_path")),
                    hard_fault_summary_path=_to_path(raw.get("hard_fault_summary_path")),
                    undetected_fault_count=_to_float(raw.get("undetected_fault_count")),
                    undetected_sa0_count=_to_float(raw.get("undetected_sa0_count")),
                    undetected_sa1_count=_to_float(raw.get("undetected_sa1_count")),
                )
            )
    return rows


def _to_path(value: str | None) -> Path | None:
    """Convert optional CSV path text to a Path."""

    if value is None:
        return None
    value = value.strip()
    return _remap_legacy_path(Path(value)) if value else None


def _remap_legacy_path(path: Path) -> Path:
    """Map absolute paths from the old server to this server's DFT root."""

    try:
        rel = path.relative_to(LEGACY_DFT_ROOT)
    except ValueError:
        return path
    remapped = DEFAULT_DFT_ROOT / rel
    if remapped.exists():
        return remapped
    parts = rel.parts
    for index, part in enumerate(parts):
        if re.fullmatch(r"subckt_\d+", part):
            compact = LOWTC_LABEL_ROOT.joinpath(*parts[index:])
            if compact.exists():
                return compact
            break
    return remapped


def _previous_state_path(path: Path | None, step: int) -> Path | None:
    """Infer the previous state's sibling artifact path for a sequence step."""

    if path is None:
        return None
    if step <= 1:
        return path.parent.parent.parent / "baseline" / path.name
    return path.parent.parent / f"step_{step - 1:04d}" / path.name


def find_bench_path(benchmark_id: str) -> Path:
    """Find the original BENCH file for one benchmark id."""

    for root in BENCH_ROOTS:
        path = root / f"{benchmark_id}.bench"
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find BENCH for benchmark_id={benchmark_id!r}")


def _parse_sequence(text: str) -> list[tuple[str, str]]:
    """Parse insertion_sequence JSON into canonical `(net, action_type)` pairs."""

    data = json.loads(text or "[]")
    actions: list[tuple[str, str]] = []
    for item in data:
        net = str(item.get("net", "")).strip()
        raw_type = str(item.get("type", "")).strip()
        if not net or raw_type not in RAW_TO_ACTION:
            continue
        actions.append((net, _canonical_action(raw_type)))
    return actions


def row_to_transition(row: LabelRow) -> TransitionSpec:
    """Convert one label row into pre-state, action, and post-state actions."""

    sequence = _parse_sequence(row.insertion_sequence)
    current = (row.net, _canonical_action(row.raw_type))
    if len(sequence) >= row.step:
        pre_actions = sequence[: row.step - 1]
        action = sequence[row.step - 1]
        post_actions = sequence[: row.step]
    else:
        pre_actions = sequence
        action = current
        post_actions = sequence + [current]
    if action[0] != current[0] or action[1] != current[1]:
        action = current
        post_actions = pre_actions + [current]

    return TransitionSpec(
        benchmark_id=row.benchmark_id,
        bench_path=find_bench_path(row.benchmark_id),
        pre_actions=pre_actions,
        post_actions=post_actions,
        action_node=action[0],
        action_type=action[1],
        delta_fault_coverage=row.delta_fault_coverage,
        delta_pattern=row.delta_pattern,
        step=row.step,
        sequence_id=row.sequence_id,
        pre_undetected_node_csv_path=_previous_state_path(row.undetected_node_csv_path, row.step),
        post_undetected_node_csv_path=row.undetected_node_csv_path,
        pre_hard_fault_summary_path=_previous_state_path(row.hard_fault_summary_path, row.step),
        post_hard_fault_summary_path=row.hard_fault_summary_path,
        pre_undetected_fault_count=None,
        post_undetected_fault_count=row.undetected_fault_count,
        pre_undetected_sa0_count=None,
        post_undetected_sa0_count=row.undetected_sa0_count,
        pre_undetected_sa1_count=None,
        post_undetected_sa1_count=row.undetected_sa1_count,
    )


def _main() -> None:
    """CLI summary used by the step-by-step validation plan."""

    import argparse
    from collections import Counter

    parser = argparse.ArgumentParser()
    parser.add_argument("labels", nargs="?", default=str(DEFAULT_LABELS))
    args = parser.parse_args()

    rows = load_labels(args.labels)
    bench_ids = sorted({row.benchmark_id for row in rows})
    type_counts = Counter(row.raw_type for row in rows)
    found = []
    missing = []
    for bench_id in bench_ids:
        try:
            find_bench_path(bench_id)
            found.append(bench_id)
        except FileNotFoundError:
            missing.append(bench_id)

    pattern_values = [row.delta_pattern for row in rows if row.delta_pattern is not None]
    pattern_valid = bool(pattern_values) and len(set(pattern_values)) > 1
    print(f"valid_rows={len(rows)}")
    print(f"benchmarks={len(bench_ids)}")
    print("type_counts=" + ",".join(f"{k}:{type_counts[k]}" for k in sorted(type_counts)))
    print(f"bench_found={len(found)}")
    print(f"bench_missing={len(missing)}")
    if missing:
        print("missing_ids=" + ",".join(missing))
    print(f"pattern_target_present={bool(pattern_values)}")
    print(f"pattern_target_valid={pattern_valid}")
    if rows:
        print(f"delta_fc_mean={mean(row.delta_fault_coverage for row in rows):.6f}")


if __name__ == "__main__":
    _main()
