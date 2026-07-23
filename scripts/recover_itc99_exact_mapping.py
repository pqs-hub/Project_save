"""Recover exact DeepTPI node provenance for the five ITC'99 circuits.

DeepTPI's direct, no-optimization AIG conversion preserves a useful invariant:
P primary-input nodes are followed by G original-gate anchor nodes in the same
order as the source BENCH; all remaining nodes are AIG expansion temporaries or
fanout buffers.  This tool verifies that invariant rather than trusting it.

For every original gate it traces the corresponding DeepTPI anchor backwards
through temporary AND/NOT/BUFF nodes, checks that the boundary is exactly the
original fanin set, and exhaustively proves the local truth table.  The resulting
candidate list contains only verified original-gate anchors and excludes sinks.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import random

try:
    from scripts.recover_original_net_mapping import load_network, simulate
except ModuleNotFoundError:  # Direct execution: python scripts/recover_itc99_exact_mapping.py
    from recover_original_net_mapping import load_network, simulate


SUPPORTED_SOURCE_OPS = {"AND", "NAND", "OR", "NOR", "NOT"}
SUPPORTED_DEEP_OPS = {"AND", "NOT", "BUFF", "BUF", "WIRE"}


def _node_number(name: str) -> int:
    if not name.startswith("N") or not name[1:].isdigit():
        raise ValueError(f"expected DeepTPI N<integer> node, got {name!r}")
    return int(name[1:])


def _logic(op: str, args: list[int], mask: int) -> int:
    if op in {"BUFF", "BUF", "WIRE"}:
        return args[0]
    if op == "NOT":
        return mask ^ args[0]
    if op in {"AND", "NAND"}:
        value = mask
        for arg in args:
            value &= arg
        return value if op == "AND" else mask ^ value
    if op in {"OR", "NOR"}:
        value = 0
        for arg in args:
            value |= arg
        return value if op == "OR" else mask ^ value
    raise ValueError(f"unsupported operation {op!r}")


def _variable_vectors(count: int) -> tuple[list[int], int]:
    patterns = 1 << count
    mask = (1 << patterns) - 1
    vectors = []
    for variable in range(count):
        value = 0
        for pattern in range(patterns):
            if (pattern >> variable) & 1:
                value |= 1 << pattern
        vectors.append(value)
    return vectors, mask


def recover_exact(
    source_path: Path,
    deep_path: Path,
    out_dir: Path,
    global_patterns: int = 1024,
    seed: int = 2026,
) -> dict[str, object]:
    source = load_network(source_path)
    deep = load_network(deep_path)
    errors: list[str] = []

    bad_source_ops = sorted({gate.op for gate in source.gates} - SUPPORTED_SOURCE_OPS)
    bad_deep_ops = sorted({gate.op for gate in deep.gates} - SUPPORTED_DEEP_OPS)
    if bad_source_ops:
        errors.append(f"unsupported source operations: {bad_source_ops}")
    if bad_deep_ops:
        errors.append(f"unsupported DeepTPI operations: {bad_deep_ops}")
    if len(source.inputs) != len(deep.inputs):
        errors.append(f"PI count differs: source={len(source.inputs)} deep={len(deep.inputs)}")

    expected_deep_inputs = [f"N{index}" for index in range(len(source.inputs))]
    if deep.inputs != expected_deep_inputs:
        errors.append("DeepTPI PI nodes are not the contiguous N0..N(P-1) prefix")

    deep_gate_by_name = {gate.output: gate for gate in deep.gates}
    source_to_deep: dict[str, str] = {}
    rows: list[dict[str, object]] = []
    for index, source_input in enumerate(source.inputs):
        deep_node = f"N{index}"
        source_to_deep[source_input] = deep_node
        rows.append(
            {
                "deep_node": deep_node,
                "original_net": source_input,
                "original_kind": "PI",
                "original_gate": "INPUT",
                "original_order": index,
                "deep_gate": "INPUT",
                "boundary_exact": True,
                "local_truth_exact": True,
                "global_signature_exact": True,
                "deep_is_sink": deep_node in set(deep.outputs),
                "paper_candidate_legal": False,
            }
        )

    anchor_start = len(source.inputs)
    anchor_names: list[str] = []
    for gate_index, source_gate in enumerate(source.gates):
        deep_node = f"N{anchor_start + gate_index}"
        anchor_names.append(deep_node)
        source_to_deep[source_gate.output] = deep_node
        if deep_node not in deep_gate_by_name:
            errors.append(f"missing expected anchor {deep_node} for {source_gate.output}")

    boundary_nodes = set(expected_deep_inputs) | set(anchor_names)
    boundary_memo: dict[str, frozenset[str]] = {}

    def boundary_sources(node: str) -> frozenset[str]:
        if node in boundary_nodes:
            return frozenset({node})
        if node in boundary_memo:
            return boundary_memo[node]
        gate = deep_gate_by_name.get(node)
        if gate is None:
            raise ValueError(f"unknown DeepTPI node {node!r} in an anchor cone")
        result = frozenset().union(*(boundary_sources(fanin) for fanin in gate.fanins))
        boundary_memo[node] = result
        return result

    global_signature_exact: dict[str, bool] = {}
    if not errors:
        generator = random.Random(seed)
        input_values = [generator.getrandbits(global_patterns) for _ in source.inputs]
        source_values = simulate(source, input_values, global_patterns)
        deep_values = simulate(deep, input_values, global_patterns)
        for original_net, deep_node in source_to_deep.items():
            global_signature_exact[original_net] = source_values[original_net] == deep_values[deep_node]

    deep_sinks = set(deep.outputs)
    boundary_exact_count = 0
    truth_exact_count = 0
    signature_exact_count = 0
    legal_count = 0
    for gate_index, source_gate in enumerate(source.gates):
        deep_node = anchor_names[gate_index]
        deep_gate = deep_gate_by_name.get(deep_node)
        expected_boundary = frozenset(source_to_deep[fanin] for fanin in source_gate.fanins)
        actual_boundary = frozenset()
        boundary_exact = False
        local_truth_exact = False
        if deep_gate is not None:
            actual_boundary = frozenset().union(*(boundary_sources(fanin) for fanin in deep_gate.fanins))
            boundary_exact = actual_boundary == expected_boundary
            if boundary_exact:
                variable_vectors, mask = _variable_vectors(len(source_gate.fanins))
                boundary_values = {
                    source_to_deep[fanin]: variable_vectors[index]
                    for index, fanin in enumerate(source_gate.fanins)
                }
                memo: dict[str, int] = {}

                def evaluate_deep(node: str) -> int:
                    if node in boundary_values:
                        return boundary_values[node]
                    if node in boundary_nodes:
                        raise ValueError(
                            f"{deep_node} cone reached unexpected anchor {node}; "
                            f"expected {sorted(expected_boundary)}"
                        )
                    if node in memo:
                        return memo[node]
                    inner_gate = deep_gate_by_name[node]
                    value = _logic(
                        inner_gate.op,
                        [evaluate_deep(fanin) for fanin in inner_gate.fanins],
                        mask,
                    ) & mask
                    memo[node] = value
                    return value

                actual_value = _logic(
                    deep_gate.op,
                    [evaluate_deep(fanin) for fanin in deep_gate.fanins],
                    mask,
                ) & mask
                expected_value = _logic(source_gate.op, variable_vectors, mask) & mask
                local_truth_exact = actual_value == expected_value

        signature_exact = global_signature_exact.get(source_gate.output, False)
        is_sink = deep_node in deep_sinks
        legal = boundary_exact and local_truth_exact and signature_exact and not is_sink
        boundary_exact_count += int(boundary_exact)
        truth_exact_count += int(local_truth_exact)
        signature_exact_count += int(signature_exact)
        legal_count += int(legal)
        rows.append(
            {
                "deep_node": deep_node,
                "original_net": source_gate.output,
                "original_kind": "GATE",
                "original_gate": source_gate.op,
                "original_order": gate_index,
                "deep_gate": deep_gate.op if deep_gate else "",
                "boundary_exact": boundary_exact,
                "local_truth_exact": local_truth_exact,
                "global_signature_exact": signature_exact,
                "deep_is_sink": is_sink,
                "paper_candidate_legal": legal,
            }
        )

    if boundary_exact_count != len(source.gates):
        errors.append(f"boundary verification failed for {len(source.gates) - boundary_exact_count} gates")
    if truth_exact_count != len(source.gates):
        errors.append(f"local truth-table proof failed for {len(source.gates) - truth_exact_count} gates")
    if signature_exact_count != len(source.gates):
        errors.append(f"global signature cross-check failed for {len(source.gates) - signature_exact_count} gates")

    anchor_end = anchor_start + len(source.gates) - 1
    expected_anchor_set = {f"N{index}" for index in range(anchor_start, anchor_end + 1)}
    if set(anchor_names) != expected_anchor_set:
        errors.append("anchor range is not contiguous")

    out_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = out_dir / "exact_node_mapping.tsv"
    with mapping_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    candidate_rows = [row for row in rows if row["paper_candidate_legal"]]
    candidate_path = out_dir / "exact_candidates.tsv"
    with candidate_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["deep_node", "original_net", "original_gate", "original_order"],
            delimiter="\t",
        )
        writer.writeheader()
        for row in candidate_rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    (out_dir / "exact_candidate_nodes.txt").write_text(
        "".join(f"{row['deep_node']}\n" for row in candidate_rows)
    )
    fanout = Counter(fanin for gate in deep.gates for fanin in gate.fanins)
    candidate_cache_path = out_dir / "candidate_cache.json"
    candidate_cache_path.write_text(
        json.dumps(
            {
                "benchmark_id": source_path.stem,
                "bench_path": str(deep_path),
                "candidate_count": len(candidate_rows),
                "provenance": "DeepTPI original-gate anchors; exhaustive local proof",
                "candidates": [
                    {
                        "id": index,
                        "net": row["deep_node"],
                        "kind": "original_gate_anchor",
                        "driver": row["deep_gate"],
                        "fanout": fanout.get(str(row["deep_node"]), 0),
                        "original_net": row["original_net"],
                        "original_gate": row["original_gate"],
                    }
                    for index, row in enumerate(candidate_rows)
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    deep_type_counts = Counter(gate.op for gate in deep.gates)
    report: dict[str, object] = {
        "version": "1",
        "status": "exact" if not errors else "failed",
        "method": "ordered_anchor_provenance_with_exhaustive_local_truth_proof",
        "source_path": str(source_path),
        "deep_path": str(deep_path),
        "source_inputs": len(source.inputs),
        "source_gates": len(source.gates),
        "deep_nodes": len(deep.inputs) + len(deep.gates),
        "deep_gate_counts": dict(sorted(deep_type_counts.items())),
        "anchor_range": [f"N{anchor_start}", f"N{anchor_end}"],
        "verified_boundaries": boundary_exact_count,
        "verified_local_truth_tables": truth_exact_count,
        "verified_global_signatures": signature_exact_count,
        "paper_candidate_count": legal_count,
        "excluded_anchor_sinks": len(source.gates) - legal_count,
        "temporary_node_count": len(deep.gates) - len(source.gates),
        "mapping_tsv": str(mapping_path),
        "candidates_tsv": str(candidate_path),
        "candidate_nodes": str(out_dir / "exact_candidate_nodes.txt"),
        "candidate_cache": str(candidate_cache_path),
        "global_patterns": global_patterns,
        "seed": seed,
        "errors": errors,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bench", type=Path, required=True)
    parser.add_argument("--deep-bench", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--global-patterns", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    report = recover_exact(
        args.source_bench,
        args.deep_bench,
        args.out_dir,
        global_patterns=args.global_patterns,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "exact":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
