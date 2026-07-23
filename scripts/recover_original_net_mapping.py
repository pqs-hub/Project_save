"""Recover DeepTPI ``N<id>`` nodes to public source-netlist names.

Recovery uses deterministic bit-parallel functional fingerprints.  A mapping is
accepted only when the DeepTPI node and a source gate output agree on every
simulation bit.  Multiple source nets with the same function are reported as
ambiguous.  Restored BUFF nodes are conservatively classified as synthetic
fanout branches and are never marked safe for insertion in the source netlist.

This is a recovery tool, not a proof of Boolean equivalence.  Increase
``--patterns`` or run a formal follow-up before making a silicon-facing claim.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Iterable


INPUT_RE = re.compile(r"^INPUT\(([^)]+)\)$", re.IGNORECASE)
OUTPUT_RE = re.compile(r"^OUTPUT\(([^)]+)\)$", re.IGNORECASE)
ASSIGN_RE = re.compile(r"^([^=\s]+)\s*=\s*([A-Za-z0-9_]+)\((.*)\)$")


@dataclass(frozen=True)
class Gate:
    output: str
    op: str
    fanins: tuple[str, ...]


@dataclass
class LogicNetwork:
    inputs: list[str]
    outputs: list[str]
    gates: list[Gate]
    source_format: str


def _logical_lines(path: Path) -> list[str]:
    lines: list[str] = []
    current = ""
    for raw in path.read_text(errors="replace").splitlines():
        text = raw.split("#", 1)[0].strip()
        if not text:
            continue
        if current:
            current += " " + text
        else:
            current = text
        if current.endswith("\\"):
            current = current[:-1].rstrip()
            continue
        lines.append(current)
        current = ""
    if current:
        lines.append(current)
    return lines


def parse_bench(path: Path) -> LogicNetwork:
    inputs: list[str] = []
    outputs: list[str] = []
    gates: list[Gate] = []
    for line in _logical_lines(path):
        match = INPUT_RE.match(line)
        if match:
            inputs.append(match.group(1).strip())
            continue
        match = OUTPUT_RE.match(line)
        if match:
            outputs.append(match.group(1).strip())
            continue
        match = ASSIGN_RE.match(line)
        if not match:
            raise ValueError(f"{path}: unsupported BENCH line: {line}")
        output, op, raw_fanins = match.groups()
        fanins = tuple(item.strip() for item in raw_fanins.split(",") if item.strip())
        gates.append(Gate(output.strip(), op.upper(), fanins))
    return LogicNetwork(inputs, outputs, gates, "bench")


def parse_blif(path: Path) -> LogicNetwork:
    lines = _logical_lines(path)
    inputs: list[str] = []
    outputs: list[str] = []
    gates: list[Gate] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith(".inputs "):
            inputs.extend(line.split()[1:])
        elif line.startswith(".outputs "):
            outputs.extend(line.split()[1:])
        elif line.startswith(".names"):
            fields = line.split()[1:]
            if not fields:
                raise ValueError(f"{path}: empty .names declaration")
            fanins = fields[:-1]
            output = fields[-1]
            table: list[str] = []
            cursor = index + 1
            while cursor < len(lines) and not lines[cursor].startswith("."):
                table.append(lines[cursor])
                cursor += 1
            if len(table) > 1:
                raise ValueError(f"{path}: multi-cube .names is not supported at {output}")
            if not table:
                gates.append(Gate(output, "CONST0", ()))
            elif not fanins:
                gates.append(Gate(output, "CONST1" if table[0].split()[-1] == "1" else "CONST0", ()))
            else:
                parts = table[0].split()
                pattern = parts[0]
                value = parts[1] if len(parts) > 1 else "1"
                if len(pattern) != len(fanins) or any(bit not in "01" for bit in pattern):
                    raise ValueError(f"{path}: unsupported .names cube at {output}: {table[0]}")
                encoded = tuple(f"{name}:{bit}" for name, bit in zip(fanins, pattern))
                gates.append(Gate(output, "CUBE1" if value == "1" else "CUBE0", encoded))
            index = cursor - 1
        index += 1
    return LogicNetwork(inputs, outputs, gates, "blif")


def load_network(path: Path) -> LogicNetwork:
    if path.suffix.lower() == ".blif":
        return parse_blif(path)
    return parse_bench(path)


def topological_gates(network: LogicNetwork) -> list[Gate]:
    producers = {gate.output: gate for gate in network.gates}
    indegree: dict[str, int] = {}
    consumers: dict[str, list[str]] = defaultdict(list)
    for gate in network.gates:
        dependencies: set[str] = set()
        for item in gate.fanins:
            name = item.rsplit(":", 1)[0] if gate.op.startswith("CUBE") else item
            if name in producers:
                dependencies.add(name)
        for name in dependencies:
            consumers[name].append(gate.output)
        indegree[gate.output] = len(dependencies)
    queue = deque(gate.output for gate in network.gates if indegree[gate.output] == 0)
    ordered: list[Gate] = []
    while queue:
        output = queue.popleft()
        ordered.append(producers[output])
        for consumer in consumers.get(output, []):
            indegree[consumer] -= 1
            if indegree[consumer] == 0:
                queue.append(consumer)
    if len(ordered) != len(network.gates):
        unresolved = [name for name, degree in indegree.items() if degree > 0][:10]
        raise ValueError(f"network is cyclic or has unresolved dependencies: {unresolved}")
    return ordered


def _reduce_and(values: Iterable[int], mask: int) -> int:
    result = mask
    for value in values:
        result &= value
    return result


def _reduce_or(values: Iterable[int]) -> int:
    result = 0
    for value in values:
        result |= value
    return result


def _reduce_xor(values: Iterable[int]) -> int:
    result = 0
    for value in values:
        result ^= value
    return result


def evaluate_gate(gate: Gate, values: dict[str, int], mask: int) -> int:
    op = gate.op
    if op in {"CUBE1", "CUBE0"}:
        terms = []
        for encoded in gate.fanins:
            name, bit = encoded.rsplit(":", 1)
            terms.append(values[name] if bit == "1" else mask ^ values[name])
        cube = _reduce_and(terms, mask)
        return cube if op == "CUBE1" else mask ^ cube
    args = [values[name] for name in gate.fanins]
    if op in {"BUFF", "BUF", "WIRE"}:
        return args[0]
    if op in {"NOT", "INV"}:
        return mask ^ args[0]
    if op == "AND":
        return _reduce_and(args, mask)
    if op == "NAND":
        return mask ^ _reduce_and(args, mask)
    if op == "OR":
        return _reduce_or(args)
    if op == "NOR":
        return mask ^ _reduce_or(args)
    if op == "XOR":
        return _reduce_xor(args)
    if op == "XNOR":
        return mask ^ _reduce_xor(args)
    if op == "CONST0":
        return 0
    if op == "CONST1":
        return mask
    raise ValueError(f"unsupported gate op {op!r} at {gate.output}")


def simulate(network: LogicNetwork, input_values: list[int], patterns: int) -> dict[str, int]:
    if len(input_values) != len(network.inputs):
        raise ValueError(f"input vector mismatch: {len(input_values)} != {len(network.inputs)}")
    mask = (1 << patterns) - 1
    values = dict(zip(network.inputs, input_values))
    for gate in topological_gates(network):
        values[gate.output] = evaluate_gate(gate, values, mask) & mask
    return values


def signature_digest(value: int, patterns: int) -> str:
    data = value.to_bytes((patterns + 7) // 8, byteorder="little")
    return hashlib.sha256(data).hexdigest()[:16]


def is_special_net(name: str) -> bool:
    lower = name.lower()
    if lower in {"0", "1", "1'b0", "1'b1", "gnd", "vdd", "false", "true"}:
        return True
    return any(token in lower for token in ("clock", "clk", "reset", "rst", "scan", "test"))


def recover(
    deep_path: Path,
    source_path: Path,
    out_dir: Path,
    patterns: int,
    seed: int,
) -> dict:
    deep = load_network(deep_path)
    source = load_network(source_path)
    if len(deep.inputs) != len(source.inputs):
        report = {
            "status": "incompatible_primary_inputs",
            "deep_inputs": len(deep.inputs),
            "source_inputs": len(source.inputs),
            "deep_path": str(deep_path),
            "source_path": str(source_path),
            "message": "provide the exact source revision or an explicit PI mapping",
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return report

    generator = random.Random(seed)
    input_values = [generator.getrandbits(patterns) for _ in deep.inputs]
    deep_values = simulate(deep, input_values, patterns)
    source_values = simulate(source, input_values, patterns)
    source_outputs = set(source.outputs)
    deep_outputs = set(deep.outputs)
    source_gate_by_name = {gate.output: gate for gate in source.gates}
    deep_gate_by_name = {gate.output: gate for gate in deep.gates}

    source_output_values = {source_values[name] for name in source.outputs if name in source_values}
    matched_deep_outputs = sum(
        deep_values[name] in source_output_values for name in deep.outputs if name in deep_values
    )
    output_alignment_rate = matched_deep_outputs / max(1, len(deep.outputs))
    logic_alignment_verified = output_alignment_rate >= 0.90

    exact_index: dict[int, list[str]] = defaultdict(list)
    for gate in source.gates:
        exact_index[source_values[gate.output]].append(gate.output)

    mask = (1 << patterns) - 1
    rows: list[dict] = []
    counters: dict[str, int] = defaultdict(int)
    for node in deep.inputs + [gate.output for gate in deep.gates]:
        value = deep_values[node]
        exact = exact_index.get(value, [])
        inverted = exact_index.get(mask ^ value, [])
        gate = deep_gate_by_name.get(node)
        gate_type = "INPUT" if gate is None else gate.op
        input_index = deep.inputs.index(node) if gate is None else None
        source_net = (
            source.inputs[input_index]
            if input_index is not None
            else exact[0] if len(exact) == 1 else ""
        )
        source_gate = source_gate_by_name.get(source_net)
        source_insertable = bool(
            source_net
            and source_gate is not None
            and source_net not in source_outputs
            and not is_special_net(source_net)
        )
        if gate is None:
            status = "primary_input"
        elif len(exact) == 1:
            status = "functional_unique"
        elif len(exact) > 1:
            status = "functional_ambiguous"
        elif len(inverted) == 1:
            status = "inverted_unique"
        elif len(inverted) > 1:
            status = "inverted_ambiguous"
        else:
            status = "unmapped"
        synthetic_branch = gate_type in {"BUFF", "BUF", "WIRE"}
        safe_insertable = bool(
            status == "functional_unique"
            and logic_alignment_verified
            and source_insertable
            and not synthetic_branch
            and node not in deep_outputs
        )
        counters[status] += 1
        if safe_insertable:
            counters["safe_insertable"] += 1
        if synthetic_branch:
            counters["synthetic_branch"] += 1
        rows.append(
            {
                "deep_node": node,
                "deep_gate": gate_type,
                "deep_is_output": node in deep_outputs,
                "status": status,
                "source_net": source_net,
                "source_gate": source_gate.op if source_gate else "",
                "source_is_output": source_net in source_outputs if source_net else False,
                "source_candidate_count": len(exact),
                "inverted_candidate_count": len(inverted),
                "synthetic_branch": synthetic_branch,
                "safe_insertable": safe_insertable,
                "signature": signature_digest(value, patterns),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with (out_dir / "node_mapping.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    safe_nodes_path = out_dir / "safe_deep_nodes.txt"
    safe_nodes_path.write_text(
        "".join(f"{row['deep_node']}\n" for row in rows if row["safe_insertable"])
    )

    deep_gate_count = len(deep.gates)
    report = {
        "version": "1",
        "status": "complete" if logic_alignment_verified else "unverified_input_alignment",
        "method": "deterministic_bit_parallel_functional_fingerprint",
        "patterns": patterns,
        "seed": seed,
        "deep_path": str(deep_path),
        "source_path": str(source_path),
        "deep_format": deep.source_format,
        "source_format": source.source_format,
        "deep_inputs": len(deep.inputs),
        "source_inputs": len(source.inputs),
        "deep_gates": deep_gate_count,
        "source_gates": len(source.gates),
        "deep_outputs": len(deep.outputs),
        "matched_deep_outputs": matched_deep_outputs,
        "output_alignment_rate": output_alignment_rate,
        "logic_alignment_verified": logic_alignment_verified,
        "counts": dict(sorted(counters.items())),
        "functional_unique_rate": counters["functional_unique"] / max(1, deep_gate_count),
        "safe_insertable_rate": counters["safe_insertable"] / max(1, deep_gate_count),
        "mapping_tsv": str(out_dir / "node_mapping.tsv"),
        "safe_nodes": str(safe_nodes_path),
        "limitations": [
            "simulation fingerprints are not a formal equivalence proof",
            "BUFF nodes are conservatively rejected as synthetic fanout branches",
            "ambiguous same-function source nets are not selected automatically",
            "safe_insertable is disabled unless at least 90% of Deep outputs align with source outputs",
        ],
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deep-bench", type=Path, required=True)
    parser.add_argument("--source-netlist", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--patterns", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.patterns < 64:
        raise SystemExit("--patterns must be at least 64")
    report = recover(args.deep_bench, args.source_netlist, args.out_dir, args.patterns, args.seed)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
