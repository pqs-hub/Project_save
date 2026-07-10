"""Restore DeepTPI ITC22 npz graph circuits to simple BENCH files.

The restored files preserve the graph topology and DeepTPI gate-type integers.
Node names are synthetic (N0, N1, ...), because the npz does not store original
BENCH net names.
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np


DEFAULT_NAMES = [
    "b15_C",
    "b20_C",
    "b21_C",
    "b22_C",
    "i2c_aig",
    "max_aig",
    "b17_C",
    "mem_ctrl_aig",
]

GATE_TYPES = {
    1: "AND",
    2: "NOT",
    3: "BUFF",
}


def load_circuits(zip_path: Path, npz_member: str) -> dict[str, dict[str, np.ndarray]]:
    with ZipFile(zip_path) as zf:
        payload = zf.read(npz_member)
    with np.load(BytesIO(payload), allow_pickle=True) as data:
        return data["circuits"].item()


def restore_bench(name: str, circuit: dict[str, np.ndarray], out_path: Path) -> dict[str, int]:
    x = circuit["x"]
    edge_index = circuit["edge_index"]
    node_count = int(x.shape[0])
    gate_type = x[:, 1].astype(int)
    levels = x[:, 2].astype(int)

    fanins: list[list[int]] = [[] for _ in range(node_count)]
    out_degree = [0] * node_count
    for src, dst in edge_index:
        src_i = int(src)
        dst_i = int(dst)
        fanins[dst_i].append(src_i)
        out_degree[src_i] += 1

    pis = [idx for idx, kind in enumerate(gate_type) if kind == 0]
    pos = [idx for idx, degree in enumerate(out_degree) if degree == 0]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write(f"# Restored from DeepTPI ITC22 npz graph: {name}\n")
        f.write("# Synthetic node names: original BENCH net names are not stored in the npz.\n")
        f.write("# Gate type mapping: 0=PI, 1=AND, 2=NOT, 3=BUFF (DeepTPI ITC22 gate_types).\n\n")
        for idx in pis:
            f.write(f"INPUT(N{idx})\n")
        f.write("\n")
        for idx in pos:
            f.write(f"OUTPUT(N{idx})\n")
        f.write("\n")
        for idx in range(node_count):
            kind = int(gate_type[idx])
            if kind == 0:
                continue
            op = GATE_TYPES.get(kind)
            if op is None:
                raise ValueError(f"{name}: unsupported gate type {kind} at node {idx}")
            args = ", ".join(f"N{fanin}" for fanin in fanins[idx])
            f.write(f"N{idx} = {op}({args})\n")

    return {
        "nodes": node_count,
        "pis": len(pis),
        "pos": len(pos),
        "levels": int(levels.max()) if len(levels) else 0,
        "edges": int(edge_index.shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=Path("/data4/pengqingsong/DeepTPI-main.zip"))
    parser.add_argument(
        "--npz-member",
        default="DeepTPI-main/data/ITC22_dataset/test/benchmarks_circuits_graphs.npz",
    )
    parser.add_argument(
        "--names",
        default=",".join(DEFAULT_NAMES),
        help="Comma-separated circuit names to restore from the npz.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("autoresearch/deeptpi_table2_restored_bench"),
    )
    args = parser.parse_args()

    circuits = load_circuits(args.zip, args.npz_member)
    names = [item.strip() for item in args.names.split(",") if item.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in names:
        if name not in circuits:
            raise KeyError(f"{name!r} not found in {args.npz_member}")
        out_path = args.out_dir / f"{name}.bench"
        stats = restore_bench(name, circuits[name], out_path)
        rows.append((name, out_path, stats))

    print("name\tbench\tnodes\tPIs\tPOs\tlevels\tedges")
    for name, out_path, stats in rows:
        print(
            f"{name}\t{out_path}\t{stats['nodes']}\t{stats['pis']}\t"
            f"{stats['pos']}\t{stats['levels']}\t{stats['edges']}"
        )


if __name__ == "__main__":
    main()
