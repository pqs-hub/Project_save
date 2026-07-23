"""Annotate a DeepTPI plan with recovered original-netlist signal names.

The output preserves every plan column and appends recovery metadata.  It does
not silently substitute unsafe nodes: ``original_net`` is populated only for
rows marked ``safe_insertable`` by the recovery step.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


RECOVERY_FIELDS = [
    "original_net",
    "recovery_status",
    "recovery_source_gate",
    "recovery_synthetic_branch",
    "safe_insertable",
]


def read_mapping(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    normalized: dict[str, dict[str, str]] = {}
    for row in rows:
        if "original_net" in row and "paper_candidate_legal" in row:
            row = dict(row)
            row.update(
                source_net=row["original_net"],
                status="structural_exact",
                source_gate=row.get("original_gate", ""),
                synthetic_branch="False",
                safe_insertable=row["paper_candidate_legal"],
            )
        normalized[row["deep_node"]] = row
    return normalized


def remap(plan_path: Path, mapping_path: Path, output_path: Path) -> dict[str, object]:
    mapping = read_mapping(mapping_path)
    delimiter = "\t" if plan_path.suffix.lower() == ".tsv" else ","
    with plan_path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames or "node" not in reader.fieldnames:
            raise ValueError(f"{plan_path}: expected a 'node' column")
        rows = list(reader)
        fieldnames = list(reader.fieldnames) + [name for name in RECOVERY_FIELDS if name not in reader.fieldnames]

    counts = {"rows": len(rows), "safe_insertable": 0, "unsafe": 0, "missing_mapping": 0}
    for row in rows:
        recovered = mapping.get(row["node"])
        if recovered is None:
            row.update(
                original_net="",
                recovery_status="missing_mapping",
                recovery_source_gate="",
                recovery_synthetic_branch="",
                safe_insertable="False",
            )
            counts["missing_mapping"] += 1
            counts["unsafe"] += 1
            continue
        safe = recovered["safe_insertable"].lower() == "true"
        row.update(
            original_net=recovered["source_net"] if safe else "",
            recovery_status=recovered["status"],
            recovery_source_gate=recovered["source_gate"],
            recovery_synthetic_branch=recovered["synthetic_branch"],
            safe_insertable=str(safe),
        )
        counts["safe_insertable" if safe else "unsafe"] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)
    report: dict[str, object] = {
        "status": "complete" if counts["unsafe"] == 0 else "contains_unsafe_nodes",
        "plan": str(plan_path),
        "mapping": str(mapping_path),
        "output": str(output_path),
        "counts": counts,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--require-all-safe", action="store_true")
    args = parser.parse_args()
    report = remap(args.plan, args.mapping, args.output)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_all_safe and report["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
