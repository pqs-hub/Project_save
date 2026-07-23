from __future__ import annotations

import csv
from pathlib import Path
import textwrap

from scripts.recover_original_net_mapping import recover
from scripts.remap_plan_to_original import remap


def _write(path: Path, text: str) -> Path:
    path.write_text(textwrap.dedent(text).strip() + "\n")
    return path


def test_recover_marks_only_unique_real_internal_gate_safe(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "source.bench",
        """
        INPUT(a)
        INPUT(b)
        OUTPUT(out)
        real_and = AND(a, b)
        out = NOT(real_and)
        """,
    )
    deep = _write(
        tmp_path / "deep.bench",
        """
        INPUT(N0)
        INPUT(N1)
        OUTPUT(N4)
        N2 = AND(N0, N1)
        N3 = BUFF(N2)
        N4 = NOT(N2)
        """,
    )

    report = recover(deep, source, tmp_path / "mapping", patterns=128, seed=7)
    assert report["status"] == "complete"
    assert report["logic_alignment_verified"] is True
    with (tmp_path / "mapping" / "node_mapping.tsv").open() as handle:
        rows = {row["deep_node"]: row for row in csv.DictReader(handle, delimiter="\t")}
    assert rows["N2"]["source_net"] == "real_and"
    assert rows["N2"]["safe_insertable"] == "True"
    assert rows["N3"]["synthetic_branch"] == "True"
    assert rows["N3"]["safe_insertable"] == "False"
    assert rows["N4"]["source_net"] == "out"
    assert rows["N4"]["safe_insertable"] == "False"


def test_recover_rejects_different_primary_input_count(tmp_path: Path) -> None:
    deep = _write(tmp_path / "deep.bench", "INPUT(N0)\nOUTPUT(N0)")
    source = _write(tmp_path / "source.bench", "INPUT(a)\nINPUT(b)\nOUTPUT(a)")
    report = recover(deep, source, tmp_path / "mapping", patterns=64, seed=1)
    assert report["status"] == "incompatible_primary_inputs"
    assert not (tmp_path / "mapping" / "node_mapping.tsv").exists()


def test_recover_disables_safe_mapping_when_pi_order_is_wrong(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "source.bench",
        """
        INPUT(a)
        INPUT(b)
        OUTPUT(out)
        inv_a = NOT(a)
        out = AND(inv_a, b)
        """,
    )
    deep = _write(
        tmp_path / "deep.bench",
        """
        INPUT(N0)
        INPUT(N1)
        OUTPUT(N3)
        N2 = NOT(N1)
        N3 = AND(N2, N0)
        """,
    )
    report = recover(deep, source, tmp_path / "mapping", patterns=128, seed=9)
    assert report["status"] == "unverified_input_alignment"
    assert report["matched_deep_outputs"] == 0
    assert report["counts"].get("safe_insertable", 0) == 0


def test_plan_remap_never_populates_unsafe_original_net(tmp_path: Path) -> None:
    mapping = _write(
        tmp_path / "node_mapping.tsv",
        """
        deep_node\tdeep_gate\tstatus\tsource_net\tsource_gate\tsynthetic_branch\tsafe_insertable
        N2\tAND\tfunctional_unique\treal_and\tAND\tFalse\tTrue
        N3\tBUFF\tfunctional_unique\treal_and\tAND\tTrue\tFalse
        """,
    )
    plan = _write(tmp_path / "plan.csv", "step,node,type\n1,N2,observe\n2,N3,control1")
    report = remap(plan, mapping, tmp_path / "recovered.csv")
    assert report["status"] == "contains_unsafe_nodes"
    with (tmp_path / "recovered.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["original_net"] == "real_and"
    assert rows[1]["original_net"] == ""
