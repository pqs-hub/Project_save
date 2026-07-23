from __future__ import annotations

import csv
from pathlib import Path
import textwrap

from scripts.recover_itc99_exact_mapping import recover_exact
from scripts.remap_plan_to_original import remap
from tpi_jepa.bench import parse_bench
from tpi_jepa.graph import build_graph
from tpi_jepa.plan import enumerate_candidates, set_candidate_allowlist


def _write(path: Path, text: str) -> Path:
    path.write_text(textwrap.dedent(text).strip() + "\n")
    return path


def test_ordered_anchors_are_locally_proved_and_sinks_excluded(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "source.bench",
        """
        INPUT(a)
        INPUT(b)
        OUTPUT(g2)
        g1 = NAND(a, b)
        g2 = OR(g1, a)
        """,
    )
    # N0,N1 are PIs; N2,N3 are the two original-gate anchors.  Expansion
    # temporaries deliberately appear later and may feed lower-numbered anchors.
    deep = _write(
        tmp_path / "deep.bench",
        """
        INPUT(N0)
        INPUT(N1)
        OUTPUT(N3)
        N2 = NOT(N4)
        N3 = NOT(N6)
        N4 = AND(N0, N1)
        N5 = NOT(N2)
        N6 = AND(N5, N7)
        N7 = NOT(N0)
        """,
    )
    report = recover_exact(source, deep, tmp_path / "exact", global_patterns=128, seed=2)
    assert report["status"] == "exact"
    assert report["anchor_range"] == ["N2", "N3"]
    assert report["verified_local_truth_tables"] == 2
    assert report["paper_candidate_count"] == 1

    with (tmp_path / "exact" / "exact_node_mapping.tsv").open() as handle:
        rows = {row["deep_node"]: row for row in csv.DictReader(handle, delimiter="\t")}
    assert rows["N2"]["original_net"] == "g1"
    assert rows["N2"]["paper_candidate_legal"] == "True"
    assert rows["N3"]["original_net"] == "g2"
    assert rows["N3"]["paper_candidate_legal"] == "False"


def test_exact_mapping_is_accepted_by_plan_remapper(tmp_path: Path) -> None:
    mapping = _write(
        tmp_path / "exact.tsv",
        """
        deep_node\toriginal_net\toriginal_kind\toriginal_gate\toriginal_order\tdeep_gate\tboundary_exact\tlocal_truth_exact\tglobal_signature_exact\tdeep_is_sink\tpaper_candidate_legal
        N2\tg1\tGATE\tNAND\t0\tNOT\tTrue\tTrue\tTrue\tFalse\tTrue
        N3\tg2\tGATE\tOR\t1\tNOT\tTrue\tTrue\tTrue\tTrue\tFalse
        """,
    )
    plan = _write(tmp_path / "plan.csv", "step,node,type\n1,N2,observe\n2,N3,control0")
    report = remap(plan, mapping, tmp_path / "mapped.csv")
    assert report["counts"]["safe_insertable"] == 1
    with (tmp_path / "mapped.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["original_net"] == "g1"
    assert rows[0]["recovery_status"] == "structural_exact"
    assert rows[1]["original_net"] == ""


def test_candidate_allowlist_filters_heuristic_recall(tmp_path: Path) -> None:
    bench = _write(
        tmp_path / "graph.bench",
        """
        INPUT(a)
        INPUT(b)
        OUTPUT(out)
        n1 = AND(a, b)
        n2 = NOT(n1)
        out = BUFF(n2)
        """,
    )
    allowlist = _write(tmp_path / "allow.txt", "n2")
    graph = build_graph(parse_bench(bench))
    try:
        set_candidate_allowlist(allowlist)
        candidates = enumerate_candidates(graph, [], 9, strategy="heuristic_recall_pool")
    finally:
        set_candidate_allowlist(None)
    assert candidates
    assert {node for node, _ in candidates} == {"n2"}


def test_candidate_allowlist_filters_hard_fault_cluster(tmp_path: Path) -> None:
    bench = _write(
        tmp_path / "graph.bench",
        """
        INPUT(a)
        INPUT(b)
        OUTPUT(out)
        n1 = AND(a, b)
        n2 = NOT(n1)
        out = BUFF(n2)
        """,
    )
    allowlist = _write(tmp_path / "allow.txt", "n2")
    graph = build_graph(parse_bench(bench))
    try:
        set_candidate_allowlist(allowlist)
        candidates = enumerate_candidates(graph, [], 9, strategy="hard_fault_cluster")
    finally:
        set_candidate_allowlist(None)
    assert candidates
    assert {node for node, _ in candidates} == {"n2"}
