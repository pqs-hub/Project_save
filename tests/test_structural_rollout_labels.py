import json
import random

from scripts.build_structural_rollout_labels import (
    select_trajectory_actions,
    trajectory_rows,
    unique_node_candidates,
)


def test_unique_node_candidates_keeps_first_ranked_action() -> None:
    candidates = [
        ("n1", "observe"),
        ("n1", "control0"),
        ("n2", "control1"),
    ]

    assert unique_node_candidates(candidates) == [("n1", "observe"), ("n2", "control1")]


def test_structural_trajectory_is_deterministic_and_unique() -> None:
    candidates = [(f"n{index}", "observe") for index in range(20)]

    first = select_trajectory_actions(
        candidates,
        length=8,
        pool_multiplier=2,
        rng=random.Random(17),
    )
    second = select_trajectory_actions(
        candidates,
        length=8,
        pool_multiplier=2,
        rng=random.Random(17),
    )

    assert first == second
    assert len({node for node, _ in first}) == 8
    assert all(int(node[1:]) < 16 for node, _ in first)


def test_trajectory_rows_store_exact_cumulative_prefixes() -> None:
    rows = trajectory_rows(
        "subckt_0001",
        "structural:0",
        [("n1", "control0"), ("n2", "observe")],
    )

    assert [row["step"] for row in rows] == [1, 2]
    assert rows[1]["type"] == "OP"
    assert json.loads(rows[1]["insertion_sequence"]) == [
        {"net": "n1", "type": "CP0"},
        {"net": "n2", "type": "OP"},
    ]
