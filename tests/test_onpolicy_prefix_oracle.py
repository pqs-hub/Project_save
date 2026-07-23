import csv
import json

import pytest

from scripts.collect_onpolicy_prefix_oracle import (
    _reduction,
    choose_candidates,
    load_plan_paths,
    read_plan,
)
from scripts.prepare_counterfactual_round5 import validate_prefix_oracle


def test_reduction_is_positive_when_hard_faults_drop():
    assert _reduction("11", 7) == 4
    assert _reduction(None, 7) is None


def test_read_plan_orders_steps_and_canonicalizes_actions(tmp_path):
    plan = tmp_path / "subckt_0002.csv"
    with plan.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "node", "type"])
        writer.writeheader()
        writer.writerow({"step": 2, "node": "n2", "type": "OP"})
        writer.writerow({"step": 1, "node": "n1", "type": "CP0"})

    assert read_plan(plan) == [("n1", "control0"), ("n2", "observe")]


def test_choose_candidates_retains_choice_types_and_pool_tail():
    pool = [
        ("n0", "observe"),
        ("n1", "observe"),
        ("n2", "control0"),
        ("n3", "control1"),
        ("n4", "observe"),
        ("n5", "observe"),
    ]

    candidates, ranks, in_pool = choose_candidates(pool, pool[4], limit=5)

    assert pool[4] in candidates
    assert {action_type for _, action_type in candidates} == {"control0", "control1", "observe"}
    assert pool[-1] in candidates
    assert ranks == sorted(ranks)
    assert in_pool is True


def test_choose_candidates_balances_action_types_when_pool_allows():
    pool = [
        (f"{action_type}_{index}", action_type)
        for index in range(6)
        for action_type in ("control0", "control1", "observe")
    ]

    candidates, _, _ = choose_candidates(pool, pool[0], limit=9)

    assert {action_type: sum(row[1] == action_type for row in candidates) for action_type in {
        "control0", "control1", "observe"
    }} == {"control0": 3, "control1": 3, "observe": 3}


def test_load_plan_paths_refuses_eval_protocol_leakage(tmp_path):
    plans = tmp_path / "plans"
    plans.mkdir()
    (plans / "heldout.csv").write_text("step,node,type\n1,n1,OP\n")
    training_manifest = tmp_path / "training.json"
    training_manifest.write_text(json.dumps({"accepted_benchmarks": ["heldout"]}))
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"benchmarks": ["heldout"]}))

    with pytest.raises(ValueError, match="refusing evaluation-protocol circuits"):
        load_plan_paths(plans, training_manifest, protocol)


def _write_prefix_oracle_fixture(tmp_path, *, patterns=300_000, source_plans=None):
    plans = source_plans or tmp_path / "plans"
    plans.mkdir(exist_ok=True)
    oracle = tmp_path / "oracle_actions.tsv"
    fields = [
        "benchmark_id",
        "state_id",
        "prefix_step",
        "action_key",
        "status",
        "source_plan_csv",
    ]
    with oracle.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for rank in range(9):
            writer.writerow(
                {
                    "benchmark_id": "subckt_0001",
                    "state_id": "prefix_0001",
                    "prefix_step": 1,
                    "action_key": f"n{rank}::observe",
                    "status": "ok",
                    "source_plan_csv": str(plans / "subckt_0001.csv"),
                }
            )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "patterns": patterns,
                "actions_per_prefix": 9,
                "prefix_steps": [1],
                "plans_dir": str(plans),
                "candidate_evaluations": 9,
                "state_count": 1,
            }
        )
    )
    return oracle, manifest, plans


def test_validate_prefix_oracle_accepts_matching_provenance(tmp_path):
    oracle, manifest, plans = _write_prefix_oracle_fixture(tmp_path)

    payload = validate_prefix_oracle(
        oracle,
        manifest,
        expected_plans_dir=plans,
        expected_prefix_steps=[1],
    )

    assert payload["patterns"] == 300_000


def test_validate_prefix_oracle_rejects_stale_pattern_budget(tmp_path):
    oracle, manifest, plans = _write_prefix_oracle_fixture(tmp_path, patterns=100_000)

    with pytest.raises(ValueError, match="patterns=100000 expected=300000"):
        validate_prefix_oracle(
            oracle,
            manifest,
            expected_plans_dir=plans,
            expected_prefix_steps=[1],
        )
