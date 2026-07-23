import csv
from types import SimpleNamespace

import pytest
import torch

from tpi_jepa.train import (
    _oracle_prefix_actions,
    _oracle_sa_reduction_targets,
    _predict_oracle_group_scores,
    _sample_oracle_groups,
    load_oracle_groups,
)


def _write_oracle(path, rows):
    fields = [
        "benchmark_id",
        "state_id",
        "candidate_strategy",
        "candidate_rank",
        "node",
        "type",
        "oracle_delta_tc",
        "state_actions",
        "prefix_sequence",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _sampling_group(best_type: str, index: int) -> list[dict[str, str]]:
    alternatives = [action for action in ("control0", "control1", "observe") if action != best_type]
    return [
        {
            "benchmark_id": f"subckt_{index:04d}",
            "state_id": "initial",
            "candidate_strategy": "test",
            "type": best_type,
            "oracle_delta_tc": "0.2",
        },
        {
            "benchmark_id": f"subckt_{index:04d}",
            "state_id": "initial",
            "candidate_strategy": "test",
            "type": alternatives[0],
            "oracle_delta_tc": "0.1",
        },
    ]


def test_oracle_group_sampling_balances_real_best_action_types() -> None:
    groups = [
        *[_sampling_group("control0", index) for index in range(2)],
        *[_sampling_group("control1", index + 10) for index in range(4)],
        *[_sampling_group("observe", index + 20) for index in range(12)],
    ]

    sampled = _sample_oracle_groups(groups, 6, "best_type_balanced")
    best_types = [max(group, key=lambda row: float(row["oracle_delta_tc"]))["type"] for group in sampled]

    assert len(sampled) == 6
    assert len({id(group) for group in sampled}) == 6
    assert {action_type: best_types.count(action_type) for action_type in set(best_types)} == {
        "control0": 2,
        "control1": 2,
        "observe": 2,
    }


def test_oracle_group_sampling_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="oracle_group_sampling"):
        _sample_oracle_groups([_sampling_group("observe", 1)], 1, "unsupported")


def test_load_oracle_groups_skips_forbidden_benchmarks(tmp_path) -> None:
    path = tmp_path / "oracle.tsv"
    _write_oracle(
        path,
        [
            {
                "benchmark_id": "subckt_0001",
                "state_id": "initial",
                "candidate_strategy": "cached_random",
                "candidate_rank": "1",
                "node": "forbidden",
                "type": "control0",
                "oracle_delta_tc": "0.1",
            },
            {
                "benchmark_id": "subckt_0002",
                "state_id": "initial",
                "candidate_strategy": "cached_random",
                "candidate_rank": "1",
                "node": "kept",
                "type": "control1",
                "oracle_delta_tc": "0.2",
            },
        ],
    )

    groups = load_oracle_groups(path, forbidden_benchmarks={"subckt_0001"})

    assert len(groups) == 1
    assert groups[0][0]["benchmark_id"] == "subckt_0002"


def test_load_oracle_groups_errors_when_filtering_removes_all_groups(tmp_path) -> None:
    path = tmp_path / "oracle.tsv"
    _write_oracle(
        path,
        [
            {
                "benchmark_id": "subckt_0001",
                "state_id": "initial",
                "candidate_strategy": "cached_random",
                "candidate_rank": "1",
                "node": "forbidden",
                "type": "control0",
                "oracle_delta_tc": "0.1",
            }
        ],
    )

    with pytest.raises(ValueError, match="after forbidden-benchmark filtering"):
        load_oracle_groups(path, forbidden_benchmarks={"subckt_0001"})


def test_load_oracle_groups_accepts_shared_noninitial_prefix(tmp_path) -> None:
    path = tmp_path / "oracle.tsv"
    shared = '[{"net":"n0","type":"CP0"},["n1","observe"]]'
    _write_oracle(
        path,
        [
            {
                "benchmark_id": "subckt_0002",
                "state_id": "prefix:2",
                "candidate_strategy": "onpolicy_counterfactual",
                "candidate_rank": str(rank),
                "node": node,
                "type": action_type,
                "oracle_delta_tc": delta,
                "state_actions": shared,
            }
            for rank, node, action_type, delta in [
                (1, "n2", "control1", "0.03"),
                (2, "n3", "observe", "0.01"),
            ]
        ],
    )

    groups = load_oracle_groups(path)

    assert len(groups) == 1
    assert _oracle_prefix_actions(groups[0]) == [("n0", "control0"), ("n1", "observe")]


def test_load_oracle_groups_rejects_noninitial_state_without_prefix(tmp_path) -> None:
    path = tmp_path / "oracle.tsv"
    _write_oracle(
        path,
        [
            {
                "benchmark_id": "subckt_0002",
                "state_id": "prefix:1",
                "candidate_strategy": "onpolicy_counterfactual",
                "candidate_rank": "1",
                "node": "n2",
                "type": "control1",
                "oracle_delta_tc": "0.03",
            }
        ],
    )

    with pytest.raises(ValueError, match="requires JSON state_actions"):
        load_oracle_groups(path)


def test_load_oracle_groups_rejects_mixed_prefixes_within_group(tmp_path) -> None:
    path = tmp_path / "oracle.tsv"
    common = {
        "benchmark_id": "subckt_0002",
        "state_id": "prefix:1",
        "candidate_strategy": "onpolicy_counterfactual",
        "type": "observe",
    }
    _write_oracle(
        path,
        [
            {
                **common,
                "candidate_rank": "1",
                "node": "n2",
                "oracle_delta_tc": "0.03",
                "prefix_sequence": '[["n0","control0"]]',
            },
            {
                **common,
                "candidate_rank": "2",
                "node": "n3",
                "oracle_delta_tc": "0.01",
                "prefix_sequence": '[["n1","control1"]]',
            },
        ],
    )

    with pytest.raises(ValueError, match="inconsistent state-action prefixes"):
        load_oracle_groups(path)


def test_predict_oracle_group_replays_prefix_before_scoring_candidates() -> None:
    graph = SimpleNamespace(
        num_nodes=4,
        node_names=["n0", "n1", "n2", "n3"],
        edge_src=torch.zeros(0, dtype=torch.long),
        edge_dst=torch.zeros(0, dtype=torch.long),
        gate_type_ids=torch.zeros(4, dtype=torch.long),
        fanin_lists=[[], [], [], []],
        fanout_lists=[[], [], [], []],
    )

    class DummyModel:
        def __init__(self):
            self.calls = []

        def online_encoder(self, x, edge_src, edge_dst, gate_type_ids):
            return x

        def predict_from_latent(
            self,
            z,
            node_id,
            action_type_id,
            relation,
            include_aux_heads=False,
            sequence_step=0,
        ):
            self.calls.append((node_id, action_type_id, sequence_step))
            scalar = z.new_tensor(float(node_id + action_type_id))
            return {
                "z_pred": z + float(action_type_id + 1),
                "q_pred": scalar,
                "reward_pred": scalar,
                "return_pred": scalar,
                "hard_reduction_pred": torch.stack([scalar, scalar + 1.0, scalar + 2.0]),
                "typed_marginal_pred": scalar,
                "typed_return_pred": scalar,
                "typed_sa_reduction_pred": torch.stack([scalar + 3.0, scalar + 4.0]),
            }

    prefix = '[["n0","CP0"],["n1","OP"]]'
    group = [
        {
            "benchmark_id": "subckt_0002",
            "state_id": "prefix:2",
            "candidate_strategy": "onpolicy_counterfactual",
            "candidate_rank": "1",
            "node": "n2",
            "type": "CP1",
            "oracle_delta_tc": "0.03",
            "prefix_sequence": prefix,
        },
        {
            "benchmark_id": "subckt_0002",
            "state_id": "prefix:2",
            "candidate_strategy": "onpolicy_counterfactual",
            "candidate_rank": "2",
            "node": "n3",
            "type": "observe",
            "oracle_delta_tc": "0.01",
            "prefix_sequence": prefix,
        },
    ]
    model = DummyModel()

    prefix_latent_cache = {}
    scores = _predict_oracle_group_scores(
        model,
        {"relation_mode": "basic", "relation_depth": 2, "coverage_scale": 100.0},
        group,
        {"subckt_0002": graph},
        {"subckt_0002": torch.zeros((4, 2))},
        torch.device("cpu"),
        prefix_latent_cache,
    )

    assert model.calls == [(0, 0, 0), (1, 2, 1), (2, 1, 2), (3, 2, 2)]
    assert scores["typed_marginal_pred"].shape == (2,)
    assert scores["typed_sa0_reduction_pred"].shape == (2,)
    assert scores["typed_sa1_reduction_pred"].shape == (2,)
    cached_scores = _predict_oracle_group_scores(
        model,
        {"relation_mode": "basic", "relation_depth": 2, "coverage_scale": 100.0},
        group,
        {"subckt_0002": graph},
        {"subckt_0002": torch.zeros((4, 2))},
        torch.device("cpu"),
        prefix_latent_cache,
    )
    assert len(prefix_latent_cache) == 1
    assert model.calls == [
        (0, 0, 0),
        (1, 2, 1),
        (2, 1, 2),
        (3, 2, 2),
        (2, 1, 2),
        (3, 2, 2),
    ]
    assert torch.equal(scores["typed_return_pred"], cached_scores["typed_return_pred"])


def test_predict_oracle_group_matches_planner_latent_norm_clipping() -> None:
    graph = SimpleNamespace(
        num_nodes=3,
        node_names=["n0", "n1", "n2"],
        edge_src=torch.zeros(0, dtype=torch.long),
        edge_dst=torch.zeros(0, dtype=torch.long),
        gate_type_ids=torch.zeros(3, dtype=torch.long),
        fanin_lists=[[], [], []],
        fanout_lists=[[], [], []],
    )

    class DummyModel:
        def __init__(self):
            self.input_norms = []

        def online_encoder(self, x, edge_src, edge_dst, gate_type_ids):
            return x

        def predict_from_latent(
            self,
            z,
            node_id,
            action_type_id,
            relation,
            include_aux_heads=False,
            sequence_step=0,
        ):
            self.input_norms.append(float(z.norm(dim=1).max().item()))
            scalar = z.norm(dim=1).max()
            return {
                "z_pred": 10.0 * z,
                "q_pred": scalar,
                "reward_pred": scalar,
                "return_pred": scalar,
                "hard_reduction_pred": torch.stack([scalar, scalar, scalar]),
                "typed_marginal_pred": scalar,
                "typed_return_pred": scalar,
                "typed_sa_reduction_pred": torch.stack([scalar, scalar]),
            }

    group = [
        {
            "benchmark_id": "subckt_0003",
            "state_id": "prefix:2",
            "candidate_strategy": "onpolicy_counterfactual",
            "candidate_rank": "1",
            "node": "n2",
            "type": "observe",
            "oracle_delta_tc": "0.01",
            "prefix_sequence": '[["n0","control0"],["n1","control1"]]',
        }
    ]
    model = DummyModel()
    _predict_oracle_group_scores(
        model,
        {
            "relation_mode": "basic",
            "relation_depth": 2,
            "coverage_scale": 100.0,
            "oracle_latent_norm_clip_ratio": 1.0,
        },
        group,
        {"subckt_0003": graph},
        {"subckt_0003": torch.tensor([[3.0, 4.0]]).repeat(3, 1)},
        torch.device("cpu"),
    )

    assert model.input_norms == pytest.approx([5.0, 5.0, 5.0])


def test_oracle_sa_targets_are_normalized_by_prefix_fault_counts() -> None:
    targets, mask = _oracle_sa_reduction_targets(
        [
            {
                "oracle_hard_reduction_sa0": "5",
                "oracle_hard_reduction_sa1": "-2",
                "prefix_undetected_sa0_count": "20",
                "prefix_undetected_sa1_count": "10",
            },
            {},
        ],
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert targets[0].tolist() == pytest.approx([0.25, -0.2])
    assert mask.tolist() == [[True, True], [False, False]]


def test_oracle_sa_targets_mask_zero_prefix_polarity() -> None:
    targets, mask = _oracle_sa_reduction_targets(
        [
            {
                "oracle_hard_reduction_sa0": "-3",
                "oracle_hard_reduction_sa1": "2",
                "prefix_undetected_sa0_count": "0",
                "prefix_undetected_sa1_count": "10",
            }
        ],
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert targets[0].tolist() == pytest.approx([0.0, 0.2])
    assert mask.tolist() == [[False, True]]
