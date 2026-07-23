import math

import pytest
import torch

from tpi_jepa.plan import (
    PLAN_FIELDNAMES,
    _adaptive_candidate_subset,
    _candidate_selection_key,
    _clip_latent_norms,
    add_candidate_context_scores,
)


def test_plan_fieldnames_cover_ensemble_typed_sa_statistics() -> None:
    assert "typed_sa_reduction_total_pred_mean" in PLAN_FIELDNAMES
    assert "typed_sa_reduction_total_pred_std" in PLAN_FIELDNAMES
    assert "typed_sa_reduction_total_pred_lcb" in PLAN_FIELDNAMES
    assert "candidate_prior_score" in PLAN_FIELDNAMES
    assert "candidate_prior_correction" in PLAN_FIELDNAMES


def test_candidate_context_score_uses_pool_support_heads() -> None:
    rows = [
        {
            "q_pred": 3.0,
            "reward_pred": 0.0,
            "return_pred": 0.0,
            "guarded_reward": 0.0,
            "hard_reduction_total_pred": 0.0,
            "derived_hard_reduction_hybrid_pred": 0.0,
            "hybrid_pred": 3.0,
            "bounded_residual_hybrid_pred": 3.0,
        },
        {
            "q_pred": 2.9,
            "reward_pred": 3.0,
            "return_pred": 3.0,
            "guarded_reward": 3.0,
            "hard_reduction_total_pred": 3.0,
            "derived_hard_reduction_hybrid_pred": 3.0,
            "hybrid_pred": 2.9,
            "bounded_residual_hybrid_pred": 2.9,
        },
        {
            "q_pred": 2.0,
            "reward_pred": 0.0,
            "return_pred": 0.0,
            "guarded_reward": 0.0,
            "hard_reduction_total_pred": 0.0,
            "derived_hard_reduction_hybrid_pred": 0.0,
            "hybrid_pred": 2.0,
            "bounded_residual_hybrid_pred": 2.0,
        },
    ]

    add_candidate_context_scores(rows)

    assert rows[0]["q_pred"] > rows[1]["q_pred"]
    assert rows[1]["q_pred_context"] > rows[0]["q_pred_context"]
    assert "reward_pred_context" in rows[1]


def test_typed_residual_is_bounded_and_alpha_zero_preserves_q_context(monkeypatch) -> None:
    rows = []
    for index in range(3):
        rows.append(
            {
                "type": "observe",
                "q_pred": float(index),
                "q_pred_lcb": float(index),
                "reward_pred": float(index),
                "return_pred": float(index),
                "typed_marginal_pred": 100.0 * index,
                "typed_return_pred": 200.0 * index,
                "typed_sa_reduction_total_pred": 300.0 * index,
                "guarded_reward": float(index),
                "hard_reduction_total_pred": float(index),
                "derived_hard_reduction_hybrid_pred": float(index),
                "hybrid_pred": float(index),
                "bounded_residual_hybrid_pred": float(index),
            }
        )

    monkeypatch.setenv("TPI_TYPED_RESIDUAL_ALPHA", "0")
    add_candidate_context_scores(rows)
    assert all(row["q_typed_residual_context"] == row["q_pred_context"] for row in rows)

    monkeypatch.setenv("TPI_TYPED_RESIDUAL_ALPHA", "0.2")
    add_candidate_context_scores(rows)
    assert all(abs(row["q_typed_residual_context"] - row["q_pred_context"]) <= 0.2 for row in rows)


def test_typed_residual_decay_reduces_ood_long_sequence_influence(monkeypatch) -> None:
    rows = []
    for index in range(3):
        rows.append(
            {
                "type": "observe",
                "q_pred": float(index),
                "q_pred_lcb": float(index),
                "reward_pred": float(index),
                "return_pred": float(index),
                "typed_marginal_pred": float(index),
                "typed_return_pred": float(index),
                "typed_sa_reduction_total_pred": float(index),
                "guarded_reward": float(index),
                "hard_reduction_total_pred": float(index),
                "derived_hard_reduction_hybrid_pred": float(index),
                "hybrid_pred": float(index),
                "bounded_residual_hybrid_pred": float(index),
            }
        )
    monkeypatch.setenv("TPI_TYPED_RESIDUAL_ALPHA", "0.2")
    monkeypatch.setenv("TPI_TYPED_RESIDUAL_DECAY_STEPS", "32")

    add_candidate_context_scores(rows, selected_count=64)

    expected = 0.2 * math.exp(-2.0)
    assert all(row["typed_residual_effective_alpha"] == pytest.approx(expected) for row in rows)


def test_typed_residual_decay_can_restart_at_residual_refresh(monkeypatch) -> None:
    rows = [_trust_row("observe", float(index), (float(index),) * 3) for index in range(3)]
    monkeypatch.setenv("TPI_TYPED_RESIDUAL_ALPHA", "0.2")
    monkeypatch.setenv("TPI_TYPED_RESIDUAL_DECAY_STEPS", "32")

    add_candidate_context_scores(rows, selected_count=208, residual_decay_start=192)

    expected = 0.2 * math.exp(-0.5)
    assert all(row["typed_residual_effective_alpha"] == pytest.approx(expected) for row in rows)


def _trust_row(action_type: str, q: float, typed: tuple[float, float, float]) -> dict:
    return {
        "node": f"{action_type}_{q}",
        "type": action_type,
        "q_pred": q,
        "q_pred_lcb": q,
        "reward_pred": q,
        "return_pred": q,
        "typed_marginal_pred": typed[0],
        "typed_return_pred": typed[1],
        "typed_sa_reduction_total_pred": typed[2],
        "guarded_reward": q,
        "hard_reduction_total_pred": q,
        "derived_hard_reduction_hybrid_pred": q,
        "hybrid_pred": q,
        "bounded_residual_hybrid_pred": q,
    }


def test_typed_trust_requires_unanimous_cp0_support(monkeypatch) -> None:
    rows = [
        _trust_row("observe", 3.0, (0.0, 0.0, 0.0)),
        _trust_row("control0", 2.95, (5.0, 5.0, -5.0)),
        _trust_row("control1", 0.0, (-5.0, -5.0, 5.0)),
    ]
    monkeypatch.setenv("TPI_TYPED_RESIDUAL_ALPHA", "1")
    monkeypatch.setenv("TPI_TYPED_TRUST_MIN_HEADS", "2")
    monkeypatch.setenv("TPI_TYPED_TRUST_CP0_MIN_HEADS", "3")

    add_candidate_context_scores(rows)

    assert rows[1]["typed_trust_support_count"] == 2
    assert rows[1]["typed_trust_eligible"] == 0
    assert rows[1]["q_typed_trust_context"] == rows[1]["q_pred_context"]
    assert rows[0]["q_typed_trust_context"] > rows[1]["q_typed_trust_context"]


def test_typed_trust_allows_supported_challenger(monkeypatch) -> None:
    rows = [
        _trust_row("observe", 3.0, (0.0, 0.0, 0.0)),
        _trust_row("control1", 2.95, (5.0, 5.0, 5.0)),
        _trust_row("control0", 0.0, (-5.0, -5.0, -5.0)),
    ]
    monkeypatch.setenv("TPI_TYPED_RESIDUAL_ALPHA", "1")
    monkeypatch.setenv("TPI_TYPED_TRUST_MIN_HEADS", "2")

    add_candidate_context_scores(rows)

    assert rows[1]["typed_trust_support_count"] == 3
    assert rows[1]["typed_trust_eligible"] == 1
    assert rows[1]["typed_trust_correction"] > 0.0
    assert rows[1]["q_typed_trust_context"] > rows[1]["q_pred_context"]


def test_typed_trust_alpha_zero_exactly_preserves_base_score(monkeypatch) -> None:
    rows = [
        _trust_row("observe", 3.0, (0.0, 0.0, 0.0)),
        _trust_row("control1", 2.0, (5.0, 5.0, 5.0)),
        _trust_row("control0", 0.0, (-5.0, -5.0, -5.0)),
    ]
    monkeypatch.setenv("TPI_TYPED_RESIDUAL_ALPHA", "0")

    add_candidate_context_scores(rows)

    assert all(row["q_typed_trust_context"] == row["q_pred_context"] for row in rows)


def test_typed_trust_rejects_invalid_head_count(monkeypatch) -> None:
    monkeypatch.setenv("TPI_TYPED_TRUST_MIN_HEADS", "4")
    with pytest.raises(ValueError, match="between 0 and 3"):
        add_candidate_context_scores([_trust_row("observe", 0.0, (0.0, 0.0, 0.0))])


def test_typed_reliable_gate_ignores_unreliable_sa_veto(monkeypatch) -> None:
    rows = [
        _trust_row("observe", 3.0, (0.0, 0.0, 5.0)),
        _trust_row("control1", 2.95, (5.0, 5.0, -5.0)),
        _trust_row("control0", 0.0, (-5.0, -5.0, 5.0)),
    ]
    monkeypatch.setenv("TPI_TYPED_RESIDUAL_ALPHA", "1")
    monkeypatch.setenv("TPI_TYPED_RELIABLE_MIN_HEADS", "2")

    add_candidate_context_scores(rows)

    assert rows[1]["typed_reliable_support_count"] == 2
    assert rows[1]["typed_reliable_eligible"] == 1
    assert rows[1]["typed_reliable_applied_correction"] > 0.0
    assert rows[1]["q_typed_reliable_context"] > rows[1]["q_pred_context"]


def test_typed_reliable_gate_keeps_strict_cp0_support(monkeypatch) -> None:
    rows = [
        _trust_row("observe", 3.0, (0.0, 0.0, 0.0)),
        _trust_row("control0", 2.95, (5.0, -5.0, 5.0)),
        _trust_row("control1", 0.0, (-5.0, 5.0, -5.0)),
    ]
    monkeypatch.setenv("TPI_TYPED_RESIDUAL_ALPHA", "1")
    monkeypatch.setenv("TPI_TYPED_RELIABLE_MIN_HEADS", "1")
    monkeypatch.setenv("TPI_TYPED_RELIABLE_CP0_MIN_HEADS", "2")

    add_candidate_context_scores(rows)

    assert rows[1]["typed_reliable_support_count"] == 1
    assert rows[1]["typed_reliable_eligible"] == 0
    assert rows[1]["q_typed_reliable_context"] == rows[1]["q_pred_context"]


def test_typed_reliable_alpha_zero_and_validation(monkeypatch) -> None:
    rows = [
        _trust_row("observe", 3.0, (0.0, 0.0, 0.0)),
        _trust_row("control1", 2.0, (5.0, 5.0, -5.0)),
        _trust_row("control0", 0.0, (-5.0, -5.0, 5.0)),
    ]
    monkeypatch.setenv("TPI_TYPED_RESIDUAL_ALPHA", "0")
    add_candidate_context_scores(rows)
    assert all(row["q_typed_reliable_context"] == row["q_pred_context"] for row in rows)

    monkeypatch.setenv("TPI_TYPED_RELIABLE_MARGINAL_WEIGHT", "1.1")
    with pytest.raises(ValueError, match="between 0 and 1"):
        add_candidate_context_scores(rows)


def test_candidate_prior_is_a_bounded_additive_residual(monkeypatch) -> None:
    base_rows = [
        {**_trust_row("observe", float(index), (float(index),) * 3), "candidate_prior_score": score}
        for index, score in enumerate((-1.0, 0.0, 1.0))
    ]
    monkeypatch.setenv("TPI_CANDIDATE_PRIOR_ALPHA", "0")
    add_candidate_context_scores(base_rows)
    base_scores = [row["q_typed_reliable_context"] for row in base_rows]

    rows = [
        {**_trust_row("observe", float(index), (float(index),) * 3), "candidate_prior_score": score}
        for index, score in enumerate((-1.0, 0.0, 1.0))
    ]
    monkeypatch.setenv("TPI_CANDIDATE_PRIOR_ALPHA", "0.2")
    add_candidate_context_scores(rows)

    assert rows[0]["q_typed_reliable_context"] < base_scores[0]
    assert rows[1]["q_typed_reliable_context"] == pytest.approx(base_scores[1])
    assert rows[2]["q_typed_reliable_context"] > base_scores[2]
    assert all(abs(row["candidate_prior_correction"]) <= 6.0 for row in rows)


def test_candidate_prior_rejects_negative_alpha(monkeypatch) -> None:
    monkeypatch.setenv("TPI_CANDIDATE_PRIOR_ALPHA", "-0.1")
    with pytest.raises(ValueError, match="must be non-negative"):
        add_candidate_context_scores([_trust_row("observe", 0.0, (0.0, 0.0, 0.0))])


def test_context_tie_break_uses_raw_head() -> None:
    rows = [
        {"score_adjusted": 8.1, "q_pred": 0.2},
        {"score_adjusted": 8.1, "q_pred": 0.7},
    ]
    best = max(rows, key=lambda row: _candidate_selection_key(row, "q_pred_context"))
    assert best is rows[1]


def test_quantized_context_tie_break_uses_stable_candidate_id(monkeypatch) -> None:
    monkeypatch.setenv("TPI_SCORE_QUANTIZATION", "0.001")
    rows = [
        {"node": "N10", "type": "observe", "score_adjusted": 8.10001, "q_pred": 0.70001},
        {"node": "N20", "type": "control1", "score_adjusted": 8.10002, "q_pred": 0.70002},
    ]

    best = max(rows, key=lambda row: _candidate_selection_key(row, "q_pred_context"))

    assert best is rows[1]


def test_negative_score_quantization_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("TPI_SCORE_QUANTIZATION", "-0.1")
    with pytest.raises(ValueError, match="must be non-negative"):
        _candidate_selection_key({"score_adjusted": 1.0}, "q_pred_context")


def test_adaptive_candidates_expand_only_for_ambiguous_prefix(monkeypatch) -> None:
    monkeypatch.setenv("TPI_ADAPTIVE_BASE_CANDIDATES", "2")
    monkeypatch.setenv("TPI_ADAPTIVE_EXPANSION_MARGIN", "0.1")
    rows = [
        {"node": "N1", "score_adjusted": 1.0},
        {"node": "N2", "score_adjusted": 0.95},
        {"node": "N3", "score_adjusted": 2.0},
    ]

    selected, gap, expanded = _adaptive_candidate_subset(rows, "q_pred")

    assert selected is rows
    assert gap == pytest.approx(0.05)
    assert expanded is True


def test_adaptive_candidates_keep_confident_prefix(monkeypatch) -> None:
    monkeypatch.setenv("TPI_ADAPTIVE_BASE_CANDIDATES", "2")
    monkeypatch.setenv("TPI_ADAPTIVE_EXPANSION_MARGIN", "0.1")
    rows = [
        {"node": "N1", "score_adjusted": 1.0},
        {"node": "N2", "score_adjusted": 0.5},
        {"node": "N3", "score_adjusted": 2.0},
    ]

    selected, gap, expanded = _adaptive_candidate_subset(rows, "q_pred")

    assert selected == rows[:2]
    assert gap == pytest.approx(0.5)
    assert expanded is False


def test_adaptive_relative_range_is_scale_invariant(monkeypatch) -> None:
    monkeypatch.setenv("TPI_ADAPTIVE_BASE_CANDIDATES", "4")
    monkeypatch.setenv("TPI_ADAPTIVE_EXPANSION_MARGIN", "0.06")
    monkeypatch.setenv("TPI_ADAPTIVE_MARGIN_MODE", "relative_range")
    rows = [
        {"node": "N1", "score_adjusted": 10.0},
        {"node": "N2", "score_adjusted": 9.5},
        {"node": "N3", "score_adjusted": 5.0},
        {"node": "N4", "score_adjusted": 0.0},
        {"node": "N5", "score_adjusted": 20.0},
    ]

    selected, gap, expanded = _adaptive_candidate_subset(rows, "q_pred")
    scaled = [{**row, "score_adjusted": 7.0 * row["score_adjusted"]} for row in rows]
    scaled_selected, scaled_gap, scaled_expanded = _adaptive_candidate_subset(scaled, "q_pred")

    assert gap == pytest.approx(0.05)
    assert scaled_gap == pytest.approx(gap)
    assert expanded is True
    assert scaled_expanded is True
    assert selected is rows
    assert scaled_selected is scaled


def test_invalid_adaptive_margin_mode_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("TPI_ADAPTIVE_BASE_CANDIDATES", "2")
    monkeypatch.setenv("TPI_ADAPTIVE_MARGIN_MODE", "circuit_specific")

    with pytest.raises(ValueError, match="TPI_ADAPTIVE_MARGIN_MODE"):
        _adaptive_candidate_subset(
            [{"score_adjusted": 1.0}, {"score_adjusted": 0.0}, {"score_adjusted": 2.0}],
            "q_pred",
        )


def test_type_context_reduces_fixed_action_type_offset() -> None:
    def row(action_type: str, q: float, reward: float, ret: float, hard: float) -> dict:
        return {
            "type": action_type,
            "q_pred": q,
            "reward_pred": reward,
            "return_pred": ret,
            "guarded_reward": min(reward, ret),
            "hard_reduction_total_pred": hard,
            "derived_hard_reduction_hybrid_pred": hard,
            "hybrid_pred": q + reward + ret + hard,
            "bounded_residual_hybrid_pred": reward + hard,
        }

    rows = [
        row("control1", 100.0, 100.0, 100.0, 100.0),
        row("control1", 99.0, 99.0, 99.0, 99.0),
        row("observe", 4.0, 4.0, 4.0, 4.0),
        row("observe", 1.0, 1.0, 1.0, 1.0),
    ]

    add_candidate_context_scores(rows)

    raw_gap = rows[1]["q_pred_context"] - rows[2]["q_pred_context"]
    calibrated_gap = rows[1]["q_pred_type_context"] - rows[2]["q_pred_type_context"]
    assert calibrated_gap < raw_gap
    assert all("consensus_pred_type_context" in row for row in rows)


def test_consensus_penalizes_single_head_spike() -> None:
    rows = [
        {
            "type": "observe",
            "q_pred": 10.0,
            "reward_pred": 0.0,
            "return_pred": 0.0,
            "guarded_reward": 0.0,
            "hard_reduction_total_pred": 0.0,
            "derived_hard_reduction_hybrid_pred": 0.0,
            "hybrid_pred": 10.0,
            "bounded_residual_hybrid_pred": 10.0,
        },
        {
            "type": "observe",
            "q_pred": 8.0,
            "reward_pred": 8.0,
            "return_pred": 8.0,
            "guarded_reward": 8.0,
            "hard_reduction_total_pred": 8.0,
            "derived_hard_reduction_hybrid_pred": 8.0,
            "hybrid_pred": 8.0,
            "bounded_residual_hybrid_pred": 8.0,
        },
        {
            "type": "observe",
            "q_pred": 0.0,
            "reward_pred": 0.0,
            "return_pred": 0.0,
            "guarded_reward": 0.0,
            "hard_reduction_total_pred": 0.0,
            "derived_hard_reduction_hybrid_pred": 0.0,
            "hybrid_pred": 0.0,
            "bounded_residual_hybrid_pred": 0.0,
        },
    ]

    add_candidate_context_scores(rows)

    assert rows[1]["consensus_pred_context"] > rows[0]["consensus_pred_context"]


def test_latent_norm_clip_preserves_small_vectors_and_bounds_large_ones() -> None:
    z = torch.tensor([[3.0, 4.0], [30.0, 40.0]])

    clipped = _clip_latent_norms(z, 10.0)

    assert torch.equal(clipped[0], z[0])
    assert torch.isclose(clipped[1].norm(), torch.tensor(10.0))
    assert torch.equal(_clip_latent_norms(z, None), z)


def test_lcb_context_blends_uncertainty_with_pool_support() -> None:
    rows = [
        {
            "type": "observe",
            "q_pred": 4.0,
            "q_pred_lcb": -2.0,
            "reward_pred": 0.0,
            "return_pred": 0.0,
            "guarded_reward": 0.0,
            "hard_reduction_total_pred": 0.0,
            "derived_hard_reduction_hybrid_pred": 0.0,
            "hybrid_pred": 4.0,
            "bounded_residual_hybrid_pred": 4.0,
        },
        {
            "type": "observe",
            "q_pred": 3.5,
            "q_pred_lcb": 3.0,
            "reward_pred": 3.0,
            "return_pred": 3.0,
            "guarded_reward": 3.0,
            "hard_reduction_total_pred": 3.0,
            "derived_hard_reduction_hybrid_pred": 3.0,
            "hybrid_pred": 3.5,
            "bounded_residual_hybrid_pred": 3.5,
        },
        {
            "type": "observe",
            "q_pred": 0.0,
            "q_pred_lcb": 0.0,
            "reward_pred": 0.0,
            "return_pred": 0.0,
            "guarded_reward": 0.0,
            "hard_reduction_total_pred": 0.0,
            "derived_hard_reduction_hybrid_pred": 0.0,
            "hybrid_pred": 0.0,
            "bounded_residual_hybrid_pred": 0.0,
        },
    ]

    add_candidate_context_scores(rows)

    assert rows[1]["q_pred_lcb_context"] > rows[0]["q_pred_lcb_context"]
    assert rows[1]["q_pred_context_lcb"] > rows[0]["q_pred_context_lcb"]


def test_q_context_support_strength_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {
            "type": "observe",
            "q_pred": 3.0,
            "reward_pred": 0.0,
            "return_pred": 0.0,
            "guarded_reward": 0.0,
            "hard_reduction_total_pred": 0.0,
            "derived_hard_reduction_hybrid_pred": 0.0,
            "hybrid_pred": 3.0,
            "bounded_residual_hybrid_pred": 3.0,
        },
        {
            "type": "observe",
            "q_pred": 2.9,
            "reward_pred": 3.0,
            "return_pred": 3.0,
            "guarded_reward": 3.0,
            "hard_reduction_total_pred": 3.0,
            "derived_hard_reduction_hybrid_pred": 3.0,
            "hybrid_pred": 2.9,
            "bounded_residual_hybrid_pred": 2.9,
        },
        {
            "type": "observe",
            "q_pred": 0.0,
            "reward_pred": 0.0,
            "return_pred": 0.0,
            "guarded_reward": 0.0,
            "hard_reduction_total_pred": 0.0,
            "derived_hard_reduction_hybrid_pred": 0.0,
            "hybrid_pred": 0.0,
            "bounded_residual_hybrid_pred": 0.0,
        },
    ]

    monkeypatch.setenv("TPI_Q_CONTEXT_SUPPORT_ALPHA", "0")
    monkeypatch.setenv("TPI_Q_CONTEXT_DISAGREEMENT_BETA", "0")
    add_candidate_context_scores(rows)

    assert rows[0]["q_pred_context"] > rows[1]["q_pred_context"]


def test_q_context_rejects_negative_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TPI_Q_CONTEXT_SUPPORT_ALPHA", "-0.1")
    with pytest.raises(ValueError, match="must be non-negative"):
        add_candidate_context_scores([{"q_pred": 0.0}])
