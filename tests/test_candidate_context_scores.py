from tpi_jepa.plan import add_candidate_context_scores


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
