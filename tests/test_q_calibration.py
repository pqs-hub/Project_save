import math

import numpy as np

from scripts.evaluate_q_calibration import (
    DEFAULT_CHECKPOINTS,
    DEFAULT_METHODS,
    add_context_stats,
    compute_scores,
    compute_spearman,
    evaluate_split,
    list_variants,
    promotion_rows,
    summary_lookup,
)


def test_monotonic_transform_preserves_ranking():
    q = np.array([1.0, 2.0, 3.0, 4.0])
    base_rank = np.argsort(q)

    transforms = ["raw", "group_center", "group_zscore", "global_zscore", "circuit_zscore", "platt"]

    for transform in transforms:
        q_t = compute_scores(q, transform=transform)
        assert np.all(np.argsort(q_t) == base_rank)


def test_no_promotion_when_thresholds_are_not_met():
    summary_rows = [
        {
            "split": "expanded",
            "checkpoint_name": "ckpt",
            "method": "raw",
            "mean_spearman": 0.49,
            "negative_top1_rate": 0.20,
            "mean_top1_real_delta_tc": 0.0,
            "mean_top1_regret": 0.02,
            "rank_changed_groups": 0,
        },
        {
            "split": "transfer",
            "checkpoint_name": "ckpt",
            "method": "raw",
            "mean_spearman": 0.19,
            "negative_top1_rate": 0.20,
            "mean_top1_real_delta_tc": 0.0,
            "mean_top1_regret": 0.02,
            "mean_sign_accuracy": 0.5,
            "rank_changed_groups": 0,
        },
    ]

    rows = promotion_rows(summary_rows, ["ckpt"], ["raw"])

    assert len(rows) == 1
    assert rows[0]["verdict"] == "REJECT"


def test_spearman_stability_with_ties():
    q = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([1, 1, 2, 2, 3], dtype=float)

    value = compute_spearman(q, y)

    assert math.isclose(value, 0.9486832980505138, rel_tol=0.0, abs_tol=1e-12)


def test_all_q_calibration_variants_present():
    variants = list_variants()

    assert len(variants) == 28
    assert len(variants) == len(DEFAULT_CHECKPOINTS) * len(DEFAULT_METHODS)
    assert ("Q_v0_rank1p0", "raw") in variants
    assert ("Q_v0_value1_rank1", "platt") in variants


def test_promotion_result_schema():
    summary_rows = [
        {
            "split": "expanded",
            "checkpoint_name": "ckpt",
            "method": "raw",
            "mean_spearman": 0.60,
            "negative_top1_rate": 0.10,
            "mean_top1_real_delta_tc": 0.1,
            "mean_top1_regret": 0.01,
            "rank_changed_groups": 0,
        },
        {
            "split": "transfer",
            "checkpoint_name": "ckpt",
            "method": "raw",
            "mean_spearman": 0.30,
            "negative_top1_rate": 0.10,
            "mean_top1_real_delta_tc": 0.1,
            "mean_top1_regret": 0.01,
            "mean_sign_accuracy": 0.8,
            "rank_changed_groups": 0,
        },
    ]

    row = promotion_rows(summary_rows, ["ckpt"], ["raw"])[0]
    required_keys = [
        "expanded_spearman",
        "transfer_spearman",
        "transfer_top1_regret",
        "expanded_negative_top1",
        "transfer_negative_top1",
        "verdict",
        "reasons",
    ]

    for key in required_keys:
        assert key in row


def test_scale_invariance_does_not_improve_regret():
    rows = [
        {
            "checkpoint_name": "ckpt",
            "benchmark_id": "b0",
            "state_id": "s0",
            "candidate_strategy": "fixed",
            "candidate_rank": "0",
            "node": "n0",
            "type": "control0",
            "action_key": "n0::control0",
            "oracle_delta_tc": "-0.20",
            "q_pred": "1.0",
        },
        {
            "checkpoint_name": "ckpt",
            "benchmark_id": "b0",
            "state_id": "s0",
            "candidate_strategy": "fixed",
            "candidate_rank": "1",
            "node": "n1",
            "type": "control1",
            "action_key": "n1::control1",
            "oracle_delta_tc": "0.10",
            "q_pred": "2.0",
        },
        {
            "checkpoint_name": "ckpt",
            "benchmark_id": "b0",
            "state_id": "s0",
            "candidate_strategy": "fixed",
            "candidate_rank": "2",
            "node": "n2",
            "type": "observe",
            "action_key": "n2::observe",
            "oracle_delta_tc": "0.40",
            "q_pred": "3.0",
        },
    ]
    stats = add_context_stats({"expanded": rows, "transfer": rows}, "q_pred")

    _, _, summary_rows = evaluate_split(
        split="transfer",
        rows=rows,
        checkpoints={"ckpt"},
        methods=["raw", "global_zscore", "platt"],
        score_field="q_pred",
        stats=stats,
    )
    raw = summary_lookup(summary_rows, "transfer", "ckpt", "raw")
    z = summary_lookup(summary_rows, "transfer", "ckpt", "global_zscore")
    platt = summary_lookup(summary_rows, "transfer", "ckpt", "platt")

    assert raw["rank_changed_groups"] == 0
    assert z["rank_changed_groups"] == 0
    assert platt["rank_changed_groups"] == 0
    assert math.isclose(raw["mean_top1_regret"], z["mean_top1_regret"], abs_tol=1e-12)
    assert math.isclose(raw["mean_top1_regret"], platt["mean_top1_regret"], abs_tol=1e-12)


def test_calibration_is_not_magic_for_monotonic_transforms():
    q = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([0.4, -0.1, 0.2, 0.0])
    raw = compute_spearman(compute_scores(q, "raw"), y)

    for transform in ["group_center", "group_zscore", "global_zscore", "circuit_zscore", "platt"]:
        calibrated = compute_spearman(compute_scores(q, transform), y)
        assert math.isclose(calibrated, raw, abs_tol=1e-12)
