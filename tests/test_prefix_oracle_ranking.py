from scripts.evaluate_prefix_oracle_ranking import same_type_pairwise_accuracy


def test_same_type_pairwise_ignores_easy_cross_type_ordering() -> None:
    predictions = [10.0, 9.0, 2.0, 1.0]
    targets = [1.0, 2.0, -1.0, -2.0]
    action_types = ["observe", "observe", "control0", "control0"]

    # OP is reversed while CP0 is correct; cross-type separation contributes
    # nothing, so the diagnostic exposes the 1/2 within-type accuracy.
    assert same_type_pairwise_accuracy(predictions, targets, action_types) == 0.5


def test_same_type_pairwise_returns_nan_without_comparable_pairs() -> None:
    result = same_type_pairwise_accuracy(
        [3.0, 2.0, 1.0],
        [1.0, 1.0, 0.0],
        ["observe", "observe", "control0"],
    )

    assert result != result
