import torch

from tpi_jepa.train import (
    _oracle_action_family_loss,
    _oracle_action_type_loss,
    _oracle_conservative_loss,
    _oracle_context_loss,
    _oracle_topk_ndcg_loss,
)


def test_action_family_loss_pools_cp0_and_cp1() -> None:
    action_types = ["control0", "control1", "observe", "observe"]
    targets = torch.tensor([2.0, 0.0, 0.0, -1.0])
    cp0_good = torch.tensor([2.0, -2.0, 0.0, 0.0])
    cp1_good = torch.tensor([-2.0, 2.0, 0.0, 0.0])
    observe_bad = torch.tensor([-2.0, -2.0, 2.0, 2.0])
    config = {
        "oracle_action_family_aggregate_temperature": 0.1,
        "oracle_action_family_target_temperature": 0.1,
        "oracle_action_family_pred_temperature": 0.35,
    }

    cp0_loss, family_count = _oracle_action_family_loss(
        cp0_good, targets, action_types, config
    )
    cp1_loss, _ = _oracle_action_family_loss(cp1_good, targets, action_types, config)
    observe_loss, _ = _oracle_action_family_loss(observe_bad, targets, action_types, config)

    assert family_count == 2
    assert torch.allclose(cp0_loss, cp1_loss)
    assert cp0_loss.item() < observe_loss.item()


def test_action_type_loss_preserves_exact_target_ties() -> None:
    action_types = ["control0", "control0", "control1", "control1", "observe", "observe"]
    targets = torch.tensor([1.0, 0.0, 1.0, -1.0, 1.0, -2.0])
    neutral_preds = torch.zeros(6, requires_grad=True)
    config = {
        "oracle_action_type_aggregate_temperature": 0.1,
        "oracle_action_type_target_temperature": 0.025,
        "oracle_action_type_pred_temperature": 0.35,
    }

    loss, type_count = _oracle_action_type_loss(
        neutral_preds,
        targets,
        action_types,
        config,
    )

    assert type_count == 3
    assert torch.isclose(loss, torch.log(torch.tensor(3.0)), atol=1e-6)
    loss.backward()
    # All three families have the same best real target, so no arbitrary TSV
    # ordering is allowed to create a family-level gradient.
    family_gradients = [neutral_preds.grad[index : index + 2].sum() for index in (0, 2, 4)]
    assert torch.allclose(torch.stack(family_gradients), torch.zeros(3), atol=1e-6)


def test_action_type_loss_prefers_the_best_family() -> None:
    action_types = ["control0", "control0", "control1", "control1", "observe", "observe"]
    targets = torch.tensor([0.0, -1.0, 2.0, 0.0, 0.0, -2.0])
    good_preds = torch.tensor([0.0, 0.0, 2.0, 2.0, 0.0, 0.0])
    bad_preds = torch.tensor([0.0, 0.0, -2.0, -2.0, 0.0, 0.0])
    config = {
        "oracle_action_type_aggregate_temperature": 0.1,
        "oracle_action_type_target_temperature": 0.1,
        "oracle_action_type_pred_temperature": 0.35,
    }

    good_loss, _ = _oracle_action_type_loss(good_preds, targets, action_types, config)
    bad_loss, _ = _oracle_action_type_loss(bad_preds, targets, action_types, config)

    assert good_loss.item() < bad_loss.item()


def test_oracle_topk_ndcg_loss_prefers_correct_order() -> None:
    targets = torch.tensor([3.0, 2.0, 1.0, -1.0])
    good_preds = torch.tensor([3.0, 2.0, 0.5, -1.0])
    bad_preds = torch.tensor([-1.0, 0.5, 2.0, 3.0])
    config = {
        "oracle_ndcg_k": 3,
        "oracle_ndcg_target_temperature": 0.5,
        "oracle_ndcg_pred_temperature": 1.0,
    }

    good_loss = _oracle_topk_ndcg_loss(good_preds, targets, config)
    bad_loss = _oracle_topk_ndcg_loss(bad_preds, targets, config)

    assert good_loss.item() < bad_loss.item()


def test_oracle_conservative_loss_penalizes_high_non_top_scores() -> None:
    targets = torch.tensor([3.0, 2.0, 1.0, -1.0])
    safe_preds = torch.tensor([2.5, 2.0, 0.0, -0.5])
    overconfident_preds = torch.tensor([2.5, 2.0, 4.0, 3.5])
    config = {
        "oracle_conservative_positive_topk": 2,
        "oracle_conservative_hard_negative_topk": 2,
        "oracle_conservative_temperature": 1.0,
        "oracle_conservative_margin": 0.1,
        "oracle_conservative_normalize": True,
    }

    safe_loss = _oracle_conservative_loss(safe_preds, targets, config)
    overconfident_loss = _oracle_conservative_loss(overconfident_preds, targets, config)

    assert safe_loss.item() < overconfident_loss.item()


def test_oracle_context_loss_prefers_matching_relative_shape() -> None:
    targets = torch.tensor([3.0, 1.0, -1.0, -2.0])
    good_preds = torch.tensor([30.0, 10.0, -10.0, -20.0])
    bad_preds = torch.tensor([-20.0, -10.0, 10.0, 30.0])
    config = {
        "oracle_context_pred_temperature": 1.0,
        "oracle_context_target_temperature": 1.0,
        "oracle_context_top_weight": 0.5,
        "oracle_context_weight_temperature": 0.75,
    }

    good_loss = _oracle_context_loss(good_preds, targets, config)
    bad_loss = _oracle_context_loss(bad_preds, targets, config)

    assert good_loss.item() < bad_loss.item()
