import torch

from tpi_jepa.train import _oracle_conservative_loss, _oracle_context_loss, _oracle_topk_ndcg_loss


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
