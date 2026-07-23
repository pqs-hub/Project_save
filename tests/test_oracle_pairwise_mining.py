import torch
from torch import nn

import tpi_jepa.train as train_module
from tpi_jepa.train import (
    _oracle_pairwise_rank_loss,
    _oracle_same_type_rank_loss,
    _oracle_same_type_topk_ndcg_loss,
    train_oracle_ranking_step,
)


def test_hard_topk_pairwise_uses_model_topk_negatives():
    preds = torch.tensor([0.1, 3.0, 2.0, 0.0], requires_grad=True)
    targets = torch.tensor([1.0, -0.5, 0.4, 0.0])

    loss, pair_count = _oracle_pairwise_rank_loss(
        preds,
        targets,
        min_delta=0.1,
        temperature=1.0,
        mode="hard_topk",
        hard_negative_topk=2,
        positive_topk=1,
        max_pairs=0,
    )

    assert pair_count == 2
    assert loss.item() > 0.0
    loss.backward()
    assert preds.grad is not None


def test_all_pairwise_mode_keeps_existing_pair_count():
    preds = torch.tensor([0.0, 1.0, 2.0])
    targets = torch.tensor([0.0, 0.2, 0.5])

    _, pair_count = _oracle_pairwise_rank_loss(
        preds,
        targets,
        min_delta=0.1,
        temperature=1.0,
        mode="all",
    )

    assert pair_count == 3


def test_same_type_pairwise_excludes_cross_type_pairs():
    preds = torch.tensor([0.1, 0.4, 0.3, 0.2, 0.5], requires_grad=True)
    targets = torch.tensor([0.5, 0.1, 0.6, 0.2, 0.0])
    action_types = ["control0", "CP0", "control1", "observe", "OP"]

    loss, pair_count = _oracle_same_type_rank_loss(
        preds,
        targets,
        action_types,
        min_delta=0.1,
        temperature=1.0,
    )

    # One CP0 pair and one OP pair; the singleton CP1 has no pair.
    assert pair_count == 2
    assert loss.item() > 0.0
    loss.backward()
    assert preds.grad is not None
    assert preds.grad[2].item() == 0.0


def test_same_type_listwise_excludes_singletons_and_cross_type_comparisons():
    preds = torch.tensor([0.1, 0.4, 0.3, 0.2, 0.5], requires_grad=True)
    targets = torch.tensor([0.5, 0.1, 0.6, 0.2, 0.0])
    action_types = ["control0", "CP0", "control1", "observe", "OP"]

    loss, type_list_count = _oracle_same_type_topk_ndcg_loss(
        preds,
        targets,
        action_types,
        {
            "oracle_ndcg_k": 2,
            "oracle_ndcg_target_temperature": 0.5,
            "oracle_ndcg_pred_temperature": 1.0,
        },
    )

    assert type_list_count == 2
    assert loss.item() > 0.0
    loss.backward()
    assert preds.grad is not None
    assert preds.grad[2].item() == 0.0


def test_auxiliary_same_type_rank_updates_independent_score_head(monkeypatch):
    class DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.values = nn.Parameter(torch.tensor([-1.0, 0.0, 1.0]))

    model = DummyModel()

    def fake_scores(model, config, group, graph_cache, base_cache, device):
        return {
            "typed_marginal_pred": torch.zeros_like(model.values),
            "typed_return_pred": model.values,
        }

    monkeypatch.setattr(train_module, "_predict_oracle_group_scores", fake_scores)
    monkeypatch.setattr(train_module, "update_ema_if_encoder_trainable", lambda *args: None)
    group = [
        {"benchmark_id": "source", "node": f"n{index}", "type": "observe", "oracle_delta_tc": value}
        for index, value in enumerate([1.0, 0.5, 0.0])
    ]
    config = {
        "lambda_q_rank": 0.0,
        "lambda_same_type_rank": 0.0,
        "lambda_aux_rank": 0.0,
        "lambda_aux_same_type_rank": 1.0,
        "lambda_q_value": 0.0,
        "lambda_candidate": 0.0,
        "lambda_ndcg_rank": 0.0,
        "lambda_same_type_ndcg_rank": 0.0,
        "lambda_conservative_q": 0.0,
        "lambda_context_rank": 0.0,
        "lambda_oracle_sa_value": 0.0,
        "oracle_ranking_score_field": "typed_marginal_pred",
        "oracle_aux_ranking_score_field": "typed_return_pred",
        "oracle_batch_groups": 1,
        "oracle_group_sampling": "uniform",
        "oracle_warmup_epochs": 0,
        "oracle_ramp_epochs": 0,
        "oracle_pairwise_min_delta": 0.001,
        "oracle_pairwise_temperature": 1.0,
        "oracle_pairwise_mode": "all",
        "coverage_scale": 1.0,
        "ema_decay": 0.99,
    }
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.values.detach().clone()

    metrics = train_oracle_ranking_step(
        model,
        [group],
        optimizer,
        config,
        torch.device("cpu"),
        {},
        {},
        epoch=1,
    )

    assert metrics["oracle_pairs"] == 0.0
    assert metrics["oracle_aux_same_type_pairs"] == 3.0
    assert metrics["oracle_aux_same_type_rank_loss"] > 0.0
    assert not torch.equal(before, model.values.detach())
