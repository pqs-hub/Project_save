import torch

from tpi_jepa.train import _oracle_pairwise_rank_loss


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
