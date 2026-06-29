"""Train the minimal TPI-JEPA world model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from .dataset import RolloutSample, TPIDataset, TPIRolloutDataset, TransitionSample, split_by_benchmark
from .features import SCOAP_END, SCOAP_START
from .labels import load_labels
from .model import TPIWorldModel, update_ema
from .protocol import excluded_benchmarks_from_config, filter_rows_by_excluded_benchmarks


REGION_START = SCOAP_END


def load_config(path: str | Path) -> dict:
    """Load a JSON training config."""

    with Path(path).open() as f:
        return json.load(f)


def _device_from_config(config: dict) -> torch.device:
    """Use configured device, falling back to CPU if CUDA is unavailable."""

    requested = str(config.get("device", "cpu"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[train] warning: CUDA requested but unavailable; using CPU")
        requested = "cpu"
    return torch.device(requested)


def _pattern_target_valid(rows) -> bool:
    """Return true only when pattern targets exist and have nonzero variation."""

    values = [row.delta_pattern for row in rows if row.delta_pattern is not None]
    return bool(values) and len(set(values)) > 1


def _sample_to_device(sample: TransitionSample, device: torch.device) -> dict:
    """Move tensor fields from one sample to the training device."""

    return {
        "graph": sample.graph,
        "x_pre": sample.x_pre.to(device),
        "x_post": sample.x_post.to(device),
        "action_node_id": sample.action_node_id,
        "action_type_id": sample.action_type_id,
        "relation_features": sample.relation_features.to(device),
        "delta_fault_coverage": sample.delta_fault_coverage.to(device),
        "delta_pattern": sample.delta_pattern.to(device),
        "has_pattern_target": sample.has_pattern_target.to(device),
        "hard_targets_post": sample.hard_targets_post.to(device),
        "hard_count_post": sample.hard_count_post.to(device),
        "hard_reduction_target": sample.hard_reduction_target.to(device),
        "has_hard_targets": sample.has_hard_targets.to(device),
    }


def _rollout_sample_to_device(sample: RolloutSample, device: torch.device) -> dict:
    """Move tensor fields from one rollout sample to the training device."""

    return {
        "graph": sample.graph,
        "x_start": sample.x_start.to(device),
        "x_targets": [x.to(device) for x in sample.x_targets],
        "action_node_ids": sample.action_node_ids,
        "action_type_ids": sample.action_type_ids,
        "relation_features": [rel.to(device) for rel in sample.relation_features],
        "delta_fault_coverages": [value.to(device) for value in sample.delta_fault_coverages],
        "delta_patterns": [value.to(device) for value in sample.delta_patterns],
        "has_pattern_targets": [value.to(device) for value in sample.has_pattern_targets],
        "hard_targets_post": [value.to(device) for value in sample.hard_targets_post],
        "hard_count_post": [value.to(device) for value in sample.hard_count_post],
        "hard_reduction_targets": [value.to(device) for value in sample.hard_reduction_targets],
        "has_hard_targets": [value.to(device) for value in sample.has_hard_targets],
    }


def discounted_return_targets(
    deltas: list[torch.Tensor],
    steps: int | None = None,
    gamma: float = 1.0,
    scale: float = 1.0,
) -> list[torch.Tensor]:
    """Return per-step discounted cumulative future delta targets."""

    limit = len(deltas) if steps is None else min(int(steps), len(deltas))
    if limit <= 0:
        return []
    returns: list[torch.Tensor] = []
    running = torch.zeros((), dtype=deltas[0].dtype, device=deltas[0].device)
    for value in reversed(deltas[:limit]):
        running = value * float(scale) + float(gamma) * running
        returns.append(running)
    returns.reverse()
    return returns


def _hard_pos_weight(hard_targets: torch.Tensor, max_weight: float = 20.0) -> torch.Tensor:
    """Build a stable positive-class weight for sparse hard-fault labels."""

    pos = hard_targets.sum(dim=0)
    neg = hard_targets.shape[0] - pos
    return (neg / pos.clamp_min(1.0)).clamp(max=max_weight)


def _hard_loss_elements(hard_logits: torch.Tensor, hard_targets: torch.Tensor, config: dict) -> torch.Tensor:
    """Compute unreduced hard-label loss elements for BCE, focal, or ASL."""

    pos_weight = _hard_pos_weight(hard_targets, float(config.get("hard_pos_weight_max", 20.0)))
    loss_type = str(config.get("hard_loss", "bce")).lower()
    if loss_type == "focal":
        bce = F.binary_cross_entropy_with_logits(
            hard_logits,
            hard_targets,
            pos_weight=pos_weight,
            reduction="none",
        )
        prob = hard_logits.sigmoid()
        p_t = prob * hard_targets + (1.0 - prob) * (1.0 - hard_targets)
        gamma = float(config.get("hard_focal_gamma", 2.0))
        element_loss = bce * (1.0 - p_t).clamp_min(0.0).pow(gamma)
        alpha = float(config.get("hard_focal_alpha", -1.0))
        if alpha >= 0.0:
            alpha_t = alpha * hard_targets + (1.0 - alpha) * (1.0 - hard_targets)
            element_loss = alpha_t * element_loss
        return element_loss
    if loss_type == "asl":
        prob = hard_logits.sigmoid()
        clip = float(config.get("hard_asl_clip", 0.05))
        gamma_pos = float(config.get("hard_asl_gamma_pos", 0.0))
        gamma_neg = float(config.get("hard_asl_gamma_neg", 4.0))
        eps = 1e-8
        neg_prob = (prob - clip).clamp_min(0.0) if clip > 0.0 else prob
        pos_loss = -torch.log(prob.clamp_min(eps)) * pos_weight.view(1, -1)
        neg_loss = -torch.log((1.0 - neg_prob).clamp_min(eps))
        if gamma_pos > 0.0:
            pos_loss = pos_loss * (1.0 - prob).clamp_min(0.0).pow(gamma_pos)
        if gamma_neg > 0.0:
            neg_loss = neg_loss * neg_prob.clamp_min(0.0).pow(gamma_neg)
        return hard_targets * pos_loss + (1.0 - hard_targets) * neg_loss
    return F.binary_cross_entropy_with_logits(
        hard_logits,
        hard_targets,
        pos_weight=pos_weight,
        reduction="none",
    )


def _reduce_hard_loss(element_loss: torch.Tensor, hard_targets: torch.Tensor, config: dict) -> torch.Tensor:
    """Reduce hard-label loss with random or score-based hard negative mining."""

    negative_ratio = int(config.get("hard_negative_sample_ratio", 0) or 0)
    if negative_ratio <= 0:
        return element_loss.mean()

    mining = str(config.get("hard_negative_mining", "random")).lower()
    pos_mask = hard_targets > 0.5
    neg_mask = ~pos_mask
    pos_count = int(pos_mask.sum().item())
    if pos_count <= 0:
        return element_loss.mean()
    neg_idx = torch.nonzero(neg_mask.flatten(), as_tuple=False).flatten()
    keep_neg = min(int(negative_ratio * pos_count), int(neg_idx.numel()))
    if keep_neg <= 0:
        return element_loss[pos_mask].mean()
    if mining in {"topk", "hard", "hard_topk"}:
        neg_loss = element_loss.flatten()[neg_idx]
        top_local = torch.topk(neg_loss, k=keep_neg).indices
        sampled_flat = neg_idx[top_local]
    elif mining == "mixed":
        hard_keep = max(1, keep_neg // 2)
        random_keep = keep_neg - hard_keep
        neg_loss = element_loss.flatten()[neg_idx]
        top_local = torch.topk(neg_loss, k=hard_keep).indices
        sampled = [neg_idx[top_local]]
        if random_keep > 0:
            remaining_mask = torch.ones(neg_idx.numel(), dtype=torch.bool, device=neg_idx.device)
            remaining_mask[top_local] = False
            remaining = neg_idx[remaining_mask]
            if remaining.numel() > 0:
                perm = torch.randperm(remaining.numel(), device=remaining.device)[: min(random_keep, remaining.numel())]
                sampled.append(remaining[perm])
        sampled_flat = torch.cat(sampled)
    else:
        perm = torch.randperm(neg_idx.numel(), device=neg_idx.device)[:keep_neg]
        sampled_flat = neg_idx[perm]
    sample_mask = pos_mask.flatten()
    sample_mask = sample_mask.clone()
    sample_mask[sampled_flat] = True
    return element_loss.flatten()[sample_mask].mean()


def _hard_bce_loss(hard_logits: torch.Tensor, hard_targets: torch.Tensor, config: dict) -> torch.Tensor:
    """Compute the configured hard-fault classification loss."""

    return _reduce_hard_loss(_hard_loss_elements(hard_logits, hard_targets, config), hard_targets, config)


def _hard_rank_loss(
    hard_logits: torch.Tensor,
    hard_targets: torch.Tensor,
    hard_count_target: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    """Pairwise ranking loss that pushes hard/high-count nodes above easy nodes."""

    margin = float(config.get("hard_rank_margin", 0.2))
    max_pairs = int(config.get("hard_rank_pairs", 256) or 256)
    scores = hard_logits.sigmoid().max(dim=1).values
    target_score = torch.maximum(hard_targets.max(dim=1).values, hard_count_target)
    pos_idx = torch.nonzero(target_score > 0.0, as_tuple=False).flatten()
    neg_idx = torch.nonzero(target_score <= 0.0, as_tuple=False).flatten()
    losses = []
    if pos_idx.numel() > 0 and neg_idx.numel() > 0:
        pos_scores = scores[pos_idx]
        neg_scores = scores[neg_idx]
        pos_keep = min(pos_scores.numel(), max(1, int(max_pairs**0.5)))
        neg_keep = min(neg_scores.numel(), max(1, max_pairs // pos_keep))
        pos_sel = torch.topk(-pos_scores, k=pos_keep).indices
        neg_sel = torch.topk(neg_scores, k=neg_keep).indices
        pair_margin = margin - (pos_scores[pos_sel].view(-1, 1) - neg_scores[neg_sel].view(1, -1))
        losses.append(pair_margin.clamp_min(0.0).mean())

    high_idx = torch.nonzero(hard_count_target > 0.0, as_tuple=False).flatten()
    if high_idx.numel() >= 2:
        count_values = hard_count_target[high_idx]
        order = torch.argsort(count_values)
        low_idx = high_idx[order[: max(1, min(max_pairs // 2, high_idx.numel() // 2))]]
        high_sel = high_idx[order[-low_idx.numel() :]]
        delta = (hard_count_target[high_sel].view(-1, 1) - hard_count_target[low_idx].view(1, -1)).clamp_min(0.0)
        valid = delta > 1e-6
        if bool(valid.any().item()):
            pair_margin = margin * delta - (scores[high_sel].view(-1, 1) - scores[low_idx].view(1, -1))
            losses.append(pair_margin[valid].clamp_min(0.0).mean())

    if not losses:
        return torch.zeros((), dtype=hard_logits.dtype, device=hard_logits.device)
    return torch.stack(losses).mean()


def _hard_brier_loss(hard_logits: torch.Tensor, hard_targets: torch.Tensor) -> torch.Tensor:
    """Probability calibration loss for hard-fault logits."""

    return (hard_logits.sigmoid() - hard_targets).pow(2).mean()


def _hard_soft_f1_loss(hard_logits: torch.Tensor, hard_targets: torch.Tensor, config: dict) -> torch.Tensor:
    """Differentiable macro soft-F1 loss over SA0/SA1 hard labels."""

    eps = float(config.get("hard_soft_f1_eps", 1e-6))
    prob = hard_logits.sigmoid()
    soft_tp = (prob * hard_targets).sum(dim=0)
    soft_fp = (prob * (1.0 - hard_targets)).sum(dim=0)
    soft_fn = ((1.0 - prob) * hard_targets).sum(dim=0)
    soft_f1 = (2.0 * soft_tp) / (2.0 * soft_tp + soft_fp + soft_fn + eps)
    return 1.0 - soft_f1.mean()


def _hard_reduction_mask(batch: dict) -> bool:
    """Return true when graph-level hard reduction targets are valid."""

    return bool(batch["has_hard_targets"].item())


def compute_loss(batch: dict, model: TPIWorldModel, config: dict, pattern_enabled: bool) -> tuple[torch.Tensor, dict]:
    """Compute world-model and hard-fault-aware losses for one sample."""

    out = model(
        batch["graph"],
        batch["x_pre"],
        batch["x_post"],
        batch["action_node_id"],
        batch["action_type_id"],
        batch["relation_features"],
    )
    jepa_loss = F.smooth_l1_loss(out["z_pred"], out["z_t1"])
    scoap_target = batch["x_post"][:, SCOAP_START:SCOAP_END]
    scoap_loss = F.smooth_l1_loss(out["scoap_pred"], scoap_target)
    delta_scoap_target = batch["x_post"][:, SCOAP_START:SCOAP_END] - batch["x_pre"][:, SCOAP_START:SCOAP_END]
    delta_scoap_loss = F.smooth_l1_loss(out["delta_scoap_pred"], delta_scoap_target)
    hard_weight = _hard_action_weight(batch["x_pre"], batch["action_node_id"], config)
    hard_targets = batch["hard_targets_post"]
    hard_bce = _hard_bce_loss(out["hard_logits"], hard_targets, config)
    hard_rank = _hard_rank_loss(out["hard_logits"], hard_targets, batch["hard_count_post"], config)
    hard_brier = _hard_brier_loss(out["hard_logits"], hard_targets)
    hard_soft_f1 = _hard_soft_f1_loss(out["hard_logits"], hard_targets, config)
    hard_count_loss = F.smooth_l1_loss(out["hard_count_pred"], batch["hard_count_post"])
    hard_reduction_loss = (
        F.smooth_l1_loss(out["hard_reduction_pred"], batch["hard_reduction_target"])
        if _hard_reduction_mask(batch)
        else torch.zeros((), dtype=out["z_pred"].dtype, device=out["z_pred"].device)
    )
    coverage_scale = float(config.get("coverage_scale", 100.0))
    reward_target = coverage_scale * batch["delta_fault_coverage"]
    reward_loss = F.smooth_l1_loss(out["reward_pred"], reward_target)

    use_pattern = bool(pattern_enabled and float(config.get("lambda_pattern", 0.0)) > 0.0 and batch["has_pattern_target"].item())
    if use_pattern:
        pattern_loss = F.smooth_l1_loss(out["pattern_pred"], batch["delta_pattern"])
    else:
        pattern_loss = torch.zeros((), dtype=reward_loss.dtype, device=reward_loss.device)
    return_scale = float(config.get("return_scale", coverage_scale))
    return_loss = F.smooth_l1_loss(out["return_pred"], return_scale * batch["delta_fault_coverage"])
    weighted_reward_loss = hard_weight * reward_loss
    weighted_return_loss = hard_weight * return_loss
    lambda_jepa = float(config["lambda_jepa"])

    total = (
        lambda_jepa * jepa_loss
        + float(config["lambda_scoap"]) * scoap_loss
        + float(config.get("lambda_delta_scoap", 0.0)) * delta_scoap_loss
        + float(config.get("lambda_hard", 0.0)) * hard_bce
        + float(config.get("lambda_hard_rank", 0.0)) * hard_rank
        + float(config.get("lambda_hard_brier", 0.0)) * hard_brier
        + float(config.get("lambda_hard_soft_f1", 0.0)) * hard_soft_f1
        + float(config.get("lambda_hard_count", 0.0)) * hard_count_loss
        + float(config.get("lambda_hard_reduction", 0.0)) * hard_reduction_loss
        + float(config["lambda_fc"]) * weighted_reward_loss
        + float(config["lambda_pattern"]) * pattern_loss
        + float(config.get("lambda_return", 0.0)) * weighted_return_loss
    )
    metrics = {
        "loss": float(total.detach().cpu().item()),
        "jepa_loss": float(jepa_loss.detach().cpu().item()),
        "scoap_loss": float(scoap_loss.detach().cpu().item()),
        "delta_scoap_loss": float(delta_scoap_loss.detach().cpu().item()),
        "hard_bce_loss": float(hard_bce.detach().cpu().item()),
        "hard_rank_loss": float(hard_rank.detach().cpu().item()),
        "hard_brier_loss": float(hard_brier.detach().cpu().item()),
        "hard_soft_f1_loss": float(hard_soft_f1.detach().cpu().item()),
        "hard_count_loss": float(hard_count_loss.detach().cpu().item()),
        "hard_reduction_loss": float(hard_reduction_loss.detach().cpu().item()),
        "fc_loss": float(reward_loss.detach().cpu().item()),
        "reward_loss": float(reward_loss.detach().cpu().item()),
        "pattern_loss": float(pattern_loss.detach().cpu().item()),
        "return_loss": float(return_loss.detach().cpu().item()),
        "hard_weight": float(hard_weight.detach().cpu().item()),
    }
    return total, metrics


def _hard_action_weight(x: torch.Tensor, action_node_id: int, config: dict) -> torch.Tensor:
    """Optionally emphasize utility-head learning on hard-to-test action regions."""

    strength = float(config.get("hard_sample_weight", 0.0))
    if strength <= 0.0:
        return torch.ones((), dtype=x.dtype, device=x.device)
    if x.shape[1] > REGION_START + 3:
        hard_score = x[action_node_id, REGION_START + 3]
    else:
        scoap = x[action_node_id, SCOAP_START:SCOAP_END]
        hard_score = (torch.maximum(scoap[0], scoap[1]) + scoap[2]) * 0.5
        hard_score = hard_score / hard_score.detach().abs().clamp_min(1.0)
    hard_score = hard_score.clamp(0.0, 1.0)
    return 1.0 + strength * hard_score


def _sample_hardness(sample: TransitionSample | RolloutSample) -> float:
    """Cheap scalar used by weighted training order for hard-fault-heavy samples."""

    if isinstance(sample, RolloutSample):
        values = []
        for hard_targets, hard_count, hard_reduction, has_hard in zip(
            sample.hard_targets_post,
            sample.hard_count_post,
            sample.hard_reduction_targets,
            sample.has_hard_targets,
        ):
            if bool(has_hard.item()):
                positives = float(hard_targets.sum().item())
                count_mass = float(hard_count.sum().item())
                reduction = float(hard_reduction.abs().mean().item())
                values.append(positives + 0.25 * count_mass + 2.0 * reduction)
        return max(values) if values else 0.0
    if not bool(sample.has_hard_targets.item()):
        return 0.0
    positives = float(sample.hard_targets_post.sum().item())
    count_mass = float(sample.hard_count_post.sum().item())
    reduction = float(sample.hard_reduction_target.abs().mean().item())
    return positives + 0.25 * count_mass + 2.0 * reduction


def _training_indices(dataset: TPIDataset, config: dict, max_steps: int | None) -> list[int]:
    """Return one epoch of indices, optionally biased toward hard-fault-rich samples."""

    indices = list(range(len(dataset)))
    original_count = len(indices)
    strategy = str(config.get("train_sample_strategy", "shuffle")).lower()
    if strategy not in {"hard_weighted", "weighted_hard", "hard"}:
        random.shuffle(indices)
        return indices
    pool_max = int(config.get("hard_sampler_pool_max", 4096) or 0)
    if pool_max > 0 and len(indices) > pool_max:
        indices = random.sample(indices, pool_max)
    weights = [_sample_hardness(dataset[idx]) for idx in indices]
    max_weight = max(weights) if weights else 0.0
    if max_weight <= 0.0:
        random.shuffle(indices)
        return indices
    alpha = float(config.get("hard_sampler_alpha", 2.0))
    weights = [1.0 + alpha * (weight / max_weight) for weight in weights]
    replacement = bool(config.get("hard_sampler_replacement", True))
    sample_count = min(original_count, max_steps) if max_steps is not None else original_count
    if replacement:
        return random.choices(indices, weights=weights, k=sample_count)
    keyed = [(random.random() ** (1.0 / weight), idx) for idx, weight in zip(indices, weights)]
    keyed.sort(reverse=True)
    return [idx for _, idx in keyed[:sample_count]]


def compute_rollout_loss(
    batch: dict,
    model: TPIWorldModel,
    config: dict,
    pattern_enabled: bool,
    horizon: int,
) -> tuple[torch.Tensor, dict]:
    """Compute latent rollout loss by feeding predicted latents into the next step."""

    edge_src = batch["graph"].edge_src.to(batch["x_start"].device)
    edge_dst = batch["graph"].edge_dst.to(batch["x_start"].device)
    gate_type_ids = batch["graph"].gate_type_ids.to(batch["x_start"].device)
    z_state = model.online_encoder(batch["x_start"], edge_src, edge_dst, gate_type_ids)
    target_was_training = model.target_encoder.training
    model.target_encoder.eval()

    total = torch.zeros((), dtype=z_state.dtype, device=z_state.device)
    metric_totals: dict[str, float] = {
        "loss": 0.0,
        "jepa_loss": 0.0,
        "scoap_loss": 0.0,
        "delta_scoap_loss": 0.0,
        "hard_bce_loss": 0.0,
        "hard_rank_loss": 0.0,
        "hard_brier_loss": 0.0,
        "hard_soft_f1_loss": 0.0,
        "hard_count_loss": 0.0,
        "hard_reduction_loss": 0.0,
        "fc_loss": 0.0,
        "reward_loss": 0.0,
        "pattern_loss": 0.0,
        "return_loss": 0.0,
    }
    steps = min(int(horizon), len(batch["x_targets"]))
    coverage_scale = float(config.get("coverage_scale", 100.0))
    return_scale = float(config.get("return_scale", coverage_scale))
    return_gamma = float(config.get("return_gamma", config.get("discount_gamma", 1.0)))
    return_targets = discounted_return_targets(batch["delta_fault_coverages"], steps, return_gamma, return_scale)
    for step in range(steps):
        prev_state = batch["x_start"] if step == 0 else batch["x_targets"][step - 1]
        pred = model.predict_from_latent(
            z_state,
            batch["action_node_ids"][step],
            batch["action_type_ids"][step],
            batch["relation_features"][step],
        )
        with torch.no_grad():
            z_target = model.target_encoder(batch["x_targets"][step], edge_src, edge_dst, gate_type_ids)

        jepa_loss = F.mse_loss(pred["z_pred"], z_target)
        scoap_target = batch["x_targets"][step][:, SCOAP_START:SCOAP_END]
        scoap_loss = F.smooth_l1_loss(pred["scoap_pred"], scoap_target)
        delta_scoap_target = batch["x_targets"][step][:, SCOAP_START:SCOAP_END] - prev_state[:, SCOAP_START:SCOAP_END]
        delta_scoap_loss = F.smooth_l1_loss(pred["delta_scoap_pred"], delta_scoap_target)
        hard_targets = batch["hard_targets_post"][step]
        hard_bce = _hard_bce_loss(pred["hard_logits"], hard_targets, config)
        hard_rank = _hard_rank_loss(pred["hard_logits"], hard_targets, batch["hard_count_post"][step], config)
        hard_brier = _hard_brier_loss(pred["hard_logits"], hard_targets)
        hard_soft_f1 = _hard_soft_f1_loss(pred["hard_logits"], hard_targets, config)
        hard_count_loss = F.smooth_l1_loss(pred["hard_count_pred"], batch["hard_count_post"][step])
        hard_reduction_loss = F.smooth_l1_loss(pred["hard_reduction_pred"], batch["hard_reduction_targets"][step])
        reward_target = coverage_scale * batch["delta_fault_coverages"][step]
        reward_loss = F.smooth_l1_loss(pred["reward_pred"], reward_target)
        hard_weight = _hard_action_weight(batch["x_start"], batch["action_node_ids"][step], config)

        use_pattern = bool(
            pattern_enabled
            and float(config.get("lambda_pattern", 0.0)) > 0.0
            and batch["has_pattern_targets"][step].item()
        )
        if use_pattern:
            pattern_loss = F.smooth_l1_loss(pred["pattern_pred"], batch["delta_patterns"][step])
        else:
            pattern_loss = torch.zeros((), dtype=reward_loss.dtype, device=reward_loss.device)
        return_loss = F.smooth_l1_loss(pred["return_pred"], return_targets[step])
        weighted_reward_loss = hard_weight * reward_loss
        weighted_return_loss = hard_weight * return_loss
        step_loss = (
            float(config["lambda_jepa"]) * jepa_loss
            + float(config["lambda_scoap"]) * scoap_loss
            + float(config.get("lambda_delta_scoap", 0.0)) * delta_scoap_loss
            + float(config.get("lambda_hard", 0.0)) * hard_bce
            + float(config.get("lambda_hard_rank", 0.0)) * hard_rank
            + float(config.get("lambda_hard_brier", 0.0)) * hard_brier
            + float(config.get("lambda_hard_soft_f1", 0.0)) * hard_soft_f1
            + float(config.get("lambda_hard_count", 0.0)) * hard_count_loss
            + float(config.get("lambda_hard_reduction", 0.0)) * hard_reduction_loss
            + float(config["lambda_fc"]) * weighted_reward_loss
            + float(config["lambda_pattern"]) * pattern_loss
            + float(config.get("lambda_return", 0.0)) * weighted_return_loss
        )
        total = total + step_loss
        z_state = pred["z_pred"]

        metric_totals["loss"] += float(step_loss.detach().cpu().item())
        metric_totals["jepa_loss"] += float(jepa_loss.detach().cpu().item())
        metric_totals["scoap_loss"] += float(scoap_loss.detach().cpu().item())
        metric_totals["delta_scoap_loss"] += float(delta_scoap_loss.detach().cpu().item())
        metric_totals["hard_bce_loss"] += float(hard_bce.detach().cpu().item())
        metric_totals["hard_rank_loss"] += float(hard_rank.detach().cpu().item())
        metric_totals["hard_brier_loss"] += float(hard_brier.detach().cpu().item())
        metric_totals["hard_soft_f1_loss"] += float(hard_soft_f1.detach().cpu().item())
        metric_totals["hard_count_loss"] += float(hard_count_loss.detach().cpu().item())
        metric_totals["hard_reduction_loss"] += float(hard_reduction_loss.detach().cpu().item())
        metric_totals["fc_loss"] += float(reward_loss.detach().cpu().item())
        metric_totals["reward_loss"] += float(reward_loss.detach().cpu().item())
        metric_totals["pattern_loss"] += float(pattern_loss.detach().cpu().item())
        metric_totals["return_loss"] += float(return_loss.detach().cpu().item())
        metric_totals["hard_weight"] = metric_totals.get("hard_weight", 0.0) + float(hard_weight.detach().cpu().item())

    model.target_encoder.train(target_was_training)
    denom = max(1, steps)
    total = total / denom
    metrics = {key: value / denom for key, value in metric_totals.items()}
    metrics["rollout_steps"] = float(steps)
    return total, metrics


def rollout_horizon_for_epoch(epoch: int, config: dict) -> int:
    """Curriculum: train one-step first, then gradually increase rollout depth."""

    max_horizon = max(1, int(config.get("rollout_max_horizon", 1)))
    start_epoch = max(1, int(config.get("rollout_start_epoch", 1)))
    increase_every = max(1, int(config.get("rollout_increase_every", 1)))
    if epoch < start_epoch:
        return 1
    return min(max_horizon, 2 + (epoch - start_epoch) // increase_every)


def train_one_epoch(
    model: TPIWorldModel,
    dataset: TPIDataset,
    optimizer: torch.optim.Optimizer,
    config: dict,
    device: torch.device,
    pattern_enabled: bool,
    max_steps: int | None = None,
    horizon: int = 1,
) -> dict:
    """Train for one shuffled pass or until `max_steps` is reached."""

    model.train()
    indices = _training_indices(dataset, config, max_steps)
    totals: dict[str, float] = {}
    steps = 0
    for idx in indices:
        sample = dataset[idx]
        is_rollout = isinstance(sample, RolloutSample)
        batch = _rollout_sample_to_device(sample, device) if is_rollout else _sample_to_device(sample, device)
        optimizer.zero_grad(set_to_none=True)
        if is_rollout:
            loss, metrics = compute_rollout_loss(batch, model, config, pattern_enabled, horizon)
        else:
            loss, metrics = compute_loss(batch, model, config, pattern_enabled)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        update_ema(model.target_encoder, model.online_encoder, float(config["ema_decay"]))
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break
    return {key: value / max(1, steps) for key, value in totals.items()} | {"steps": float(steps)}


@torch.no_grad()
def evaluate(
    model: TPIWorldModel,
    dataset: TPIDataset,
    config: dict,
    device: torch.device,
    pattern_enabled: bool,
    max_steps: int | None = None,
    horizon: int = 1,
) -> dict:
    """Evaluate average losses on a dataset."""

    model.eval()
    totals: dict[str, float] = {}
    steps = 0
    for idx in range(len(dataset)):
        sample = dataset[idx]
        is_rollout = isinstance(sample, RolloutSample)
        batch = _rollout_sample_to_device(sample, device) if is_rollout else _sample_to_device(sample, device)
        if is_rollout:
            _, metrics = compute_rollout_loss(batch, model, config, pattern_enabled, horizon)
        else:
            _, metrics = compute_loss(batch, model, config, pattern_enabled)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break
    return {key: value / max(1, steps) for key, value in totals.items()} | {"steps": float(steps)}


def save_checkpoint(
    path: str | Path,
    model: TPIWorldModel,
    config: dict,
    feature_dim: int,
    relation_dim: int,
) -> None:
    """Save model weights and enough metadata to reload for planning."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "feature_dim": feature_dim,
            "relation_dim": relation_dim,
        },
        path,
    )


def _write_history(path: Path, rows: list[dict]) -> None:
    """Write training history as CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """CLI entry point for training."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tiny.json")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    device = _device_from_config(config)
    run_dir = Path(config["run_dir"])

    all_rows = load_labels(config["labels"])
    excluded = excluded_benchmarks_from_config(config)
    rows = filter_rows_by_excluded_benchmarks(all_rows, excluded)
    if excluded:
        removed = len(all_rows) - len(rows)
        print(
            "[train] excluded_eval_benchmarks="
            + ",".join(sorted(excluded))
            + f" removed_rows={removed} remaining_rows={len(rows)}"
        )
    if not rows:
        raise RuntimeError("No labels remain after benchmark exclusion")
    train_rows, val_rows, _ = split_by_benchmark(
        rows,
        int(config["seed"]),
        train_frac=float(config.get("train_frac", 0.70)),
        val_frac=float(config.get("val_frac", 0.15)),
    )
    config["lambda_score"] = 0.0
    config["score_pattern_weight"] = 0.0
    config.setdefault("lambda_delta_scoap", 0.3)
    config.setdefault("lambda_hard", 0.7)
    config.setdefault("lambda_hard_count", 0.1)
    config.setdefault("lambda_hard_reduction", 0.5)
    config.setdefault("hard_pos_weight_max", 20.0)
    config.setdefault("hard_loss", "bce")
    config.setdefault("lambda_hard_rank", 0.0)
    config.setdefault("lambda_hard_brier", 0.0)
    config.setdefault("lambda_hard_soft_f1", 0.0)
    config.setdefault("hard_soft_f1_eps", 1e-6)
    config.setdefault("hard_rank_margin", 0.2)
    config.setdefault("hard_negative_sample_ratio", 0)
    config.setdefault("hard_negative_mining", "random")
    config.setdefault("hard_head_type", "mlp")
    config.setdefault("encoder_type", "mean")
    config.setdefault("summary_mode", "global")
    config.setdefault("train_sample_strategy", "shuffle")
    pattern_enabled = _pattern_target_valid(train_rows)
    if not pattern_enabled or float(config.get("lambda_pattern", 0.0)) <= 0.0:
        config["lambda_pattern"] = 0.0
        print("[train] disabling pattern loss")
    if float(config.get("lambda_fc", 0.0)) > 0.0:
        print(
            "[train] note: delta_test_coverage is retained as a weak objective; "
            "the main hard-fault target is hard_reduction"
        )

    rollout_training = bool(config.get("rollout_training", True))
    dataset_cls = TPIRolloutDataset if rollout_training else TPIDataset
    dataset_kwargs = {
        "feature_mode": str(config.get("feature_mode", "basic")),
        "relation_mode": str(config.get("relation_mode", "basic")),
        "relation_depth": int(config.get("relation_depth", 8)),
        "state_update_mode": str(config.get("state_update_mode", "static")),
        "state_update_depth": int(config.get("state_update_depth", config.get("relation_depth", 8))),
        "real_fault_prior_path": config.get("real_fault_priors") or config.get("real_fault_prior_path"),
        "activation_prior_path": config.get("activation_priors") or config.get("activation_prior_path"),
        "cache_samples": bool(config.get("cache_samples", False)),
        "sample_cache_max_entries": int(config.get("sample_cache_max_entries", 0)) or None,
    }
    rollout_kwargs = (
        {
            "max_horizon": int(config.get("rollout_max_horizon", 1)),
            "validate_nodes": bool(config.get("validate_rollout_nodes", True)),
            "require_full_horizon": bool(config.get("require_full_horizon", False)),
        }
        if rollout_training
        else {}
    )
    train_rollout_kwargs = (
        {**rollout_kwargs, "repeat_to_max_specs": bool(config.get("repeat_train_samples", False))}
        if rollout_training
        else {}
    )
    train_set = dataset_cls(
        train_rows,
        max_specs=int(config.get("max_train_samples", 0)) or None,
        max_nodes=int(config.get("max_nodes", 0)) or None,
        **dataset_kwargs,
        **train_rollout_kwargs,
    )
    val_set = dataset_cls(
        val_rows,
        max_specs=int(config.get("max_val_samples", 0)) or None,
        max_nodes=int(config.get("max_nodes", 0)) or None,
        **dataset_kwargs,
        **rollout_kwargs,
    )
    if len(train_set) == 0:
        raise RuntimeError(f"Empty training dataset after filtering: train={len(train_set)}")
    if len(val_set) == 0:
        print("[train] warning: validation split is empty after max_nodes filtering; using train rows for sanity validation")
        val_set = dataset_cls(
            train_rows,
            max_specs=int(config.get("max_val_samples", 0)) or None,
            max_nodes=int(config.get("max_nodes", 0)) or None,
            **dataset_kwargs,
            **rollout_kwargs,
        )

    first = train_set[0]
    feature_dim = first.x_start.shape[1] if isinstance(first, RolloutSample) else first.x_pre.shape[1]
    relation_dim = (
        first.relation_features[0].shape[1]
        if isinstance(first, RolloutSample)
        else first.relation_features.shape[1]
    )
    print(
        f"[train] rows={len(rows)} train_rows={len(train_rows)} val_rows={len(val_rows)} "
        f"train_samples={len(train_set)} val_samples={len(val_set)} rollout_training={rollout_training} "
        f"feature_dim={feature_dim} relation_dim={relation_dim} "
        f"cache_samples={bool(config.get('cache_samples', False))} "
        f"sample_cache_max_entries={int(config.get('sample_cache_max_entries', 0)) or 'unlimited'}"
    )
    model = TPIWorldModel(
        feature_dim=feature_dim,
        latent_dim=int(config["latent_dim"]),
        encoder_layers=int(config["encoder_layers"]),
        action_type_dim=int(config["action_type_dim"]),
        dropout=float(config["dropout"]),
        head_context=bool(config.get("head_context", False)),
        relation_dim=relation_dim,
        edge_weight_mode=str(config.get("edge_weight_mode", "mean")),
        edge_keep_ratio=float(config.get("edge_keep_ratio", 1.0)),
        residual_dynamics=bool(config.get("residual_dynamics", False)),
        relation_gate=bool(config.get("relation_gate", False)),
        hard_head_type=str(config.get("hard_head_type", "mlp")),
        encoder_type=str(config.get("encoder_type", "mean")),
        summary_mode=str(config.get("summary_mode", "global")),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]))

    history: list[dict] = []
    best_val = float("inf")
    best_val_by_horizon: dict[int, float] = {}
    for epoch in range(1, int(config["epochs"]) + 1):
        horizon = rollout_horizon_for_epoch(epoch, config) if rollout_training else 1
        epoch_max_steps = (
            args.max_steps
            if args.max_steps is not None
            else (int(config.get("max_train_steps_per_epoch", 0)) or None)
        )
        train_metrics = train_one_epoch(
            model,
            train_set,
            optimizer,
            config,
            device,
            pattern_enabled,
            epoch_max_steps,
            horizon=horizon,
        )
        val_metrics = evaluate(
            model,
            val_set,
            config,
            device,
            pattern_enabled,
            max_steps=int(config.get("max_val_steps", 64)),
            horizon=horizon,
        )
        row = {
            "epoch": epoch,
            "horizon": horizon,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(
            f"[train] epoch={epoch} horizon={horizon} train_loss={row['train_loss']:.6f} "
            f"val_loss={row['val_loss']:.6f} train_steps={int(row['train_steps'])}"
        )
        save_checkpoint(run_dir / "latest.pt", model, config, feature_dim, relation_dim)
        if bool(config.get("save_epoch_checkpoints", False)):
            save_checkpoint(run_dir / f"epoch_{epoch:03d}.pt", model, config, feature_dim, relation_dim)
        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            save_checkpoint(run_dir / "best.pt", model, config, feature_dim, relation_dim)
        horizon_best = best_val_by_horizon.get(horizon, float("inf"))
        if row["val_loss"] < horizon_best:
            best_val_by_horizon[horizon] = row["val_loss"]
            save_checkpoint(run_dir / f"best_h{horizon}.pt", model, config, feature_dim, relation_dim)
            if horizon == int(config.get("rollout_max_horizon", horizon)):
                save_checkpoint(run_dir / "best_final_horizon.pt", model, config, feature_dim, relation_dim)
        _write_history(run_dir / "history.csv", history)
        if args.max_steps is not None:
            break


if __name__ == "__main__":
    main()
