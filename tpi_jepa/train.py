"""Train the minimal TPI-JEPA world model."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
from typing import Any

import torch
import torch.nn.functional as F

from .bench import parse_bench
from .dataset import RolloutSample, TPIDataset, TPIRolloutDataset, TransitionSample, split_by_benchmark
from .features import (
    SCOAP_END,
    SCOAP_START,
    action_type_to_id,
    make_action_relation_features,
    make_base_node_features,
    make_state_features,
)
from .graph import build_graph
from .labels import find_bench_path, load_labels
from .model import TPIWorldModel, update_ema
from .plan import _clip_latent_norms, set_real_fault_context
from .protocol import excluded_benchmarks_from_config, filter_rows_by_excluded_benchmarks, parse_benchmark_list


REGION_START = SCOAP_END


def _safe_float(value: Any, default: float = float("nan")) -> float:
    """Parse TSV numeric fields while preserving missing values as NaN."""

    if value in (None, "", "NA"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_tsv(path: str | Path) -> list[dict[str, str]]:
    """Read a tab-separated oracle action file."""

    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _oracle_group_key(row: dict[str, Any]) -> tuple[str, str, str]:
    """Group candidate actions by benchmark, state, and candidate generator."""

    return (
        str(row.get("benchmark_id", "")),
        str(row.get("state_id", "")),
        str(row.get("candidate_strategy", "")),
    )


def _canonical_oracle_action_type(value: Any) -> str:
    """Normalize backend and planner action names used in oracle prefixes."""

    key = str(value or "").strip().lower()
    aliases = {
        "control0": "control0",
        "cp0": "control0",
        "control1": "control1",
        "cp1": "control1",
        "observe": "observe",
        "op": "observe",
    }
    if key not in aliases:
        raise ValueError(f"unsupported action type in oracle prefix: {value!r}")
    return aliases[key]


def _oracle_prefix_from_row(row: dict[str, Any]) -> list[tuple[str, str]]:
    """Parse the action prefix shared by a counterfactual candidate group.

    Non-initial states must carry ``state_actions`` as JSON;
    ``prefix_sequence`` is accepted as a relabeler-compatible alias.  Each
    action may be a ``{\"net\": ..., \"type\": ...}`` object or a two-item
    ``[node, type]`` sequence.  Requiring an explicit prefix prevents two
    different planner states from being silently merged under one state id.
    """

    raw = row.get("state_actions")
    field = "state_actions"
    if raw in (None, ""):
        raw = row.get("prefix_sequence")
        field = "prefix_sequence"
    state_id = str(row.get("state_id", "")).strip()
    if raw in (None, ""):
        if state_id == "initial":
            return []
        raise ValueError(
            f"oracle state_id={state_id!r} requires JSON state_actions; "
            "prefix_sequence is accepted as a compatibility alias; "
            "only state_id='initial' may omit it"
        )
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid oracle {field} JSON for state_id={state_id!r}") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"oracle {field} must be a JSON list for state_id={state_id!r}")
    actions: list[tuple[str, str]] = []
    for index, item in enumerate(parsed):
        if isinstance(item, dict):
            node = item.get("net", item.get("node"))
            action_type = item.get("type")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            node, action_type = item
        else:
            raise ValueError(
                f"invalid oracle prefix action at index={index} state_id={state_id!r}: {item!r}"
            )
        node_text = str(node or "").strip()
        if not node_text:
            raise ValueError(f"empty oracle prefix node at index={index} state_id={state_id!r}")
        actions.append((node_text, _canonical_oracle_action_type(action_type)))
    return actions


def _oracle_prefix_actions(group: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Return and validate the one prefix shared by all candidates in a group."""

    if not group:
        return []
    expected = _oracle_prefix_from_row(group[0])
    for row in group[1:]:
        current = _oracle_prefix_from_row(row)
        if current != expected:
            key = _oracle_group_key(group[0])
            raise ValueError(f"oracle candidate group {key!r} contains inconsistent state-action prefixes")
    return expected


def _load_oracle_groups_one(
    path: str | Path,
    max_actions_per_group: int | None = None,
    forbidden_benchmarks: set[str] | None = None,
) -> list[list[dict[str, str]]]:
    """Load one backend-labeled oracle action file for ranking supervision."""

    forbidden = forbidden_benchmarks or set()
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    skipped_forbidden: Counter[str] = Counter()
    for row in _read_tsv(path):
        benchmark_id = str(row.get("benchmark_id", "")).strip()
        if benchmark_id in forbidden:
            skipped_forbidden[benchmark_id] += 1
            continue
        delta = _safe_float(row.get("oracle_delta_tc"))
        if not math.isfinite(delta):
            continue
        grouped[_oracle_group_key(row)].append(row)

    if skipped_forbidden:
        preview = ",".join(f"{bid}:{count}" for bid, count in skipped_forbidden.most_common(8))
        if len(skipped_forbidden) > 8:
            preview += f",...(+{len(skipped_forbidden) - 8})"
        print(
            f"[train] oracle skipped forbidden rows path={path} "
            f"rows={sum(skipped_forbidden.values())} benchmarks={preview}",
            flush=True,
        )

    groups: list[list[dict[str, str]]] = []
    for _, group in sorted(grouped.items()):
        group = sorted(group, key=lambda row: int(float(row.get("candidate_rank") or 0)))
        if max_actions_per_group is not None and max_actions_per_group > 0:
            group = group[:max_actions_per_group]
        if group:
            _oracle_prefix_actions(group)
            groups.append(group)
    if not groups:
        raise ValueError(f"no oracle groups with finite actions in {path} after forbidden-benchmark filtering")
    return groups


def load_oracle_groups(
    path: Any,
    max_actions_per_group: int | None = None,
    forbidden_benchmarks: set[str] | None = None,
) -> list[list[dict[str, str]]]:
    """Load one or more oracle action files.

    Configs may pass a single TSV path, a list of paths, or a list of objects
    like {"path": "...", "repeat": 4}. Repeating a file repeats its groups in
    the sampler, which is useful for small non-target auxiliary sets.
    """

    if isinstance(path, (list, tuple)):
        groups: list[list[dict[str, str]]] = []
        for item in path:
            repeat = 1
            item_path: Any = item
            if isinstance(item, dict):
                item_path = item.get("path")
                repeat = max(1, int(item.get("repeat", 1) or 1))
            if not item_path:
                raise ValueError(f"invalid oracle_actions entry: {item!r}")
            item_groups = _load_oracle_groups_one(item_path, max_actions_per_group, forbidden_benchmarks)
            for _ in range(repeat):
                groups.extend(item_groups)
        if not groups:
            raise ValueError(f"no oracle groups with finite actions in {path!r}")
        return groups
    return _load_oracle_groups_one(path, max_actions_per_group, forbidden_benchmarks)


def load_config(path: str | Path) -> dict:
    """Load a JSON training config."""

    with Path(path).open() as f:
        return json.load(f)


def _device_from_config(config: dict) -> torch.device:
    """Use the configured device and refuse silent deep-learning CPU fallback."""

    requested = str(config.get("device", "cpu"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for training but is unavailable")
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
    typed_marginal_loss = (
        F.smooth_l1_loss(out["typed_marginal_pred"], reward_target)
        if "typed_marginal_pred" in out
        else reward_loss.new_zeros(())
    )
    typed_return_loss = (
        F.smooth_l1_loss(out["typed_return_pred"], return_scale * batch["delta_fault_coverage"])
        if "typed_return_pred" in out
        else reward_loss.new_zeros(())
    )
    typed_sa_reduction_loss = (
        F.smooth_l1_loss(out["typed_sa_reduction_pred"], batch["hard_reduction_target"][1:3])
        if "typed_sa_reduction_pred" in out and _hard_reduction_mask(batch)
        else reward_loss.new_zeros(())
    )
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
        + float(config.get("lambda_typed_marginal", 0.0)) * hard_weight * typed_marginal_loss
        + float(config.get("lambda_typed_return", 0.0)) * hard_weight * typed_return_loss
        + float(config.get("lambda_typed_sa_reduction", 0.0)) * typed_sa_reduction_loss
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
        "typed_marginal_loss": float(typed_marginal_loss.detach().cpu().item()),
        "typed_return_loss": float(typed_return_loss.detach().cpu().item()),
        "typed_sa_reduction_loss": float(typed_sa_reduction_loss.detach().cpu().item()),
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
        "typed_marginal_loss": 0.0,
        "typed_return_loss": 0.0,
        "typed_sa_reduction_loss": 0.0,
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
            sequence_step=step,
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
        typed_marginal_loss = (
            F.smooth_l1_loss(pred["typed_marginal_pred"], reward_target)
            if "typed_marginal_pred" in pred
            else reward_loss.new_zeros(())
        )
        typed_return_loss = (
            F.smooth_l1_loss(pred["typed_return_pred"], return_targets[step])
            if "typed_return_pred" in pred
            else reward_loss.new_zeros(())
        )
        typed_sa_reduction_loss = (
            F.smooth_l1_loss(pred["typed_sa_reduction_pred"], batch["hard_reduction_targets"][step][1:3])
            if "typed_sa_reduction_pred" in pred and bool(batch["has_hard_targets"][step].item())
            else reward_loss.new_zeros(())
        )
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
            + float(config.get("lambda_typed_marginal", 0.0)) * hard_weight * typed_marginal_loss
            + float(config.get("lambda_typed_return", 0.0)) * hard_weight * typed_return_loss
            + float(config.get("lambda_typed_sa_reduction", 0.0)) * typed_sa_reduction_loss
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
        metric_totals["typed_marginal_loss"] += float(typed_marginal_loss.detach().cpu().item())
        metric_totals["typed_return_loss"] += float(typed_return_loss.detach().cpu().item())
        metric_totals["typed_sa_reduction_loss"] += float(typed_sa_reduction_loss.detach().cpu().item())
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
    explicit_schedule = config.get("rollout_horizon_schedule")
    if explicit_schedule:
        if isinstance(explicit_schedule, str):
            values = [int(value.strip()) for value in explicit_schedule.split(",") if value.strip()]
        else:
            values = [int(value) for value in explicit_schedule]
        if not values or any(value <= 0 for value in values):
            raise ValueError("rollout_horizon_schedule must contain positive integers")
        return min(max_horizon, values[min(max(1, epoch) - 1, len(values) - 1)])
    start_epoch = max(1, int(config.get("rollout_start_epoch", 1)))
    increase_every = max(1, int(config.get("rollout_increase_every", 1)))
    if epoch < start_epoch:
        return 1
    start_horizon = max(2, int(config.get("rollout_start_horizon", 2)))
    horizon_increment = max(1, int(config.get("rollout_horizon_increment", 1)))
    increments = (epoch - start_epoch) // increase_every
    return min(max_horizon, start_horizon + horizon_increment * increments)


def initialize_from_checkpoint(
    model: TPIWorldModel,
    checkpoint_path: str | Path,
    device: torch.device,
    *,
    feature_dim: int,
    relation_dim: int,
    strict: bool = True,
) -> None:
    """Initialize a training model from a shape-compatible planner checkpoint."""

    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    saved_feature_dim = int(checkpoint.get("feature_dim", feature_dim))
    saved_relation_dim = int(checkpoint.get("relation_dim", relation_dim))
    if saved_feature_dim != int(feature_dim):
        raise ValueError(
            f"init checkpoint feature_dim={saved_feature_dim} does not match dataset feature_dim={feature_dim}"
        )
    if saved_relation_dim != int(relation_dim):
        raise ValueError(
            f"init checkpoint relation_dim={saved_relation_dim} does not match dataset relation_dim={relation_dim}"
        )
    model.load_state_dict(checkpoint["model_state"], strict=strict)


def configure_trainable_parameters(model: TPIWorldModel, mode: str = "all") -> list[torch.nn.Parameter]:
    """Select which model parameters may change during a fine-tuning run."""

    normalized = str(mode or "all").strip().lower()
    if normalized == "all":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    elif normalized in {"rollout_dynamics", "dynamics"}:
        prefixes = ("action_encoder.", "dynamics.")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(prefixes))
    elif normalized in {"utility_posttrain", "world_model_posttrain"}:
        prefixes = (
            "action_encoder.",
            "dynamics.",
            "q_head.",
            "q_node_head.",
            "q_type_head.",
            "reward_head.",
            "return_head.",
            "hard_reduction_head.",
            "typed_utility_head.",
            "typed_cone_utility_head.",
        )
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(prefixes))
    elif normalized in {"typed_utility_only", "typed_head_only"}:
        # Keep the production encoder, dynamics, and legacy utility ordering
        # bit-identical.  Real ATPG labels may only learn a bounded residual
        # through the action-type-conditioned auxiliary head.
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(
                name.startswith("typed_utility_head.")
                or name.startswith("typed_cone_utility_head.")
            )
    elif normalized in {"typed_experts_only", "typed_moe_experts_only"}:
        # Preserve the selected shared cone head exactly and learn only the
        # zero-initialized type-gated residual experts.  This gives the new
        # within-type ranking objective a bounded correction path instead of
        # allowing it to erase an already validated b15 policy.
        prefixes = (
            "typed_cone_utility_head.expert_gate.",
            "typed_cone_utility_head.type_experts.",
        )
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(prefixes))
    elif normalized in {"typed_horizon_only", "typed_horizon_experts_only"}:
        # The inherited Round8 shared and type-expert paths stay bit-identical;
        # only the new zero-initialized sequence-position experts can move.
        prefix = "typed_cone_utility_head.horizon_experts."
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(prefix))
    elif normalized in {"typed_return_rank_only", "typed_return_adapter_only"}:
        # Preserve every incumbent prediction except the isolated, initially
        # zero return-ranking residual introduced for real-ATPG supervision.
        prefix = "typed_cone_utility_head.return_rank_experts."
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(prefix))
    elif normalized in {
        "typed_return_horizon_only",
        "typed_horizon_return_only",
        "typed_return_horizon_adapter_only",
    }:
        # Retain the learned return ranker exactly and learn only a bounded
        # sequence-position correction from ultra-long real-ATPG prefixes.
        prefix = "typed_cone_utility_head.horizon_return_rank_experts."
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(prefix))
    elif normalized in {
        "typed_return_late_horizon_only",
        "typed_late_horizon_return_only",
        "typed_return_late_horizon_adapter_only",
    }:
        # Preserve every b15-selected parameter and train only the branch
        # whose structural gate is zero through sequence step 277.
        prefix = "typed_cone_utility_head.late_return_rank_experts."
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(prefix))
    elif normalized in {
        "typed_return_late_type_only",
        "typed_late_type_return_only",
        "typed_return_late_type_adapter_only",
    }:
        # A tiny family-level adapter learns only CP0/CP1/OP phase shifts;
        # inherited node ordering and every b15 prediction stay frozen.
        prefix = "typed_cone_utility_head.late_type_calibrator."
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(prefix))
    elif normalized in {
        "typed_return_late_control_only",
        "typed_late_control_return_only",
        "typed_return_late_control_adapter_only",
    }:
        # Learn only a shared CP0/CP1-vs-OP shift.  Polarity and node ranking
        # remain exactly those of the inherited b15-selected model.
        prefix = "typed_cone_utility_head.late_control_calibrator."
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(prefix))
    elif normalized in {"typed_marginal_rank_only", "typed_marginal_adapter_only"}:
        # Learn only the sequence-conditioned marginal-TC residual; the
        # Round8 MoE and all other utility voters remain bit-identical.
        prefix = "typed_cone_utility_head.marginal_rank_experts."
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(prefix))
    else:
        raise ValueError(
            f"unsupported trainable_modules={mode!r}; expected 'all', 'rollout_dynamics', "
            "'utility_posttrain', 'typed_utility_only', 'typed_experts_only', "
            "'typed_horizon_only', 'typed_return_rank_only', 'typed_return_horizon_only', "
            "'typed_return_late_horizon_only', 'typed_return_late_type_only', "
            "'typed_return_late_control_only', "
            "or 'typed_marginal_rank_only'"
        )
    selected = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not selected:
        raise ValueError(f"trainable_modules={mode!r} selected no parameters")
    return selected


def update_ema_if_encoder_trainable(model: TPIWorldModel, decay: float) -> None:
    """Advance the JEPA target only when the online encoder is being optimized."""

    if any(parameter.requires_grad for parameter in model.online_encoder.parameters()):
        update_ema(model.target_encoder, model.online_encoder, decay)


def _oracle_graph_and_base_for(
    benchmark_id: str,
    graph_cache: dict[str, Any],
    base_cache: dict[str, torch.Tensor],
    config: dict,
) -> tuple[Any, torch.Tensor]:
    """Build and cache graph/base features for oracle action groups."""

    if benchmark_id not in graph_cache:
        set_real_fault_context(benchmark_id, config.get("real_fault_priors"), config.get("activation_priors"))
        graph_cache[benchmark_id] = build_graph(parse_bench(find_bench_path(benchmark_id)))
    if benchmark_id not in base_cache:
        graph = graph_cache[benchmark_id]
        base_cache[benchmark_id] = make_base_node_features(
            graph,
            str(config.get("feature_mode", "basic")),
            benchmark_id=benchmark_id,
            real_fault_prior_path=config.get("real_fault_priors") or config.get("real_fault_prior_path"),
            activation_prior_path=config.get("activation_priors") or config.get("activation_prior_path"),
        )
    return graph_cache[benchmark_id], base_cache[benchmark_id]


def _predict_oracle_group_scores(
    model: TPIWorldModel,
    config: dict,
    group: list[dict[str, str]],
    graph_cache: dict[str, Any],
    base_cache: dict[str, torch.Tensor],
    device: torch.device,
    prefix_latent_cache: dict[tuple[str, tuple[tuple[str, str], ...]], torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Score candidate actions after replaying their shared planner prefix."""

    benchmark_id = str(group[0]["benchmark_id"])
    graph, base_features = _oracle_graph_and_base_for(benchmark_id, graph_cache, base_cache, config)
    relation_mode = str(config.get("relation_mode", "basic"))
    relation_depth = int(config.get("relation_depth", 8))
    node_ids = {name: idx for idx, name in enumerate(graph.node_names)}
    prefix = _oracle_prefix_actions(group)
    prefix_key = (benchmark_id, tuple(prefix))
    z_state = prefix_latent_cache.get(prefix_key) if prefix_latent_cache is not None else None
    if z_state is None:
        x_state = make_state_features(graph, [], base_features).to(device)
        z_state = model.online_encoder(
            x_state,
            graph.edge_src.to(device),
            graph.edge_dst.to(device),
            graph.gate_type_ids.to(device),
        )
        latent_clip_ratio = float(config.get("oracle_latent_norm_clip_ratio", 0.0))
        latent_norm_limit = None
        if latent_clip_ratio > 0.0:
            latent_norm_limit = float(z_state.norm(dim=1).median().item()) * latent_clip_ratio
        if prefix:
            # The prefix is context, not a second rollout-training objective.
            # Stop gradients through it by default so long prefixes do not
            # retain a large graph.  Candidate heads below remain trainable.
            detach_prefix = bool(config.get("oracle_prefix_detach", True))
            for prefix_step, (prefix_node, prefix_type) in enumerate(prefix):
                if prefix_node not in node_ids:
                    raise ValueError(f"prefix node {prefix_node!r} not found in {benchmark_id}")
                prefix_node_id = node_ids[prefix_node]
                relation = make_action_relation_features(
                    graph,
                    prefix_node_id,
                    relation_mode,
                    relation_depth,
                ).to(device)
                prefix_pred = model.predict_from_latent(
                    z_state,
                    prefix_node_id,
                    action_type_to_id(prefix_type),
                    relation,
                    include_aux_heads=False,
                    sequence_step=prefix_step,
                )
                z_state = _clip_latent_norms(prefix_pred["z_pred"], latent_norm_limit)
                if detach_prefix:
                    z_state = z_state.detach()
        if prefix_latent_cache is not None:
            prefix_latent_cache[prefix_key] = z_state.detach()
    coverage_scale = float(config.get("coverage_scale", 100.0))
    score_lists: dict[str, list[torch.Tensor]] = {
        "q_pred": [],
        "score_pred": [],
        "reward_pred": [],
        "return_pred": [],
        "guarded_reward": [],
        "hard_reduction_total_pred": [],
        "hybrid_pred": [],
        "derived_hard_reduction_total_pred": [],
        "derived_hard_reduction_hybrid_pred": [],
        "typed_marginal_pred": [],
        "typed_return_pred": [],
        "typed_sa_reduction_total_pred": [],
        "typed_sa0_reduction_pred": [],
        "typed_sa1_reduction_pred": [],
    }
    for row in group:
        node = row["node"]
        action_type = _canonical_oracle_action_type(row["type"])
        if node not in node_ids:
            raise ValueError(f"node {node!r} from oracle TSV not found in {benchmark_id}")
        action_node_id = node_ids[node]
        relation = make_action_relation_features(graph, action_node_id, relation_mode, relation_depth).to(device)
        pred = model.predict_from_latent(
            z_state,
            action_node_id,
            action_type_to_id(action_type),
            relation,
            include_aux_heads=False,
            sequence_step=len(prefix),
        )
        q_pred = pred["q_pred"]
        reward_pred = pred["reward_pred"]
        return_pred = pred["return_pred"]
        hard_reduction_pred = pred["hard_reduction_pred"].view(-1)
        hard_reduction_total = hard_reduction_pred[0] if hard_reduction_pred.numel() > 0 else reward_pred.new_zeros(())
        derived_reduction = pred.get("derived_hard_reduction_pred")
        derived_reduction = (
            derived_reduction.view(-1)
            if derived_reduction is not None
            else hard_reduction_pred.new_zeros(3)
        )
        derived_hard_reduction_total = (
            derived_reduction[0] if derived_reduction.numel() > 0 else reward_pred.new_zeros(())
        )
        score_lists["q_pred"].append(q_pred)
        score_lists["score_pred"].append(q_pred)
        score_lists["reward_pred"].append(reward_pred)
        score_lists["return_pred"].append(return_pred)
        score_lists["guarded_reward"].append(torch.minimum(reward_pred, return_pred))
        score_lists["hard_reduction_total_pred"].append(hard_reduction_total)
        score_lists["hybrid_pred"].append(return_pred + reward_pred + hard_reduction_total * coverage_scale)
        score_lists["derived_hard_reduction_total_pred"].append(derived_hard_reduction_total)
        score_lists["derived_hard_reduction_hybrid_pred"].append(derived_hard_reduction_total * coverage_scale)
        typed_marginal = pred.get("typed_marginal_pred", reward_pred)
        typed_return = pred.get("typed_return_pred", return_pred)
        typed_sa = pred.get("typed_sa_reduction_pred")
        if typed_sa is None:
            typed_sa0 = hard_reduction_pred[1] if hard_reduction_pred.numel() > 1 else hard_reduction_total
            typed_sa1 = hard_reduction_pred[2] if hard_reduction_pred.numel() > 2 else hard_reduction_total
        else:
            typed_sa = typed_sa.view(-1)
            typed_sa0 = typed_sa[0]
            typed_sa1 = typed_sa[1]
        typed_sa_total = 0.5 * (typed_sa0 + typed_sa1)
        score_lists["typed_marginal_pred"].append(typed_marginal)
        score_lists["typed_return_pred"].append(typed_return)
        score_lists["typed_sa_reduction_total_pred"].append(typed_sa_total)
        score_lists["typed_sa0_reduction_pred"].append(typed_sa0)
        score_lists["typed_sa1_reduction_pred"].append(typed_sa1)
    return {field: torch.stack(values) for field, values in score_lists.items() if values}


def _oracle_score(scores: dict[str, torch.Tensor], field: str) -> torch.Tensor:
    """Select the configured score tensor for oracle ranking."""

    if field not in scores:
        valid = ", ".join(sorted(scores))
        raise ValueError(f"unsupported oracle_ranking_score_field {field!r}; valid fields: {valid}")
    return scores[field]


def _oracle_pairwise_rank_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    min_delta: float,
    temperature: float,
    mode: str = "all",
    hard_negative_topk: int = 0,
    positive_topk: int = 1,
    max_pairs: int = 0,
) -> tuple[torch.Tensor, int]:
    """Pairwise logistic ranking loss against backend-labeled delta-TC order."""

    pairs: list[tuple[int, int, float]] = []
    temp = max(1e-6, float(temperature))
    mode = str(mode or "all").lower()
    min_delta_value = float(min_delta)
    if mode == "all":
        for i in range(targets.numel()):
            for j in range(i + 1, targets.numel()):
                diff = float((targets[i] - targets[j]).detach().cpu().item())
                if abs(diff) < min_delta_value:
                    continue
                if diff > 0.0:
                    pairs.append((i, j, 1.0))
                else:
                    pairs.append((j, i, 1.0))
    elif mode in {"hard_topk", "best_vs_hard_topk"}:
        target_order = torch.argsort(targets.detach(), descending=True).cpu().tolist()
        pred_order = torch.argsort(preds.detach(), descending=True).cpu().tolist()
        pos_count = max(1, int(positive_topk or 1))
        hard_count = int(hard_negative_topk or 0)
        hard_pool = pred_order[:hard_count] if hard_count > 0 else pred_order
        for pos_idx in target_order[:pos_count]:
            negs = [
                neg_idx
                for neg_idx in hard_pool
                if neg_idx != pos_idx and float((targets[pos_idx] - targets[neg_idx]).detach().cpu().item()) >= min_delta_value
            ]
            if mode == "best_vs_hard_topk" and max_pairs > 0:
                per_positive_limit = max(1, int(max_pairs) // pos_count)
                if len(negs) < per_positive_limit:
                    seen = set(negs)
                    for neg_idx in pred_order:
                        if neg_idx == pos_idx or neg_idx in seen:
                            continue
                        diff = float((targets[pos_idx] - targets[neg_idx]).detach().cpu().item())
                        if diff < min_delta_value:
                            continue
                        negs.append(neg_idx)
                        seen.add(neg_idx)
                        if len(negs) >= per_positive_limit:
                            break
            elif not negs:
                negs = [
                    neg_idx
                    for neg_idx in pred_order
                    if neg_idx != pos_idx and float((targets[pos_idx] - targets[neg_idx]).detach().cpu().item()) >= min_delta_value
                ]
            pairs.extend((pos_idx, neg_idx, 1.0) for neg_idx in negs)
    else:
        raise ValueError(f"unsupported oracle_pairwise_mode {mode!r}")

    if max_pairs > 0 and len(pairs) > max_pairs:
        pairs = random.sample(pairs, k=max_pairs)
    if not pairs:
        return torch.zeros((), dtype=preds.dtype, device=preds.device), 0
    losses = [F.softplus(-(preds[left] - preds[right]) / temp) for left, right, _ in pairs]
    return torch.stack(losses).mean(), len(pairs)


def _oracle_same_type_rank_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    action_types: list[str],
    min_delta: float,
    temperature: float,
    mode: str = "all",
    hard_negative_topk: int = 0,
    positive_topk: int = 1,
    max_pairs: int = 0,
) -> tuple[torch.Tensor, int]:
    """Rank nodes inside each action type before aggregating pair losses.

    Cross-type comparisons are already handled by the ordinary oracle loss.
    This auxiliary term prevents those easier comparisons from overwhelming
    the harder node-ordering signal once the policy has learned the right
    CP0/CP1/OP family.
    """

    if len(action_types) != preds.numel():
        raise ValueError("action_types length must match prediction count")
    weighted_losses: list[torch.Tensor] = []
    total_pairs = 0
    canonical_types = [_canonical_oracle_action_type(value) for value in action_types]
    for action_type in sorted(set(canonical_types)):
        indices = [index for index, value in enumerate(canonical_types) if value == action_type]
        if len(indices) < 2:
            continue
        index_tensor = torch.tensor(indices, dtype=torch.long, device=preds.device)
        loss, pair_count = _oracle_pairwise_rank_loss(
            preds.index_select(0, index_tensor),
            targets.index_select(0, index_tensor),
            min_delta,
            temperature,
            mode=mode,
            hard_negative_topk=hard_negative_topk,
            positive_topk=positive_topk,
            max_pairs=max_pairs,
        )
        if pair_count > 0:
            weighted_losses.append(loss * float(pair_count))
            total_pairs += pair_count
    if total_pairs == 0:
        return torch.zeros((), dtype=preds.dtype, device=preds.device), 0
    return torch.stack(weighted_losses).sum() / float(total_pairs), total_pairs


def _oracle_rank_weight(config: dict, epoch: int) -> float:
    """Return the current oracle weight after warmup and linear ramp."""

    target = float(config.get("lambda_oracle_rank", 0.0))
    return _oracle_ramped_weight(config, epoch, target)


def _oracle_ramped_weight(config: dict, epoch: int, target: float) -> float:
    """Return an oracle auxiliary loss weight after warmup and linear ramp."""

    if target <= 0.0:
        return 0.0
    warmup = max(0, int(config.get("oracle_warmup_epochs", 0) or 0))
    ramp = max(0, int(config.get("oracle_ramp_epochs", 0) or 0))
    if epoch <= warmup:
        return 0.0
    if ramp <= 0:
        return target
    ramp_step = min(ramp, max(1, epoch - warmup))
    return target * float(ramp_step) / float(ramp)


def _q_rank_weight(config: dict, epoch: int) -> float:
    """Return Q ranking weight, falling back to legacy oracle rank weight."""

    if "lambda_q_rank" in config:
        return _oracle_ramped_weight(config, epoch, float(config.get("lambda_q_rank", 0.0)))
    return _oracle_rank_weight(config, epoch)


def _q_same_type_rank_weight(config: dict, epoch: int) -> float:
    """Return the within-action-type node ranking weight."""

    return _oracle_ramped_weight(config, epoch, float(config.get("lambda_same_type_rank", 0.0)))


def _aux_rank_weight(config: dict, epoch: int) -> float:
    """Return the auxiliary score-head ranking weight."""

    return _oracle_ramped_weight(config, epoch, float(config.get("lambda_aux_rank", 0.0)))


def _aux_same_type_rank_weight(config: dict, epoch: int) -> float:
    """Return auxiliary within-type ranking weight for an independent voter."""

    return _oracle_ramped_weight(
        config,
        epoch,
        float(config.get("lambda_aux_same_type_rank", 0.0)),
    )


def _q_value_weight(config: dict, epoch: int) -> float:
    """Return Q value-regression weight."""

    return _oracle_ramped_weight(
        config,
        epoch,
        float(config.get("lambda_q_value", config.get("lambda_oracle_value", 0.0))),
    )


def _q_candidate_weight(config: dict, epoch: int) -> float:
    """Return candidate-list softmax loss weight."""

    return _oracle_ramped_weight(config, epoch, float(config.get("lambda_candidate", 0.0)))


def _q_action_type_weight(config: dict, epoch: int) -> float:
    """Return tie-aware CP0/CP1/OP listwise loss weight."""

    return _oracle_ramped_weight(
        config,
        epoch,
        float(config.get("lambda_action_type_rank", 0.0)),
    )


def _q_action_family_weight(config: dict, epoch: int) -> float:
    """Return tie-aware Control-vs-Observe loss weight."""

    return _oracle_ramped_weight(
        config,
        epoch,
        float(config.get("lambda_action_family_rank", 0.0)),
    )


def _q_ndcg_weight(config: dict, epoch: int) -> float:
    """Return top-heavy NDCG-style list loss weight."""

    return _oracle_ramped_weight(config, epoch, float(config.get("lambda_ndcg_rank", 0.0)))


def _q_same_type_ndcg_weight(config: dict, epoch: int) -> float:
    """Return top-heavy list-ranking weight inside each action type."""

    return _oracle_ramped_weight(
        config,
        epoch,
        float(config.get("lambda_same_type_ndcg_rank", 0.0)),
    )


def _q_conservative_weight(config: dict, epoch: int) -> float:
    """Return conservative score penalty weight."""

    return _oracle_ramped_weight(config, epoch, float(config.get("lambda_conservative_q", 0.0)))


def _q_context_weight(config: dict, epoch: int) -> float:
    """Return candidate-context relative-shape loss weight."""

    return _oracle_ramped_weight(config, epoch, float(config.get("lambda_context_rank", 0.0)))


def _q_sa_value_weight(config: dict, epoch: int) -> float:
    """Return prefix-oracle SA0/SA1 reduction regression weight."""

    return _oracle_ramped_weight(config, epoch, float(config.get("lambda_oracle_sa_value", 0.0)))


def _oracle_sa_reduction_targets(
    group: list[dict[str, str]],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build normalized SA0/SA1 hard-fault reduction targets and validity mask."""

    targets: list[list[float]] = []
    masks: list[list[bool]] = []
    for row in group:
        row_targets: list[float] = []
        row_masks: list[bool] = []
        for suffix in ("sa0", "sa1"):
            reduction = _safe_float(row.get(f"oracle_hard_reduction_{suffix}"))
            prefix_count = _safe_float(row.get(f"prefix_undetected_{suffix}_count"))
            # A polarity absent from the prefix fault set has no meaningful
            # fractional-reduction target.  Treating its zero denominator as
            # one turns fault-universe changes into a large, spurious label
            # (especially for sparse SA0 states), so exclude it from the
            # polarity loss instead.
            valid = (
                math.isfinite(reduction)
                and math.isfinite(prefix_count)
                and prefix_count > 0.0
            )
            row_targets.append(max(-1.0, min(1.0, reduction / prefix_count)) if valid else 0.0)
            row_masks.append(valid)
        targets.append(row_targets)
        masks.append(row_masks)
    return (
        torch.tensor(targets, dtype=dtype, device=device),
        torch.tensor(masks, dtype=torch.bool, device=device),
    )


def _oracle_candidate_loss(preds: torch.Tensor, targets: torch.Tensor, config: dict) -> torch.Tensor:
    """Listwise candidate loss that pushes probability mass toward high-delta-TC actions."""

    target_temp = max(1e-6, float(config.get("candidate_target_temperature", 1.0)))
    pred_temp = max(1e-6, float(config.get("candidate_pred_temperature", 1.0)))
    target_prob = torch.softmax(targets.detach() / target_temp, dim=0)
    pred_log_prob = torch.log_softmax(preds / pred_temp, dim=0)
    return -(target_prob * pred_log_prob).sum()


def _oracle_action_type_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    action_types: list[str],
    config: dict,
) -> tuple[torch.Tensor, int]:
    """Tie-aware listwise loss over CP0, CP1, and OP families.

    ATPG delta-TC labels are quantized by the finite fault universe, so several
    candidate actions frequently share the exact best value.  Hard ``argmax``
    type labels turn those ties into TSV-order noise.  Here each type is first
    represented by its best real target and a smooth maximum of model scores;
    a soft target distribution then preserves exact ties instead of inventing
    a winner.  Log-mean-exp removes bias when type pools have different sizes.
    """

    if len(action_types) != preds.numel():
        raise ValueError("action_types length must match prediction count")
    canonical_types = [_canonical_oracle_action_type(value) for value in action_types]
    unique_types = [
        action_type
        for action_type in ("control0", "control1", "observe")
        if action_type in canonical_types
    ]
    if len(unique_types) < 2:
        return torch.zeros((), dtype=preds.dtype, device=preds.device), 0

    aggregate_temp = max(
        1e-6,
        float(config.get("oracle_action_type_aggregate_temperature", 0.10)),
    )
    type_preds: list[torch.Tensor] = []
    type_targets: list[torch.Tensor] = []
    for action_type in unique_types:
        indices = [index for index, value in enumerate(canonical_types) if value == action_type]
        index_tensor = torch.tensor(indices, dtype=torch.long, device=preds.device)
        family_preds = preds.index_select(0, index_tensor)
        smooth_max = aggregate_temp * (
            torch.logsumexp(family_preds / aggregate_temp, dim=0)
            - math.log(len(indices))
        )
        type_preds.append(smooth_max)
        type_targets.append(targets.index_select(0, index_tensor).max().detach())

    pred_temp = max(1e-6, float(config.get("oracle_action_type_pred_temperature", 0.35)))
    target_temp = max(1e-6, float(config.get("oracle_action_type_target_temperature", 0.025)))
    pred_vector = torch.stack(type_preds)
    target_vector = torch.stack(type_targets)
    target_prob = torch.softmax(target_vector / target_temp, dim=0)
    pred_log_prob = torch.log_softmax(pred_vector / pred_temp, dim=0)
    return -(target_prob * pred_log_prob).sum(), len(unique_types)


def _oracle_action_family_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    action_types: list[str],
    config: dict,
) -> tuple[torch.Tensor, int]:
    """Tie-aware binary loss over Control (CP0/CP1) versus Observe.

    CP0 and CP1 are intentionally pooled before both target and prediction
    aggregation.  The objective can correct a late OP/control imbalance while
    providing no incentive to overfit forcing polarity on a small circuit set.
    """

    if len(action_types) != preds.numel():
        raise ValueError("action_types length must match prediction count")
    canonical_types = [_canonical_oracle_action_type(value) for value in action_types]
    family_indices = {
        "control": [index for index, value in enumerate(canonical_types) if value != "observe"],
        "observe": [index for index, value in enumerate(canonical_types) if value == "observe"],
    }
    if any(not indices for indices in family_indices.values()):
        return torch.zeros((), dtype=preds.dtype, device=preds.device), 0

    aggregate_temp = max(
        1e-6,
        float(config.get("oracle_action_family_aggregate_temperature", 0.10)),
    )
    family_preds: list[torch.Tensor] = []
    family_targets: list[torch.Tensor] = []
    for family in ("control", "observe"):
        indices = family_indices[family]
        index_tensor = torch.tensor(indices, dtype=torch.long, device=preds.device)
        selected_preds = preds.index_select(0, index_tensor)
        family_preds.append(
            aggregate_temp
            * (
                torch.logsumexp(selected_preds / aggregate_temp, dim=0)
                - math.log(len(indices))
            )
        )
        family_targets.append(targets.index_select(0, index_tensor).max().detach())

    pred_temp = max(1e-6, float(config.get("oracle_action_family_pred_temperature", 0.35)))
    target_temp = max(1e-6, float(config.get("oracle_action_family_target_temperature", 0.025)))
    target_prob = torch.softmax(torch.stack(family_targets) / target_temp, dim=0)
    pred_log_prob = torch.log_softmax(torch.stack(family_preds) / pred_temp, dim=0)
    return -(target_prob * pred_log_prob).sum(), 2


def _oracle_topk_ndcg_loss(preds: torch.Tensor, targets: torch.Tensor, config: dict) -> torch.Tensor:
    """Top-heavy list loss that concentrates supervision on oracle top-k actions."""

    if preds.numel() < 2:
        return torch.zeros((), dtype=preds.dtype, device=preds.device)
    k = min(max(1, int(config.get("oracle_ndcg_k", 8) or 8)), preds.numel())
    target_temp = max(1e-6, float(config.get("oracle_ndcg_target_temperature", 0.5)))
    pred_temp = max(1e-6, float(config.get("oracle_ndcg_pred_temperature", 1.0)))
    top_idx = torch.argsort(targets.detach(), descending=True)[:k]
    top_targets = targets.detach()[top_idx]
    rank_positions = torch.arange(k, dtype=preds.dtype, device=preds.device)
    discounts = 1.0 / torch.log2(rank_positions + 2.0)
    target_prob = torch.softmax(top_targets / target_temp, dim=0) * discounts
    target_prob = target_prob / target_prob.sum().clamp_min(1e-12)
    pred_log_prob = torch.log_softmax(preds / pred_temp, dim=0)
    return -(target_prob * pred_log_prob[top_idx]).sum()


def _oracle_same_type_topk_ndcg_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    action_types: list[str],
    config: dict,
) -> tuple[torch.Tensor, int]:
    """Apply the top-heavy list loss separately to CP0, CP1, and OP nodes.

    Cross-type utility gaps are much easier than choosing the best node within
    a type and can dominate a full candidate-list objective.  Equal weighting
    across non-singleton type lists keeps the loss focused on the planner's
    observed within-type ranking bottleneck.
    """

    if len(action_types) != preds.numel():
        raise ValueError("action_types length must match prediction count")
    canonical_types = [_canonical_oracle_action_type(value) for value in action_types]
    losses: list[torch.Tensor] = []
    for action_type in sorted(set(canonical_types)):
        indices = [index for index, value in enumerate(canonical_types) if value == action_type]
        if len(indices) < 2:
            continue
        index_tensor = torch.tensor(indices, dtype=torch.long, device=preds.device)
        losses.append(
            _oracle_topk_ndcg_loss(
                preds.index_select(0, index_tensor),
                targets.index_select(0, index_tensor),
                config,
            )
        )
    if not losses:
        return torch.zeros((), dtype=preds.dtype, device=preds.device), 0
    return torch.stack(losses).mean(), len(losses)


def _oracle_conservative_loss(preds: torch.Tensor, targets: torch.Tensor, config: dict) -> torch.Tensor:
    """CQL-style penalty that discourages high scores on non-oracle-top actions."""

    if preds.numel() < 2:
        return torch.zeros((), dtype=preds.dtype, device=preds.device)
    pos_k = min(max(1, int(config.get("oracle_conservative_positive_topk", 1) or 1)), preds.numel() - 1)
    order = torch.argsort(targets.detach(), descending=True)
    pos_idx = order[:pos_k]
    neg_idx = order[pos_k:]
    if neg_idx.numel() == 0:
        return torch.zeros((), dtype=preds.dtype, device=preds.device)
    hard_topk = int(config.get("oracle_conservative_hard_negative_topk", 0) or 0)
    if hard_topk > 0 and neg_idx.numel() > hard_topk:
        neg_order = torch.argsort(preds.detach()[neg_idx], descending=True)[:hard_topk]
        neg_idx = neg_idx[neg_order]
    temp = max(1e-6, float(config.get("oracle_conservative_temperature", 1.0)))
    margin = float(config.get("oracle_conservative_margin", 0.0))
    pos_ref = preds[pos_idx].mean()
    neg_scores = preds[neg_idx]
    neg_lse = temp * torch.logsumexp(neg_scores / temp, dim=0)
    if bool(config.get("oracle_conservative_normalize", True)):
        neg_lse = neg_lse - temp * math.log(max(1, int(neg_scores.numel())))
    return F.softplus(neg_lse - pos_ref + margin)


def _standardize_oracle_vector(values: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize one candidate pool to relative advantages."""

    centered = values - values.mean()
    scale = centered.pow(2).mean().sqrt()
    return centered / scale.clamp_min(eps)


def _oracle_context_loss(preds: torch.Tensor, targets: torch.Tensor, config: dict) -> torch.Tensor:
    """Align normalized score shape with normalized oracle delta-TC within a candidate pool."""

    if preds.numel() < 2:
        return torch.zeros((), dtype=preds.dtype, device=preds.device)
    pred_temp = max(1e-6, float(config.get("oracle_context_pred_temperature", 1.0)))
    target_temp = max(1e-6, float(config.get("oracle_context_target_temperature", 1.0)))
    pred_z = _standardize_oracle_vector(preds / pred_temp)
    target_z = _standardize_oracle_vector(targets.detach() / target_temp)
    elements = F.smooth_l1_loss(pred_z, target_z, reduction="none")
    top_weight = max(0.0, min(1.0, float(config.get("oracle_context_top_weight", 0.0))))
    if top_weight <= 0.0:
        return elements.mean()
    weight_temp = max(1e-6, float(config.get("oracle_context_weight_temperature", 0.5)))
    target_prob = torch.softmax(targets.detach() / weight_temp, dim=0)
    uniform = torch.full_like(target_prob, 1.0 / float(target_prob.numel()))
    weights = (1.0 - top_weight) * uniform + top_weight * target_prob
    return (weights * elements).sum()


def _oracle_best_action_type(group: list[dict[str, Any]]) -> str:
    """Return the action type with the largest real delta-TC in one state."""

    if not group:
        raise ValueError("cannot classify an empty oracle group")
    best = max(group, key=lambda row: _safe_float(row.get("oracle_delta_tc")))
    return _canonical_oracle_action_type(best.get("type"))


def _sample_oracle_groups(
    oracle_groups: list[list[dict[str, Any]]],
    group_count: int,
    mode: str = "uniform",
) -> list[list[dict[str, Any]]]:
    """Sample state groups uniformly or balance their real best action type."""

    count = min(len(oracle_groups), max(1, int(group_count)))
    normalized = str(mode or "uniform").strip().lower()
    if normalized in {"uniform", "random"}:
        return random.sample(oracle_groups, k=count)
    if normalized not in {"best_type_balanced", "type_balanced"}:
        raise ValueError(f"unsupported oracle_group_sampling={mode!r}")

    buckets: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for group in oracle_groups:
        buckets[_oracle_best_action_type(group)].append(group)
    action_types = sorted(buckets)
    random.shuffle(action_types)
    shuffled = {key: random.sample(buckets[key], k=len(buckets[key])) for key in action_types}
    offsets = {key: 0 for key in action_types}
    selected: list[list[dict[str, Any]]] = []
    while len(selected) < count:
        advanced = False
        for action_type in action_types:
            offset = offsets[action_type]
            if offset >= len(shuffled[action_type]):
                continue
            selected.append(shuffled[action_type][offset])
            offsets[action_type] = offset + 1
            advanced = True
            if len(selected) == count:
                break
        if not advanced:
            break
    return selected


def train_oracle_ranking_step(
    model: TPIWorldModel,
    oracle_groups: list[list[dict[str, str]]],
    optimizer: torch.optim.Optimizer,
    config: dict,
    device: torch.device,
    graph_cache: dict[str, Any],
    base_cache: dict[str, torch.Tensor],
    epoch: int,
    prefix_latent_cache: dict[tuple[str, tuple[tuple[str, str], ...]], torch.Tensor] | None = None,
) -> dict[str, float]:
    """Run one auxiliary oracle pairwise-ranking optimizer step."""

    rank_weight = _q_rank_weight(config, epoch)
    same_type_rank_weight = _q_same_type_rank_weight(config, epoch)
    aux_rank_weight = _aux_rank_weight(config, epoch)
    aux_same_type_rank_weight = _aux_same_type_rank_weight(config, epoch)
    value_weight = _q_value_weight(config, epoch)
    candidate_weight = _q_candidate_weight(config, epoch)
    action_type_weight = _q_action_type_weight(config, epoch)
    action_family_weight = _q_action_family_weight(config, epoch)
    ndcg_weight = _q_ndcg_weight(config, epoch)
    same_type_ndcg_weight = _q_same_type_ndcg_weight(config, epoch)
    conservative_weight = _q_conservative_weight(config, epoch)
    context_weight = _q_context_weight(config, epoch)
    sa_value_weight = _q_sa_value_weight(config, epoch)
    oracle_weight = max(
        rank_weight,
        same_type_rank_weight,
        aux_rank_weight,
        aux_same_type_rank_weight,
        value_weight,
        candidate_weight,
        action_type_weight,
        action_family_weight,
        ndcg_weight,
        same_type_ndcg_weight,
        conservative_weight,
        context_weight,
        sa_value_weight,
    )
    if oracle_weight <= 0.0 or not oracle_groups:
        return {
            "oracle_loss": 0.0,
            "oracle_rank_loss": 0.0,
            "oracle_same_type_rank_loss": 0.0,
            "oracle_aux_rank_loss": 0.0,
            "oracle_aux_same_type_rank_loss": 0.0,
            "oracle_value_loss": 0.0,
            "oracle_candidate_loss": 0.0,
            "oracle_action_type_loss": 0.0,
            "oracle_action_family_loss": 0.0,
            "oracle_ndcg_loss": 0.0,
            "oracle_same_type_ndcg_loss": 0.0,
            "oracle_conservative_loss": 0.0,
            "oracle_context_loss": 0.0,
            "oracle_sa_value_loss": 0.0,
            "oracle_pairs": 0.0,
            "oracle_same_type_pairs": 0.0,
            "oracle_aux_pairs": 0.0,
            "oracle_aux_same_type_pairs": 0.0,
            "oracle_action_types": 0.0,
            "oracle_action_families": 0.0,
            "oracle_groups": 0.0,
            "oracle_weight": oracle_weight,
        }
    group_count = max(1, int(config.get("oracle_batch_groups", 1) or 1))
    groups = _sample_oracle_groups(
        oracle_groups,
        group_count,
        str(config.get("oracle_group_sampling", "uniform")),
    )
    score_field = str(config.get("oracle_ranking_score_field", "q_pred"))
    aux_score_field = str(config.get("oracle_aux_ranking_score_field", "typed_return_pred"))
    coverage_scale = float(config.get("coverage_scale", 100.0))
    min_delta = float(config.get("oracle_pairwise_min_delta", 0.001)) * coverage_scale
    temperature = float(config.get("oracle_pairwise_temperature", 1.0))
    pairwise_mode = str(config.get("oracle_pairwise_mode", "all"))
    hard_negative_topk = int(config.get("oracle_hard_negative_topk", 0) or 0)
    positive_topk = int(config.get("oracle_positive_topk", 1) or 1)
    max_pairs_per_group = int(config.get("oracle_max_pairs_per_group", 0) or 0)

    rank_losses: list[torch.Tensor] = []
    same_type_rank_losses: list[torch.Tensor] = []
    aux_rank_losses: list[torch.Tensor] = []
    aux_same_type_rank_losses: list[torch.Tensor] = []
    value_losses: list[torch.Tensor] = []
    candidate_losses: list[torch.Tensor] = []
    action_type_losses: list[torch.Tensor] = []
    action_family_losses: list[torch.Tensor] = []
    ndcg_losses: list[torch.Tensor] = []
    same_type_ndcg_losses: list[torch.Tensor] = []
    conservative_losses: list[torch.Tensor] = []
    context_losses: list[torch.Tensor] = []
    sa_value_losses: list[torch.Tensor] = []
    pair_total = 0
    same_type_pair_total = 0
    aux_pair_total = 0
    aux_same_type_pair_total = 0
    action_type_total = 0
    action_family_total = 0
    for group in groups:
        if prefix_latent_cache is None:
            scores = _predict_oracle_group_scores(
                model,
                config,
                group,
                graph_cache,
                base_cache,
                device,
            )
        else:
            scores = _predict_oracle_group_scores(
                model,
                config,
                group,
                graph_cache,
                base_cache,
                device,
                prefix_latent_cache,
            )
        preds = _oracle_score(scores, score_field)
        aux_preds = _oracle_score(scores, aux_score_field)
        targets = torch.tensor(
            [coverage_scale * _safe_float(row.get("oracle_delta_tc")) for row in group],
            dtype=preds.dtype,
            device=device,
        )
        if rank_weight > 0.0:
            rank_loss, pair_count = _oracle_pairwise_rank_loss(
                preds,
                targets,
                min_delta,
                temperature,
                mode=pairwise_mode,
                hard_negative_topk=hard_negative_topk,
                positive_topk=positive_topk,
                max_pairs=max_pairs_per_group,
            )
            if pair_count > 0:
                rank_losses.append(rank_loss)
                pair_total += int(pair_count)
        if same_type_rank_weight > 0.0:
            same_type_loss, same_type_pair_count = _oracle_same_type_rank_loss(
                preds,
                targets,
                [str(row.get("type") or "") for row in group],
                min_delta,
                temperature,
                mode=pairwise_mode,
                hard_negative_topk=hard_negative_topk,
                positive_topk=positive_topk,
                max_pairs=max_pairs_per_group,
            )
            if same_type_pair_count > 0:
                same_type_rank_losses.append(same_type_loss)
                same_type_pair_total += int(same_type_pair_count)
        if aux_rank_weight > 0.0:
            aux_rank_loss, aux_pair_count = _oracle_pairwise_rank_loss(
                aux_preds,
                targets,
                min_delta,
                temperature,
                mode=pairwise_mode,
                hard_negative_topk=hard_negative_topk,
                positive_topk=positive_topk,
                max_pairs=max_pairs_per_group,
            )
            if aux_pair_count > 0:
                aux_rank_losses.append(aux_rank_loss)
                aux_pair_total += int(aux_pair_count)
        if aux_same_type_rank_weight > 0.0:
            aux_same_type_loss, aux_same_type_pair_count = _oracle_same_type_rank_loss(
                aux_preds,
                targets,
                [str(row.get("type") or "") for row in group],
                min_delta,
                temperature,
                mode=pairwise_mode,
                hard_negative_topk=hard_negative_topk,
                positive_topk=positive_topk,
                max_pairs=max_pairs_per_group,
            )
            if aux_same_type_pair_count > 0:
                aux_same_type_rank_losses.append(aux_same_type_loss)
                aux_same_type_pair_total += int(aux_same_type_pair_count)
        if value_weight > 0.0:
            value_losses.append(F.smooth_l1_loss(preds, targets))
        if candidate_weight > 0.0 and preds.numel() >= 2:
            candidate_losses.append(_oracle_candidate_loss(preds, targets, config))
        if action_type_weight > 0.0 and preds.numel() >= 2:
            action_type_loss, action_type_count = _oracle_action_type_loss(
                preds,
                targets,
                [str(row.get("type") or "") for row in group],
                config,
            )
            if action_type_count > 0:
                action_type_losses.append(action_type_loss)
                action_type_total += int(action_type_count)
        if action_family_weight > 0.0 and preds.numel() >= 2:
            action_family_loss, action_family_count = _oracle_action_family_loss(
                preds,
                targets,
                [str(row.get("type") or "") for row in group],
                config,
            )
            if action_family_count > 0:
                action_family_losses.append(action_family_loss)
                action_family_total += int(action_family_count)
        if ndcg_weight > 0.0 and preds.numel() >= 2:
            ndcg_losses.append(_oracle_topk_ndcg_loss(preds, targets, config))
        if same_type_ndcg_weight > 0.0 and preds.numel() >= 2:
            same_type_ndcg_loss, type_list_count = _oracle_same_type_topk_ndcg_loss(
                preds,
                targets,
                [str(row.get("type") or "") for row in group],
                config,
            )
            if type_list_count > 0:
                same_type_ndcg_losses.append(same_type_ndcg_loss)
        if conservative_weight > 0.0 and preds.numel() >= 2:
            conservative_losses.append(_oracle_conservative_loss(preds, targets, config))
        if context_weight > 0.0 and preds.numel() >= 2:
            context_losses.append(_oracle_context_loss(preds, targets, config))
        if sa_value_weight > 0.0:
            sa_preds = torch.stack(
                [scores["typed_sa0_reduction_pred"], scores["typed_sa1_reduction_pred"]],
                dim=1,
            )
            sa_targets, sa_mask = _oracle_sa_reduction_targets(
                group,
                dtype=sa_preds.dtype,
                device=device,
            )
            if bool(sa_mask.any().item()):
                sa_elements = F.smooth_l1_loss(sa_preds, sa_targets, reduction="none")
                sa_value_losses.append(sa_elements[sa_mask].mean())

    if (
        not rank_losses
        and not same_type_rank_losses
        and not aux_rank_losses
        and not aux_same_type_rank_losses
        and not value_losses
        and not candidate_losses
        and not action_type_losses
        and not action_family_losses
        and not ndcg_losses
        and not same_type_ndcg_losses
        and not conservative_losses
        and not context_losses
        and not sa_value_losses
    ):
        return {
            "oracle_loss": 0.0,
            "oracle_rank_loss": 0.0,
            "oracle_same_type_rank_loss": 0.0,
            "oracle_aux_rank_loss": 0.0,
            "oracle_aux_same_type_rank_loss": 0.0,
            "oracle_value_loss": 0.0,
            "oracle_candidate_loss": 0.0,
            "oracle_action_type_loss": 0.0,
            "oracle_action_family_loss": 0.0,
            "oracle_ndcg_loss": 0.0,
            "oracle_same_type_ndcg_loss": 0.0,
            "oracle_conservative_loss": 0.0,
            "oracle_context_loss": 0.0,
            "oracle_sa_value_loss": 0.0,
            "oracle_pairs": 0.0,
            "oracle_same_type_pairs": 0.0,
            "oracle_aux_pairs": 0.0,
            "oracle_aux_same_type_pairs": 0.0,
            "oracle_action_types": 0.0,
            "oracle_action_families": 0.0,
            "oracle_groups": float(len(groups)),
            "oracle_weight": oracle_weight,
        }
    zero = torch.zeros((), dtype=torch.float32, device=device)
    rank_loss = torch.stack(rank_losses).mean() if rank_losses else zero
    same_type_rank_loss = torch.stack(same_type_rank_losses).mean() if same_type_rank_losses else zero
    aux_rank_loss = torch.stack(aux_rank_losses).mean() if aux_rank_losses else zero
    aux_same_type_rank_loss = (
        torch.stack(aux_same_type_rank_losses).mean() if aux_same_type_rank_losses else zero
    )
    value_loss = torch.stack(value_losses).mean() if value_losses else zero
    candidate_loss = torch.stack(candidate_losses).mean() if candidate_losses else zero
    action_type_loss = torch.stack(action_type_losses).mean() if action_type_losses else zero
    action_family_loss = (
        torch.stack(action_family_losses).mean() if action_family_losses else zero
    )
    ndcg_loss = torch.stack(ndcg_losses).mean() if ndcg_losses else zero
    same_type_ndcg_loss = (
        torch.stack(same_type_ndcg_losses).mean() if same_type_ndcg_losses else zero
    )
    conservative_loss = torch.stack(conservative_losses).mean() if conservative_losses else zero
    context_loss = torch.stack(context_losses).mean() if context_losses else zero
    sa_value_loss = torch.stack(sa_value_losses).mean() if sa_value_losses else zero
    loss = (
        rank_weight * rank_loss
        + same_type_rank_weight * same_type_rank_loss
        + aux_rank_weight * aux_rank_loss
        + aux_same_type_rank_weight * aux_same_type_rank_loss
        + value_weight * value_loss
        + candidate_weight * candidate_loss
        + action_type_weight * action_type_loss
        + action_family_weight * action_family_loss
        + ndcg_weight * ndcg_loss
        + same_type_ndcg_weight * same_type_ndcg_loss
        + conservative_weight * conservative_loss
        + context_weight * context_loss
        + sa_value_weight * sa_value_loss
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    update_ema_if_encoder_trainable(model, float(config["ema_decay"]))
    return {
        "oracle_loss": float(loss.detach().cpu().item()),
        "oracle_rank_loss": float(rank_loss.detach().cpu().item()),
        "oracle_same_type_rank_loss": float(same_type_rank_loss.detach().cpu().item()),
        "oracle_aux_rank_loss": float(aux_rank_loss.detach().cpu().item()),
        "oracle_aux_same_type_rank_loss": float(aux_same_type_rank_loss.detach().cpu().item()),
        "oracle_value_loss": float(value_loss.detach().cpu().item()),
        "oracle_candidate_loss": float(candidate_loss.detach().cpu().item()),
        "oracle_action_type_loss": float(action_type_loss.detach().cpu().item()),
        "oracle_action_family_loss": float(action_family_loss.detach().cpu().item()),
        "oracle_ndcg_loss": float(ndcg_loss.detach().cpu().item()),
        "oracle_same_type_ndcg_loss": float(same_type_ndcg_loss.detach().cpu().item()),
        "oracle_conservative_loss": float(conservative_loss.detach().cpu().item()),
        "oracle_context_loss": float(context_loss.detach().cpu().item()),
        "oracle_sa_value_loss": float(sa_value_loss.detach().cpu().item()),
        "oracle_pairs": float(pair_total),
        "oracle_same_type_pairs": float(same_type_pair_total),
        "oracle_aux_pairs": float(aux_pair_total),
        "oracle_aux_same_type_pairs": float(aux_same_type_pair_total),
        "oracle_action_types": float(action_type_total),
        "oracle_action_families": float(action_family_total),
        "oracle_groups": float(
            max(
                len(rank_losses),
                len(same_type_rank_losses),
                len(aux_rank_losses),
                len(aux_same_type_rank_losses),
                len(value_losses),
                len(candidate_losses),
                len(action_type_losses),
                len(action_family_losses),
                len(ndcg_losses),
                len(same_type_ndcg_losses),
                len(conservative_losses),
                len(context_losses),
                len(sa_value_losses),
            )
        ),
        "oracle_weight": oracle_weight,
    }


def train_one_epoch(
    model: TPIWorldModel,
    dataset: TPIDataset,
    optimizer: torch.optim.Optimizer,
    config: dict,
    device: torch.device,
    pattern_enabled: bool,
    max_steps: int | None = None,
    horizon: int = 1,
    epoch: int = 1,
    oracle_groups: list[list[dict[str, str]]] | None = None,
    oracle_graph_cache: dict[str, Any] | None = None,
    oracle_base_cache: dict[str, torch.Tensor] | None = None,
    oracle_prefix_latent_cache: (
        dict[tuple[str, tuple[tuple[str, str], ...]], torch.Tensor] | None
    ) = None,
) -> dict:
    """Train for one shuffled pass or until `max_steps` is reached."""

    model.train()
    indices = _training_indices(dataset, config, max_steps)
    totals: dict[str, float] = {}
    steps = 0
    oracle_steps = 0
    oracle_every = max(1, int(config.get("oracle_every_n_steps", 4) or 4))
    oracle_graph_cache = oracle_graph_cache if oracle_graph_cache is not None else {}
    oracle_base_cache = oracle_base_cache if oracle_base_cache is not None else {}
    progress_enabled = bool(config.get("progress_bar", False))
    progress_every = max(1, int(config.get("progress_log_every", 25) or 25))
    total_steps = len(indices)
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
        update_ema_if_encoder_trainable(model, float(config["ema_decay"]))
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        steps += 1
        if oracle_groups and steps % oracle_every == 0:
            oracle_metrics = train_oracle_ranking_step(
                model,
                oracle_groups,
                optimizer,
                config,
                device,
                oracle_graph_cache,
                oracle_base_cache,
                epoch,
                oracle_prefix_latent_cache,
            )
            oracle_steps += 1
            for key, value in oracle_metrics.items():
                totals[key] = totals.get(key, 0.0) + value
        if progress_enabled and (steps == 1 or steps % progress_every == 0 or steps == total_steps):
            print(_train_progress_line(epoch, steps, total_steps, totals, oracle_steps), flush=True)
        if max_steps is not None and steps >= max_steps:
            break
    averaged = {}
    for key, value in totals.items():
        denom = max(1, oracle_steps) if key.startswith("oracle_") else max(1, steps)
        averaged[key] = value / denom
    return averaged | {"steps": float(steps), "oracle_steps": float(oracle_steps)}


def _avg_metric(totals: dict[str, float], key: str, steps: int, oracle_steps: int) -> float:
    denom = max(1, oracle_steps) if key.startswith("oracle_") else max(1, steps)
    return totals.get(key, 0.0) / denom


def _train_progress_line(
    epoch: int,
    step: int,
    total_steps: int,
    totals: dict[str, float],
    oracle_steps: int,
) -> str:
    width = 24
    done = int(width * step / max(1, total_steps))
    bar = "#" * done + "-" * (width - done)
    pct = 100.0 * step / max(1, total_steps)
    parts = [
        f"[train-progress] epoch={epoch}",
        f"[{bar}]",
        f"{step}/{total_steps}",
        f"{pct:5.1f}%",
        f"loss={_avg_metric(totals, 'loss', step, oracle_steps):.5f}",
        f"jepa={_avg_metric(totals, 'jepa_loss', step, oracle_steps):.5f}",
        f"hard={_avg_metric(totals, 'hard_bce_loss', step, oracle_steps):.5f}",
        f"hard_red={_avg_metric(totals, 'hard_reduction_loss', step, oracle_steps):.5f}",
    ]
    if "typed_marginal_loss" in totals:
        parts.extend(
            [
                f"typed_m={_avg_metric(totals, 'typed_marginal_loss', step, oracle_steps):.5f}",
                f"typed_ret={_avg_metric(totals, 'typed_return_loss', step, oracle_steps):.5f}",
                f"typed_sa={_avg_metric(totals, 'typed_sa_reduction_loss', step, oracle_steps):.5f}",
            ]
        )
    if oracle_steps > 0:
        parts.extend(
            [
                f"oracle={_avg_metric(totals, 'oracle_loss', step, oracle_steps):.5f}",
                f"rank={_avg_metric(totals, 'oracle_rank_loss', step, oracle_steps):.5f}",
                f"same={_avg_metric(totals, 'oracle_same_type_rank_loss', step, oracle_steps):.5f}",
                f"aux_rank={_avg_metric(totals, 'oracle_aux_rank_loss', step, oracle_steps):.5f}",
                f"aux_same={_avg_metric(totals, 'oracle_aux_same_type_rank_loss', step, oracle_steps):.5f}",
                f"value={_avg_metric(totals, 'oracle_value_loss', step, oracle_steps):.5f}",
                f"cand={_avg_metric(totals, 'oracle_candidate_loss', step, oracle_steps):.5f}",
                f"type={_avg_metric(totals, 'oracle_action_type_loss', step, oracle_steps):.5f}",
                f"family={_avg_metric(totals, 'oracle_action_family_loss', step, oracle_steps):.5f}",
                f"ndcg={_avg_metric(totals, 'oracle_ndcg_loss', step, oracle_steps):.5f}",
                f"same_ndcg={_avg_metric(totals, 'oracle_same_type_ndcg_loss', step, oracle_steps):.5f}",
                f"cql={_avg_metric(totals, 'oracle_conservative_loss', step, oracle_steps):.5f}",
                f"ctx={_avg_metric(totals, 'oracle_context_loss', step, oracle_steps):.5f}",
                f"pairs={_avg_metric(totals, 'oracle_pairs', step, oracle_steps):.1f}",
            ]
        )
    return " ".join(parts)


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
    config.setdefault("lambda_typed_marginal", 0.0)
    config.setdefault("lambda_typed_return", 0.0)
    config.setdefault("lambda_typed_sa_reduction", 0.0)
    config.setdefault("utility_head_type", "legacy")
    config.setdefault("hard_pos_weight_max", 20.0)
    config.setdefault("hard_loss", "bce")
    config.setdefault("lambda_hard_rank", 0.0)
    config.setdefault("lambda_hard_brier", 0.0)
    config.setdefault("lambda_hard_soft_f1", 0.0)
    config.setdefault("lambda_oracle_rank", 0.0)
    config.setdefault("lambda_oracle_value", 0.0)
    config.setdefault("lambda_q_value", config.get("lambda_oracle_value", 0.0))
    config.setdefault("lambda_q_rank", config.get("lambda_oracle_rank", 0.0))
    config.setdefault("lambda_same_type_rank", 0.0)
    config.setdefault("lambda_aux_rank", 0.0)
    config.setdefault("lambda_aux_same_type_rank", 0.0)
    config.setdefault("lambda_candidate", 0.0)
    config.setdefault("lambda_action_type_rank", 0.0)
    config.setdefault("oracle_action_type_aggregate_temperature", 0.10)
    config.setdefault("oracle_action_type_target_temperature", 0.025)
    config.setdefault("oracle_action_type_pred_temperature", 0.35)
    config.setdefault("lambda_action_family_rank", 0.0)
    config.setdefault("oracle_action_family_aggregate_temperature", 0.10)
    config.setdefault("oracle_action_family_target_temperature", 0.025)
    config.setdefault("oracle_action_family_pred_temperature", 0.35)
    config.setdefault("lambda_ndcg_rank", 0.0)
    config.setdefault("lambda_same_type_ndcg_rank", 0.0)
    config.setdefault("oracle_ndcg_k", 8)
    config.setdefault("oracle_ndcg_target_temperature", 0.5)
    config.setdefault("oracle_ndcg_pred_temperature", 1.0)
    config.setdefault("lambda_conservative_q", 0.0)
    config.setdefault("oracle_conservative_positive_topk", 1)
    config.setdefault("oracle_conservative_hard_negative_topk", 0)
    config.setdefault("oracle_conservative_temperature", 1.0)
    config.setdefault("oracle_conservative_margin", 0.0)
    config.setdefault("oracle_conservative_normalize", True)
    config.setdefault("lambda_context_rank", 0.0)
    config.setdefault("oracle_context_pred_temperature", 1.0)
    config.setdefault("oracle_context_target_temperature", 1.0)
    config.setdefault("oracle_context_top_weight", 0.0)
    config.setdefault("oracle_context_weight_temperature", 0.5)
    config.setdefault("oracle_ranking_score_field", "q_pred")
    config.setdefault("oracle_aux_ranking_score_field", "typed_return_pred")
    config.setdefault("candidate_target_temperature", 1.0)
    config.setdefault("candidate_pred_temperature", 1.0)
    config.setdefault("oracle_every_n_steps", 4)
    config.setdefault("oracle_batch_groups", 4)
    config.setdefault("oracle_group_sampling", "uniform")
    config.setdefault("oracle_warmup_epochs", 1)
    config.setdefault("oracle_ramp_epochs", 2)
    config.setdefault("oracle_pairwise_min_delta", 0.001)
    config.setdefault("oracle_pairwise_temperature", 1.0)
    config.setdefault("oracle_pairwise_mode", "all")
    config.setdefault("oracle_hard_negative_topk", 0)
    config.setdefault("oracle_positive_topk", 1)
    config.setdefault("oracle_max_pairs_per_group", 0)
    config.setdefault("oracle_prefix_detach", True)
    config.setdefault("hard_soft_f1_eps", 1e-6)
    config.setdefault("hard_rank_margin", 0.2)
    config.setdefault("hard_negative_sample_ratio", 0)
    config.setdefault("hard_negative_mining", "random")
    config.setdefault("hard_head_type", "mlp")
    config.setdefault("encoder_type", "mean")
    config.setdefault("summary_mode", "global")
    config.setdefault("train_sample_strategy", "shuffle")
    config.setdefault("progress_bar", False)
    config.setdefault("progress_log_every", 25)
    pattern_enabled = _pattern_target_valid(train_rows)
    if not pattern_enabled or float(config.get("lambda_pattern", 0.0)) <= 0.0:
        config["lambda_pattern"] = 0.0
        print("[train] disabling pattern loss")
    if float(config.get("lambda_fc", 0.0)) > 0.0:
        print(
            "[train] note: delta_test_coverage is retained as a weak objective; "
            "the main hard-fault target is hard_reduction"
        )
    oracle_groups = None
    oracle_enabled = bool(
        config.get("oracle_actions")
        and max(
            float(config.get("lambda_q_rank", config.get("lambda_oracle_rank", 0.0))),
            float(config.get("lambda_same_type_rank", 0.0)),
            float(config.get("lambda_aux_rank", 0.0)),
            float(config.get("lambda_aux_same_type_rank", 0.0)),
            float(config.get("lambda_q_value", config.get("lambda_oracle_value", 0.0))),
            float(config.get("lambda_candidate", 0.0)),
            float(config.get("lambda_action_type_rank", 0.0)),
            float(config.get("lambda_action_family_rank", 0.0)),
            float(config.get("lambda_ndcg_rank", 0.0)),
            float(config.get("lambda_same_type_ndcg_rank", 0.0)),
            float(config.get("lambda_conservative_q", 0.0)),
            float(config.get("lambda_context_rank", 0.0)),
            float(config.get("lambda_oracle_sa_value", 0.0)),
        )
        > 0.0
    )
    if oracle_enabled:
        oracle_forbidden = set(excluded)
        oracle_forbidden.update(parse_benchmark_list(config.get("oracle_forbidden_benchmarks")))
        oracle_groups = load_oracle_groups(
            config["oracle_actions"],
            max_actions_per_group=int(config.get("oracle_max_actions_per_group", 0)) or None,
            forbidden_benchmarks=oracle_forbidden,
        )
        print(
            f"[train] q_oracle enabled groups={len(oracle_groups)} "
            f"score_field={config['oracle_ranking_score_field']} "
            f"aux_score_field={config['oracle_aux_ranking_score_field']} "
            f"lambda_q_value={float(config['lambda_q_value'])} "
            f"lambda_q_rank={float(config['lambda_q_rank'])} "
            f"lambda_same_type_rank={float(config['lambda_same_type_rank'])} "
            f"lambda_aux_rank={float(config['lambda_aux_rank'])} "
            f"lambda_aux_same_type_rank={float(config['lambda_aux_same_type_rank'])} "
            f"lambda_candidate={float(config['lambda_candidate'])} "
            f"lambda_action_type_rank={float(config['lambda_action_type_rank'])} "
            f"lambda_action_family_rank={float(config['lambda_action_family_rank'])} "
            f"lambda_ndcg_rank={float(config['lambda_ndcg_rank'])} "
            f"lambda_same_type_ndcg_rank={float(config['lambda_same_type_ndcg_rank'])} "
            f"lambda_conservative_q={float(config['lambda_conservative_q'])} "
            f"lambda_context_rank={float(config['lambda_context_rank'])} "
            f"lambda_oracle_sa_value={float(config['lambda_oracle_sa_value'])} "
            f"group_sampling={config['oracle_group_sampling']} "
            f"pairwise_mode={config['oracle_pairwise_mode']} "
            f"forbidden_benchmarks={len(oracle_forbidden)}"
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
        q_head_type=str(config.get("q_head_type", "summary")),
        utility_head_type=str(config.get("utility_head_type", "legacy")),
    ).to(device)
    if config.get("init_checkpoint"):
        initialize_from_checkpoint(
            model,
            config["init_checkpoint"],
            device,
            feature_dim=feature_dim,
            relation_dim=relation_dim,
            strict=bool(config.get("init_checkpoint_strict", True)),
        )
        print(
            f"[train] initialized checkpoint={config['init_checkpoint']} "
            f"strict={bool(config.get('init_checkpoint_strict', True))}"
        )
    trainable_parameters = configure_trainable_parameters(
        model,
        str(config.get("trainable_modules", "all")),
    )
    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    total_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"[train] trainable_modules={config.get('trainable_modules', 'all')} "
        f"trainable_parameters={trainable_count}/{total_count}"
    )
    optimizer = torch.optim.AdamW(trainable_parameters, lr=float(config["lr"]))
    oracle_graph_cache: dict[str, Any] = {}
    oracle_base_cache: dict[str, torch.Tensor] = {}
    oracle_prefix_latent_cache = None
    if bool(config.get("oracle_cache_prefix_latents", False)):
        prefix_modules = (model.online_encoder, model.action_encoder, model.dynamics)
        if any(parameter.requires_grad for module in prefix_modules for parameter in module.parameters()):
            raise ValueError(
                "oracle_cache_prefix_latents requires frozen online_encoder, action_encoder, and dynamics"
            )
        if float(config.get("dropout", 0.0)) != 0.0:
            raise ValueError("oracle_cache_prefix_latents requires dropout=0 for exact replay caching")
        oracle_prefix_latent_cache = {}
        print("[train] oracle prefix-latent cache enabled", flush=True)

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
            epoch=epoch,
            oracle_groups=oracle_groups,
            oracle_graph_cache=oracle_graph_cache,
            oracle_base_cache=oracle_base_cache,
            oracle_prefix_latent_cache=oracle_prefix_latent_cache,
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
