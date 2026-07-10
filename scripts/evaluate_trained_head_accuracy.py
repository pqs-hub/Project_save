"""Evaluate accuracy-style metrics for trained TPI-JEPA heads.

This intentionally avoids loss and MAE metrics. Continuous heads are converted
to sign/direction classification where that is meaningful.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tpi_jepa.dataset import TPIDataset, split_by_benchmark
from tpi_jepa.features import SCOAP_START, SCOAP_END
from tpi_jepa.labels import load_labels
from tpi_jepa.plan import load_checkpoint


def sign(value: float, eps: float = 1e-9) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def safe_div(num: float, den: float) -> float:
    return num / den if den else float("nan")


def add_sign(stats: dict[str, dict[str, int]], name: str, pred: float, target: float) -> None:
    pred_sign = sign(float(pred))
    target_sign = sign(float(target))
    row = stats[name]
    row["n"] += 1
    row["correct"] += int(pred_sign == target_sign)
    row["pred_pos"] += int(pred_sign > 0)
    row["target_pos"] += int(target_sign > 0)


def add_binary(conf: dict[str, dict[str, int]], name: str, pred: torch.Tensor, target: torch.Tensor) -> None:
    pred = pred.to(torch.int64).cpu()
    target = target.to(torch.int64).cpu()
    row = conf[name]
    row["tp"] += int(((pred == 1) & (target == 1)).sum().item())
    row["tn"] += int(((pred == 0) & (target == 0)).sum().item())
    row["fp"] += int(((pred == 1) & (target == 0)).sum().item())
    row["fn"] += int(((pred == 0) & (target == 1)).sum().item())


def binary_metric_row(task: str, counts: dict[str, int]) -> dict[str, Any]:
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]
    n = tp + tn + fp + fn
    tpr = safe_div(tp, tp + fn)
    tnr = safe_div(tn, tn + fp)
    precision = safe_div(tp, tp + fp)
    recall = tpr
    f1 = (
        safe_div(2.0 * precision * recall, precision + recall)
        if math.isfinite(precision) and math.isfinite(recall) and precision + recall > 0.0
        else float("nan")
    )
    return {
        "metric_type": "binary",
        "task": task,
        "n": n,
        "accuracy": safe_div(tp + tn, n),
        "balanced_accuracy": (tpr + tnr) / 2.0 if math.isfinite(tpr) and math.isfinite(tnr) else float("nan"),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "positive_rate": safe_div(tp + fn, n),
    }


def sign_metric_row(task: str, counts: dict[str, int]) -> dict[str, Any]:
    n = counts["n"]
    return {
        "metric_type": "sign",
        "task": task,
        "n": n,
        "accuracy": safe_div(counts["correct"], n),
        "balanced_accuracy": "",
        "precision": "",
        "recall": "",
        "f1": "",
        "positive_rate": safe_div(counts["target_pos"], n),
        "pred_positive_rate": safe_div(counts["pred_pos"], n),
    }


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "metric_type",
        "task",
        "n",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "positive_rate",
        "pred_positive_rate",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None, help="Training config JSON. Defaults to checkpoint config.")
    parser.add_argument("--max-samples", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--include-delta-scoap", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA was requested but is not visible to PyTorch")
    model, checkpoint_config = load_checkpoint(args.checkpoint, device)
    config = json.loads(Path(args.config).read_text()) if args.config else checkpoint_config
    model.eval()

    rows = load_labels(config["labels"])
    _, val_rows, _ = split_by_benchmark(
        rows,
        int(config.get("seed", 1334)),
        train_frac=float(config.get("train_frac", 0.70)),
        val_frac=float(config.get("val_frac", 0.15)),
    )
    dataset = TPIDataset(
        val_rows,
        max_specs=int(args.max_samples),
        max_nodes=int(config.get("max_nodes", 0)) or None,
        feature_mode=str(config.get("feature_mode", "basic")),
        relation_mode=str(config.get("relation_mode", "basic")),
        relation_depth=int(config.get("relation_depth", 8)),
        state_update_mode=str(config.get("state_update_mode", "static")),
        state_update_depth=int(config.get("state_update_depth", 8)),
        real_fault_prior_path=config.get("real_fault_priors") or config.get("real_fault_prior_path"),
        activation_prior_path=config.get("activation_priors") or config.get("activation_prior_path"),
    )

    coverage_scale = float(checkpoint_config.get("coverage_scale", getattr(model, "coverage_scale", 100.0)))
    sign_stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    conf: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0})
    top_hits: dict[int, float] = defaultdict(float)
    top_total: dict[int, float] = defaultdict(float)

    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            out = model(
                sample.graph,
                sample.x_pre.to(device),
                sample.x_post.to(device),
                sample.action_node_id,
                sample.action_type_id,
                sample.relation_features.to(device),
            )
            reward_target = float((coverage_scale * sample.delta_fault_coverage).item())
            add_sign(sign_stats, "reward_pred_sign", float(out["reward_pred"].detach().cpu().item()), reward_target)
            add_sign(sign_stats, "return_pred_sign", float(out["return_pred"].detach().cpu().item()), reward_target)

            hard_pred = out["hard_reduction_pred"].detach().cpu().view(-1)
            hard_target = sample.hard_reduction_target.detach().cpu().view(-1)
            for dim, name in enumerate(["total", "sa0", "sa1"]):
                add_sign(sign_stats, f"hard_reduction_{name}_sign", float(hard_pred[dim]), float(hard_target[dim]))

            hard_prob = out["hard_logits"].detach().cpu().sigmoid()
            hard_label = (sample.hard_targets_post.detach().cpu() >= 0.5).to(torch.int64)
            hard_binary = (hard_prob >= 0.5).to(torch.int64)
            add_binary(conf, "hard_node_sa0", hard_binary[:, 0], hard_label[:, 0])
            add_binary(conf, "hard_node_sa1", hard_binary[:, 1], hard_label[:, 1])
            hard_any_pred = (hard_binary.sum(dim=1) > 0).to(torch.int64)
            hard_any_label = (hard_label.sum(dim=1) > 0).to(torch.int64)
            add_binary(conf, "hard_node_any", hard_any_pred, hard_any_label)

            true_hard = torch.nonzero(hard_any_label, as_tuple=False).flatten()
            if true_hard.numel() > 0:
                scores = hard_prob.max(dim=1).values
                true_set = set(true_hard.tolist())
                for k in [10, 20, 50, 100]:
                    top = set(torch.topk(scores, min(k, scores.numel())).indices.tolist())
                    top_hits[k] += len(top & true_set) / max(1, len(true_set))
                    top_total[k] += 1.0

            if args.include_delta_scoap:
                delta_pred = out["delta_scoap_pred"].detach().cpu()
                delta_target = (
                    sample.x_post[:, SCOAP_START:SCOAP_END] - sample.x_pre[:, SCOAP_START:SCOAP_END]
                ).detach().cpu()
                for dim, name in enumerate(["cc0", "cc1", "co"]):
                    for pred_value, target_value in zip(delta_pred[:, dim].tolist(), delta_target[:, dim].tolist()):
                        add_sign(sign_stats, f"delta_scoap_{name}_sign", pred_value, target_value)

    metric_rows = [sign_metric_row(task, sign_stats[task]) for task in sorted(sign_stats)]
    metric_rows.extend(binary_metric_row(task, conf[task]) for task in sorted(conf))
    for k in sorted(top_total):
        metric_rows.append(
            {
                "metric_type": "topk_recall",
                "task": f"hard_node_top{k}_recall",
                "n": int(top_total[k]),
                "accuracy": safe_div(top_hits[k], top_total[k]),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(args.out_dir / "trained_head_accuracy.tsv", metric_rows)
    summary = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config or "<checkpoint>"),
        "device": str(device),
        "samples": len(dataset),
        "metrics": metric_rows,
    }
    (args.out_dir / "trained_head_accuracy.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
