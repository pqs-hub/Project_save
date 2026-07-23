#!/usr/bin/env python3
"""Compare clipped and unclipped latent replay on one oracle prefix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tpi_jepa.features import (  # noqa: E402
    action_type_to_id,
    make_action_relation_features,
    make_state_features,
)
from tpi_jepa.plan import _clip_latent_norms, load_checkpoint  # noqa: E402
from tpi_jepa.train import (  # noqa: E402
    _oracle_graph_and_base_for,
    _oracle_prefix_actions,
    load_oracle_groups,
)


def norm_summary(z: torch.Tensor) -> dict[str, float]:
    norms = z.norm(dim=1)
    return {
        "median": float(norms.median().item()),
        "mean": float(norms.mean().item()),
        "max": float(norms.max().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-actions", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--clip-ratio", type=float, default=4.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, config = load_checkpoint(args.checkpoint, device)
    model.eval()
    groups = [
        group
        for group in load_oracle_groups(args.oracle_actions)
        if group[0]["benchmark_id"] == args.benchmark_id
    ]
    if not groups:
        raise ValueError(f"no groups for benchmark {args.benchmark_id}")
    group = max(groups, key=lambda rows: len(_oracle_prefix_actions(rows)))
    prefix = _oracle_prefix_actions(group)
    graph_cache = {}
    base_cache = {}
    graph, base_features = _oracle_graph_and_base_for(
        args.benchmark_id,
        graph_cache,
        base_cache,
        config,
    )
    node_ids = {name: index for index, name in enumerate(graph.node_names)}
    x_state = make_state_features(graph, [], base_features).to(device)
    with torch.no_grad():
        initial = model.online_encoder(
            x_state,
            graph.edge_src.to(device),
            graph.edge_dst.to(device),
            graph.gate_type_ids.to(device),
        )
        limit = float(initial.norm(dim=1).median().item()) * args.clip_ratio
        unclipped = initial
        clipped = initial
        clipped_node_events = 0
        checkpoints = {0, 32, 64, 128, 192, 256, 320, 448, 576, 704, len(prefix)}
        trajectory = []
        for step, (node, action_type) in enumerate(prefix, start=1):
            relation = make_action_relation_features(
                graph,
                node_ids[node],
                str(config.get("relation_mode", "basic")),
                int(config.get("relation_depth", 8)),
            ).to(device)
            unclipped = model.predict_from_latent(
                unclipped,
                node_ids[node],
                action_type_to_id(action_type),
                relation,
                include_aux_heads=False,
                sequence_step=step - 1,
            )["z_pred"]
            clipped_raw = model.predict_from_latent(
                clipped,
                node_ids[node],
                action_type_to_id(action_type),
                relation,
                include_aux_heads=False,
                sequence_step=step - 1,
            )["z_pred"]
            clipped_node_events += int((clipped_raw.norm(dim=1) > limit).sum().item())
            clipped = _clip_latent_norms(clipped_raw, limit)
            if step in checkpoints:
                trajectory.append(
                    {
                        "step": step,
                        "unclipped": norm_summary(unclipped),
                        "clipped": norm_summary(clipped),
                    }
                )

    payload = {
        "benchmark_id": args.benchmark_id,
        "state_id": group[0]["state_id"],
        "prefix_steps": len(prefix),
        "clip_ratio": args.clip_ratio,
        "initial": norm_summary(initial),
        "norm_limit": limit,
        "clipped_node_events": clipped_node_events,
        "trajectory": trajectory,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
