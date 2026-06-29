"""Exercise planner initialization across encoder, planner, and relation modes."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tpi_jepa.bench import parse_bench
from tpi_jepa.features import make_action_relation_features, make_state_features
from tpi_jepa.graph import build_graph
from tpi_jepa.labels import find_bench_path
from tpi_jepa.model import TPIWorldModel
from tpi_jepa.plan import beam_rollout_plan, greedy_plan


def _split_csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-id", default="subckt_0001")
    parser.add_argument("--encoders", default="mean,gate_dir")
    parser.add_argument("--planners", default="greedy,beam")
    parser.add_argument("--relations", default="basic,cone")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--budget", type=int, default=1)
    parser.add_argument("--max-candidates", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device)
    graph = build_graph(parse_bench(find_bench_path(args.benchmark_id)))
    feature_dim = make_state_features(graph, []).shape[1]
    passed = 0
    for encoder in _split_csv(args.encoders):
        for relation in _split_csv(args.relations):
            rel_dim = make_action_relation_features(graph, 0, relation).shape[1]
            model = TPIWorldModel(
                feature_dim=feature_dim,
                relation_dim=rel_dim,
                encoder_type=encoder,
                latent_dim=16,
                encoder_layers=1,
            ).to(device)
            model.eval()
            for planner in _split_csv(args.planners):
                if planner == "greedy":
                    rows = greedy_plan(
                        model,
                        graph,
                        args.budget,
                        device,
                        max_candidates=args.max_candidates,
                        relation_mode=relation,
                    )
                elif planner == "beam":
                    rows = beam_rollout_plan(
                        model,
                        graph,
                        args.budget,
                        device,
                        max_candidates=args.max_candidates,
                        beam_width=2,
                        lookahead_depth=1,
                        relation_mode=relation,
                    )
                else:
                    raise SystemExit(f"unsupported planner: {planner}")
                if not rows:
                    raise SystemExit(f"planner returned no rows: encoder={encoder} planner={planner} relation={relation}")
                print(f"pass encoder={encoder} planner={planner} relation={relation}")
                passed += 1
    print(f"matrix_passed={passed}")


if __name__ == "__main__":
    main()
