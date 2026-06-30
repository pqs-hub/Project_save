"""End-to-end smoke test for the minimal TPI-JEPA project."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .bench import parse_bench
from .dataset import TPIDataset
from .features import SCOAP_END, SCOAP_START, make_action_relation_features, make_state_features
from .graph import build_graph
from .labels import DEFAULT_LABELS, find_bench_path, load_labels
from .model import TPIWorldModel
from .scoap import compute_scoap_proxy


def main() -> None:
    """Run parser, graph, feature, dataset, model, and backward checks."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    args = parser.parse_args()

    all_rows = load_labels(Path(args.labels))
    preferred = [row for row in all_rows if row.benchmark_id == "iscas89__s838"]
    if preferred:
        rows = preferred
        bench_id = "iscas89__s838"
    elif all_rows:
        bench_id = sorted({row.benchmark_id for row in all_rows})[0]
        rows = [row for row in all_rows if row.benchmark_id == bench_id]
    else:
        raise RuntimeError("No usable rows found for smoke test")

    bench = find_bench_path(bench_id)
    circuit = parse_bench(bench)
    graph = build_graph(circuit)
    scoap = compute_scoap_proxy(graph)
    x = make_state_features(graph, [])
    rel = make_action_relation_features(graph, 0)
    dataset = TPIDataset(rows, max_specs=1)
    sample = dataset[0]

    model = TPIWorldModel(feature_dim=sample.x_pre.shape[1])
    out = model(
        sample.graph,
        sample.x_pre,
        sample.x_post,
        sample.action_node_id,
        sample.action_type_id,
        sample.relation_features,
    )
    scoap_target = sample.x_post[:, SCOAP_START:SCOAP_END]
    loss = (
        torch.nn.functional.mse_loss(out["z_pred"], out["z_t1"])
        + torch.nn.functional.mse_loss(out["scoap_pred"], scoap_target)
        + out["q_pred"].pow(2)
    )
    loss.backward()

    assert graph.num_nodes > 0
    assert scoap.shape[0] == graph.num_nodes
    assert x.shape[0] == graph.num_nodes
    assert rel.shape == (graph.num_nodes, 4)
    assert out["scoap_pred"].shape == (sample.graph.num_nodes, 3)
    assert out["q_pred"].ndim == 0
    assert torch.equal(out["score_pred"], out["q_pred"])
    assert torch.isfinite(out["scoap_pred"]).all().item()
    assert torch.isfinite(loss).item()
    print(f"benchmark_id: {bench_id}")
    print("parse_bench: ok")
    print("build_graph: ok")
    print("compute_scoap_proxy: ok")
    print("make_state_features: ok")
    print("load_labels: ok")
    print("build dataset sample: ok")
    print("model forward: ok")
    print("q head: ok")
    print("scoap head: ok")
    print("one backward step: ok")
    print("smoke_test: ok")


if __name__ == "__main__":
    main()
