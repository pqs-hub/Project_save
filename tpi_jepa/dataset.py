"""Dataset construction for single-action TPI world-model transitions."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .bench import parse_bench
from .features import action_type_to_id, make_action_relation_features, make_state_features
from .graph import GraphData, build_graph
from .labels import LabelRow, TransitionSpec, load_labels, row_to_transition


@dataclass
class TransitionSample:
    """One fully tensorized world-model training sample."""

    benchmark_id: str
    sequence_id: str
    step: int
    action_node_name: str
    action_type: str
    state_key: str
    pre_action_count: int
    graph: GraphData
    x_pre: torch.Tensor
    x_post: torch.Tensor
    action_node_id: int
    action_type_id: int
    relation_features: torch.Tensor
    delta_fault_coverage: torch.Tensor
    delta_pattern: torch.Tensor
    has_pattern_target: torch.Tensor
    hard_targets_post: torch.Tensor
    hard_count_post: torch.Tensor
    hard_reduction_target: torch.Tensor
    has_hard_targets: torch.Tensor


@dataclass
class RolloutSample:
    """One tensorized cumulative sequence for latent rollout training."""

    benchmark_id: str
    graph: GraphData
    x_start: torch.Tensor
    x_targets: list[torch.Tensor]
    action_node_ids: list[int]
    action_type_ids: list[int]
    relation_features: list[torch.Tensor]
    delta_fault_coverages: list[torch.Tensor]
    delta_patterns: list[torch.Tensor]
    has_pattern_targets: list[torch.Tensor]
    hard_targets_post: list[torch.Tensor]
    hard_count_post: list[torch.Tensor]
    hard_reduction_targets: list[torch.Tensor]
    has_hard_targets: list[torch.Tensor]


def _read_hard_node_csv(path: Path | None) -> dict[str, tuple[float, float, float]]:
    """Read node-level hard-fault labels as node -> (sa0, sa1, count)."""

    if path is None or not path.exists():
        return {}
    labels: dict[str, tuple[float, float, float]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            node = (row.get("node") or "").strip()
            if not node:
                continue
            output_sa0 = float(row.get("output_sa0") or 0.0)
            output_sa1 = float(row.get("output_sa1") or 0.0)
            input_sa0 = float(row.get("input_edge_sa0") or 0.0)
            input_sa1 = float(row.get("input_edge_sa1") or 0.0)
            sa0 = output_sa0 + input_sa0
            sa1 = output_sa1 + input_sa1
            count = float(row.get("total_undetected_faults") or (sa0 + sa1))
            labels[node] = (float(sa0 > 0.0), float(sa1 > 0.0), count)
    return labels


def _read_hard_summary(path: Path | None) -> tuple[float, float, float] | None:
    """Read graph-level hard-fault totals as (total, sa0, sa1)."""

    if path is None or not path.exists():
        return None
    data = json.loads(path.read_text())
    return (
        float(data.get("undetected_fault_count") or data.get("hard_fault_count") or 0.0),
        float(data.get("undetected_sa0_count") or 0.0),
        float(data.get("undetected_sa1_count") or 0.0),
    )


def _hard_node_tensors(
    graph: GraphData,
    labels: dict[str, tuple[float, float, float]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create dense per-node hard labels aligned to graph nodes."""

    hard = torch.zeros((graph.num_nodes, 2), dtype=torch.float32)
    count = torch.zeros((graph.num_nodes,), dtype=torch.float32)
    if not labels:
        return hard, count, torch.tensor(False, dtype=torch.bool)
    for idx, name in enumerate(graph.node_names):
        sa0, sa1, value = labels.get(name, (0.0, 0.0, 0.0))
        hard[idx, 0] = sa0
        hard[idx, 1] = sa1
        count[idx] = value
    if count.max() > 0:
        count = torch.log1p(count) / torch.log1p(count.max())
    return hard, count, torch.tensor(True, dtype=torch.bool)


def _hard_reduction_target(
    pre_summary: tuple[float, float, float] | None,
    post_summary: tuple[float, float, float] | None,
) -> torch.Tensor:
    """Return normalized reduction ratios for total, SA0, and SA1 hard faults."""

    if pre_summary is None or post_summary is None:
        return torch.zeros((3,), dtype=torch.float32)
    values = []
    for pre, post in zip(pre_summary, post_summary):
        values.append((pre - post) / max(1.0, pre))
    return torch.tensor(values, dtype=torch.float32).clamp(-1.0, 1.0)


def _cache_get(cache: dict[int, Any], idx: int) -> Any | None:
    return cache.get(idx)


def _cache_put(
    cache: dict[int, Any],
    idx: int,
    sample: Any,
    max_entries: int | None,
) -> Any:
    if max_entries is None or max_entries <= 0 or len(cache) < max_entries:
        cache[idx] = sample
    return sample


class TPIDataset(Dataset):
    """Lazy dataset that builds graph transition tensors from label rows."""

    def __init__(
        self,
        rows: list[LabelRow],
        max_specs: int | None = None,
        max_nodes: int | None = None,
        feature_mode: str = "basic",
        relation_mode: str = "basic",
        relation_depth: int = 8,
        state_update_mode: str = "static",
        state_update_depth: int = 8,
        real_fault_prior_path: str | Path | None = None,
        activation_prior_path: str | Path | None = None,
        cache_samples: bool = False,
        sample_cache_max_entries: int | None = None,
    ):
        self.rows = rows
        self.max_specs = max_specs
        self.max_nodes = max_nodes
        self.feature_mode = feature_mode
        self.relation_mode = relation_mode
        self.relation_depth = relation_depth
        self.state_update_mode = state_update_mode
        self.state_update_depth = state_update_depth
        self.real_fault_prior_path = real_fault_prior_path
        self.activation_prior_path = activation_prior_path
        self.cache_samples = cache_samples
        self.sample_cache_max_entries = sample_cache_max_entries
        self._graph_cache: dict[str, GraphData] = {}
        self._base_feature_cache: dict[str, torch.Tensor] = {}
        self._hard_node_cache: dict[str, dict[str, tuple[float, float, float]]] = {}
        self._hard_summary_cache: dict[str, tuple[float, float, float] | None] = {}
        self._sample_cache: dict[int, TransitionSample] = {}
        self._specs = self._filter_specs(rows)

    def _graph_for(self, spec: TransitionSpec) -> GraphData:
        """Load and cache one benchmark graph."""

        if spec.benchmark_id not in self._graph_cache:
            self._graph_cache[spec.benchmark_id] = build_graph(parse_bench(spec.bench_path))
        return self._graph_cache[spec.benchmark_id]

    def _base_features_for(self, spec: TransitionSpec, graph: GraphData) -> torch.Tensor:
        """Load and cache action-independent node features for one benchmark."""

        if spec.benchmark_id not in self._base_feature_cache:
            self._base_feature_cache[spec.benchmark_id] = make_state_features(
                graph,
                [],
                feature_mode=self.feature_mode,
                benchmark_id=spec.benchmark_id,
                real_fault_prior_path=self.real_fault_prior_path,
                activation_prior_path=self.activation_prior_path,
            )[:, :-3]
        return self._base_feature_cache[spec.benchmark_id]

    def _hard_nodes_for(self, path: Path | None) -> dict[str, tuple[float, float, float]]:
        key = str(path) if path is not None else ""
        if key not in self._hard_node_cache:
            self._hard_node_cache[key] = _read_hard_node_csv(path)
        return self._hard_node_cache[key]

    def _hard_summary_for(self, path: Path | None) -> tuple[float, float, float] | None:
        key = str(path) if path is not None else ""
        if key not in self._hard_summary_cache:
            self._hard_summary_cache[key] = _read_hard_summary(path)
        return self._hard_summary_cache[key]

    def _hard_targets_for(self, spec: TransitionSpec, graph: GraphData) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hard, count, has_hard = _hard_node_tensors(
            graph,
            self._hard_nodes_for(spec.post_undetected_node_csv_path),
        )
        reduction = _hard_reduction_target(
            self._hard_summary_for(spec.pre_hard_fault_summary_path),
            self._hard_summary_for(spec.post_hard_fault_summary_path),
        )
        return hard, count, reduction, has_hard

    def _filter_specs(self, rows: list[LabelRow]) -> list[TransitionSpec]:
        """Keep only transitions whose action/history nodes exist in the BENCH graph."""

        specs: list[TransitionSpec] = []
        for row in rows:
            spec = row_to_transition(row)
            graph = self._graph_for(spec)
            if self.max_nodes is not None and graph.num_nodes > self.max_nodes:
                continue
            node_set = set(graph.node_names)
            all_nodes = [node for node, _ in spec.post_actions]
            if spec.action_node not in node_set:
                continue
            if any(node not in node_set for node in all_nodes):
                continue
            specs.append(spec)
            if self.max_specs is not None and len(specs) >= self.max_specs:
                break
        return specs

    def __len__(self) -> int:
        """Return the number of usable transition samples."""

        return len(self._specs)

    def __getitem__(self, idx: int) -> TransitionSample:
        """Build tensors for one transition sample."""

        if self.cache_samples:
            cached = _cache_get(self._sample_cache, idx)
            if cached is not None:
                return cached
        spec = self._specs[idx]
        graph = self._graph_for(spec)
        base = self._base_features_for(spec, graph)
        action_node_id = graph.node_names.index(spec.action_node)
        x_pre = make_state_features(
            graph,
            spec.pre_actions,
            base,
            state_update_mode=self.state_update_mode,
            update_depth=self.state_update_depth,
        )
        x_post = make_state_features(
            graph,
            spec.post_actions,
            base,
            state_update_mode=self.state_update_mode,
            update_depth=self.state_update_depth,
        )
        relation = make_action_relation_features(
            graph,
            action_node_id,
            self.relation_mode,
            self.relation_depth,
        )
        has_pattern = spec.delta_pattern is not None
        delta_pattern = 0.0 if spec.delta_pattern is None else spec.delta_pattern
        hard_targets, hard_count, hard_reduction, has_hard = self._hard_targets_for(spec, graph)
        state_actions = ";".join(f"{node}:{action}" for node, action in sorted(spec.pre_actions))
        state_key = f"{spec.benchmark_id}|{state_actions}"
        sample = TransitionSample(
            benchmark_id=spec.benchmark_id,
            sequence_id=spec.sequence_id,
            step=spec.step,
            action_node_name=spec.action_node,
            action_type=spec.action_type,
            state_key=state_key,
            pre_action_count=len(spec.pre_actions),
            graph=graph,
            x_pre=x_pre,
            x_post=x_post,
            action_node_id=action_node_id,
            action_type_id=action_type_to_id(spec.action_type),
            relation_features=relation,
            delta_fault_coverage=torch.tensor(float(spec.delta_fault_coverage), dtype=torch.float32),
            delta_pattern=torch.tensor(float(delta_pattern), dtype=torch.float32),
            has_pattern_target=torch.tensor(bool(has_pattern), dtype=torch.bool),
            hard_targets_post=hard_targets,
            hard_count_post=hard_count,
            hard_reduction_target=hard_reduction,
            has_hard_targets=has_hard,
        )
        if self.cache_samples:
            return _cache_put(self._sample_cache, idx, sample, self.sample_cache_max_entries)
        return sample

    def cache_info(self) -> dict[str, int | bool | None]:
        return {
            "enabled": self.cache_samples,
            "entries": len(self._sample_cache),
            "max_entries": self.sample_cache_max_entries,
        }


class TPIRolloutDataset(Dataset):
    """Dataset of contiguous label prefixes for multi-step latent rollout."""

    def __init__(
        self,
        rows: list[LabelRow],
        max_specs: int | None = None,
        max_nodes: int | None = None,
        max_horizon: int = 5,
        validate_nodes: bool = True,
        repeat_to_max_specs: bool = False,
        require_full_horizon: bool = False,
        feature_mode: str = "basic",
        relation_mode: str = "basic",
        relation_depth: int = 8,
        state_update_mode: str = "static",
        state_update_depth: int = 8,
        real_fault_prior_path: str | Path | None = None,
        activation_prior_path: str | Path | None = None,
        cache_samples: bool = False,
        sample_cache_max_entries: int | None = None,
    ):
        self.rows = rows
        self.max_specs = max_specs
        self.max_nodes = max_nodes
        self.max_horizon = max(1, int(max_horizon))
        self.validate_nodes = validate_nodes
        self.repeat_to_max_specs = repeat_to_max_specs
        self.require_full_horizon = require_full_horizon
        self.feature_mode = feature_mode
        self.relation_mode = relation_mode
        self.relation_depth = relation_depth
        self.state_update_mode = state_update_mode
        self.state_update_depth = state_update_depth
        self.real_fault_prior_path = real_fault_prior_path
        self.activation_prior_path = activation_prior_path
        self.cache_samples = cache_samples
        self.sample_cache_max_entries = sample_cache_max_entries
        self._graph_cache: dict[str, GraphData] = {}
        self._base_feature_cache: dict[str, torch.Tensor] = {}
        self._node_id_cache: dict[str, dict[str, int]] = {}
        self._hard_node_cache: dict[str, dict[str, tuple[float, float, float]]] = {}
        self._hard_summary_cache: dict[str, tuple[float, float, float] | None] = {}
        self._sample_cache: dict[int, RolloutSample] = {}
        self._sequences = self._build_sequences(rows)
        if (
            self.repeat_to_max_specs
            and self.max_specs is not None
            and self._sequences
            and len(self._sequences) < self.max_specs
        ):
            base = list(self._sequences)
            while len(self._sequences) < self.max_specs:
                need = self.max_specs - len(self._sequences)
                self._sequences.extend(base[:need])

    def _graph_for(self, spec: TransitionSpec) -> GraphData:
        if spec.benchmark_id not in self._graph_cache:
            self._graph_cache[spec.benchmark_id] = build_graph(parse_bench(spec.bench_path))
        return self._graph_cache[spec.benchmark_id]

    def _base_features_for(self, spec: TransitionSpec, graph: GraphData) -> torch.Tensor:
        if spec.benchmark_id not in self._base_feature_cache:
            self._base_feature_cache[spec.benchmark_id] = make_state_features(
                graph,
                [],
                feature_mode=self.feature_mode,
                benchmark_id=spec.benchmark_id,
                real_fault_prior_path=self.real_fault_prior_path,
                activation_prior_path=self.activation_prior_path,
            )[:, :-3]
        return self._base_feature_cache[spec.benchmark_id]

    def _node_ids_for(self, spec: TransitionSpec, graph: GraphData) -> dict[str, int]:
        if spec.benchmark_id not in self._node_id_cache:
            self._node_id_cache[spec.benchmark_id] = {name: idx for idx, name in enumerate(graph.node_names)}
        return self._node_id_cache[spec.benchmark_id]

    def _actions_to_ids(
        self,
        actions: list[tuple[str, str]],
        node_ids: dict[str, int],
    ) -> list[tuple[int, str]]:
        return [(node_ids[node], action_type) for node, action_type in actions]

    def _hard_nodes_for(self, path: Path | None) -> dict[str, tuple[float, float, float]]:
        key = str(path) if path is not None else ""
        if key not in self._hard_node_cache:
            self._hard_node_cache[key] = _read_hard_node_csv(path)
        return self._hard_node_cache[key]

    def _hard_summary_for(self, path: Path | None) -> tuple[float, float, float] | None:
        key = str(path) if path is not None else ""
        if key not in self._hard_summary_cache:
            self._hard_summary_cache[key] = _read_hard_summary(path)
        return self._hard_summary_cache[key]

    def _hard_targets_for(self, spec: TransitionSpec, graph: GraphData) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hard, count, has_hard = _hard_node_tensors(
            graph,
            self._hard_nodes_for(spec.post_undetected_node_csv_path),
        )
        reduction = _hard_reduction_target(
            self._hard_summary_for(spec.pre_hard_fault_summary_path),
            self._hard_summary_for(spec.post_hard_fault_summary_path),
        )
        return hard, count, reduction, has_hard

    def _build_sequences(self, rows: list[LabelRow]) -> list[list[TransitionSpec]]:
        grouped: dict[tuple[str, str], list[LabelRow]] = {}
        for row in rows:
            grouped.setdefault((row.benchmark_id, row.sequence_id), []).append(row)

        sequences: list[list[TransitionSpec]] = []
        for _, group in sorted(grouped.items()):
            ordered = sorted(group, key=lambda row: row.step)
            specs = [row_to_transition(row) for row in ordered]
            if not specs:
                continue
            if self.validate_nodes:
                graph = self._graph_for(specs[0])
                if self.max_nodes is not None and graph.num_nodes > self.max_nodes:
                    continue
                node_set = set(graph.node_names)
                valid_specs = []
                for spec in specs:
                    all_nodes = [node for node, _ in spec.post_actions]
                    if spec.action_node not in node_set:
                        continue
                    if any(node not in node_set for node in all_nodes):
                        continue
                    valid_specs.append(spec)
            else:
                valid_specs = specs
            for start in range(len(valid_specs)):
                seq = valid_specs[start : start + self.max_horizon]
                if seq:
                    if self.require_full_horizon and len(seq) < self.max_horizon:
                        continue
                    sequences.append(seq)
                    if self.max_specs is not None and len(sequences) >= self.max_specs:
                        return sequences
        return sequences

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> RolloutSample:
        if self.cache_samples:
            cached = _cache_get(self._sample_cache, idx)
            if cached is not None:
                return cached
        specs = self._sequences[idx]
        first = specs[0]
        graph = self._graph_for(first)
        base = self._base_features_for(first, graph)
        node_ids = self._node_ids_for(first, graph)
        x_start = make_state_features(
            graph,
            self._actions_to_ids(first.pre_actions, node_ids),
            base,
            state_update_mode=self.state_update_mode,
            update_depth=self.state_update_depth,
        )
        x_targets: list[torch.Tensor] = []
        action_node_ids: list[int] = []
        action_type_ids: list[int] = []
        relation_features: list[torch.Tensor] = []
        delta_fault_coverages: list[torch.Tensor] = []
        delta_patterns: list[torch.Tensor] = []
        has_pattern_targets: list[torch.Tensor] = []
        hard_targets_post: list[torch.Tensor] = []
        hard_count_post: list[torch.Tensor] = []
        hard_reduction_targets: list[torch.Tensor] = []
        has_hard_targets: list[torch.Tensor] = []

        for spec in specs:
            action_node_id = node_ids[spec.action_node]
            x_targets.append(
                make_state_features(
                    graph,
                    self._actions_to_ids(spec.post_actions, node_ids),
                    base,
                    state_update_mode=self.state_update_mode,
                    update_depth=self.state_update_depth,
                )
            )
            action_node_ids.append(action_node_id)
            action_type_ids.append(action_type_to_id(spec.action_type))
            relation_features.append(
                make_action_relation_features(
                    graph,
                    action_node_id,
                    self.relation_mode,
                    self.relation_depth,
                )
            )
            has_pattern = spec.delta_pattern is not None
            delta_pattern = 0.0 if spec.delta_pattern is None else spec.delta_pattern
            delta_fault_coverages.append(torch.tensor(float(spec.delta_fault_coverage), dtype=torch.float32))
            delta_patterns.append(torch.tensor(float(delta_pattern), dtype=torch.float32))
            has_pattern_targets.append(torch.tensor(bool(has_pattern), dtype=torch.bool))
            hard_targets, hard_count, hard_reduction, has_hard = self._hard_targets_for(spec, graph)
            hard_targets_post.append(hard_targets)
            hard_count_post.append(hard_count)
            hard_reduction_targets.append(hard_reduction)
            has_hard_targets.append(has_hard)

        sample = RolloutSample(
            benchmark_id=first.benchmark_id,
            graph=graph,
            x_start=x_start,
            x_targets=x_targets,
            action_node_ids=action_node_ids,
            action_type_ids=action_type_ids,
            relation_features=relation_features,
            delta_fault_coverages=delta_fault_coverages,
            delta_patterns=delta_patterns,
            has_pattern_targets=has_pattern_targets,
            hard_targets_post=hard_targets_post,
            hard_count_post=hard_count_post,
            hard_reduction_targets=hard_reduction_targets,
            has_hard_targets=has_hard_targets,
        )
        if self.cache_samples:
            return _cache_put(self._sample_cache, idx, sample, self.sample_cache_max_entries)
        return sample

    def cache_info(self) -> dict[str, int | bool | None]:
        return {
            "enabled": self.cache_samples,
            "entries": len(self._sample_cache),
            "max_entries": self.sample_cache_max_entries,
        }


def split_by_benchmark(
    rows: list[LabelRow],
    seed: int = 1334,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[list[LabelRow], list[LabelRow], list[LabelRow]]:
    """Split rows by benchmark id so circuits do not leak across splits."""

    bench_ids = sorted({row.benchmark_id for row in rows})
    rng = random.Random(seed)
    rng.shuffle(bench_ids)
    n = len(bench_ids)
    if train_frac + val_frac >= 1.0 and n >= 2:
        train_n = min(n - 1, max(1, int(n * train_frac)))
        val_n = n - train_n
        train_ids = set(bench_ids[:train_n])
        val_ids = set(bench_ids[train_n:])
        return (
            [row for row in rows if row.benchmark_id in train_ids],
            [row for row in rows if row.benchmark_id in val_ids],
            [],
        )
    train_n = max(1, int(n * train_frac))
    val_n = max(1, int(n * val_frac)) if n >= 3 else 0
    if train_n + val_n >= n:
        train_n = max(1, n - 2)
        val_n = 1 if n >= 2 else 0
    train_ids = set(bench_ids[:train_n])
    val_ids = set(bench_ids[train_n : train_n + val_n])
    test_ids = set(bench_ids[train_n + val_n :])
    return (
        [row for row in rows if row.benchmark_id in train_ids],
        [row for row in rows if row.benchmark_id in val_ids],
        [row for row in rows if row.benchmark_id in test_ids],
    )


def collate_one(sample: TransitionSample) -> TransitionSample:
    """Return a single sample unchanged because v1 uses batch size 1."""

    return sample


def _main() -> None:
    """CLI summary used by the step-by-step validation plan."""

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("labels")
    args = parser.parse_args()

    rows = load_labels(Path(args.labels))
    train_rows, val_rows, test_rows = split_by_benchmark(rows)
    demo_rows = [row for row in rows if row.benchmark_id == "iscas89__s838"] or train_rows
    dataset = TPIDataset(demo_rows, max_specs=256)
    sample = dataset[0]
    split_overlap = (
        set(row.benchmark_id for row in train_rows) & set(row.benchmark_id for row in val_rows)
        or set(row.benchmark_id for row in train_rows) & set(row.benchmark_id for row in test_rows)
        or set(row.benchmark_id for row in val_rows) & set(row.benchmark_id for row in test_rows)
    )
    print(f"rows={len(rows)}")
    print(f"train_rows={len(train_rows)} val_rows={len(val_rows)} test_rows={len(test_rows)}")
    print(f"usable_train_samples={len(dataset)}")
    print(f"x_pre_shape={tuple(sample.x_pre.shape)}")
    print(f"x_post_shape={tuple(sample.x_post.shape)}")
    print(f"action_node_valid={0 <= sample.action_node_id < sample.graph.num_nodes}")
    print(f"delta_fc_finite={bool(torch.isfinite(sample.delta_fault_coverage).item())}")
    print(f"hard_targets_shape={tuple(sample.hard_targets_post.shape)}")
    print(f"hard_count_shape={tuple(sample.hard_count_post.shape)}")
    print(f"hard_reduction_target={sample.hard_reduction_target.tolist()}")
    print(f"has_hard_targets={bool(sample.has_hard_targets.item())}")
    print(f"split_overlap={bool(split_overlap)}")


if __name__ == "__main__":
    _main()
