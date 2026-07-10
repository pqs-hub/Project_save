# TPI-my.3 Codebase Guide

This note is a compact map of the current codebase. It is intended to help a
new contributor understand where the data comes from, how it becomes tensors,
what the model predicts, and which commands are worth running first.

## What This Project Does

`TPI-my.3` trains a JEPA-style world model for test point insertion on BENCH
circuits. The model learns from labeled action transitions and predicts:

- next latent circuit state
- SCOAP proxy values
- hard-fault node labels
- hard-fault count and reduction signals
- reward / coverage-style scores for planning

The current implementation is centered on the low-TC AIG subcircuit dataset,
with support for legacy BENCH roots and fixed evaluation protocols.

## Main Data Flow

1. `tpi_jepa/labels.py` reads `labels.csv`, filters usable rows, and resolves
   each benchmark to a local `.bench` file.
2. `tpi_jepa/dataset.py` turns one label row into a graph transition sample.
3. `tpi_jepa/bench.py` parses BENCH netlists into a simple circuit structure.
4. `tpi_jepa/graph.py` converts the circuit into tensor-friendly graph data.
5. `tpi_jepa/features.py` builds node features and action masks.
6. `tpi_jepa/model.py` encodes the graph, conditions on the action, and
   predicts the next-state targets.
7. `tpi_jepa/train.py` trains the model and writes checkpoints.
8. `tpi_jepa/plan.py` loads a checkpoint and produces candidate plans.

## Core Modules

### `tpi_jepa/labels.py`

Responsibilities:

- load the CSV labels
- map raw action types (`CP0`, `CP1`, `OP`) to canonical actions
- locate the matching BENCH file for each benchmark
- reconstruct pre-action and post-action state from the insertion sequence

Important environment behavior:

- `DFT_ROOT` overrides the default DFT root
- `TPI_BENCH_ROOT` can prepend extra BENCH search roots
- legacy `/data3/...` paths are remapped to the local `/data4/...` tree when
  possible

### `tpi_jepa/bench.py`

Parses a small BENCH subset into a `Circuit` object. It also:

- handles simple assignment forms
- expands common LUT patterns when possible
- normalizes constants, buffers, and unary gates

### `tpi_jepa/graph.py`

Builds a `GraphData` object with:

- node names
- gate type ids
- directed edge tensors
- fanin and fanout adjacency lists
- input and output masks

It treats DFFs as a combinational boundary when building the tensor graph.

### `tpi_jepa/features.py`

Builds action-independent and action-conditioned node features.

Feature blocks include:

- gate one-hot encoding
- structural features
- SCOAP proxy features
- optional real-fault priors
- optional activation priors
- action masks for `control0`, `control1`, and `observe`

State updates are conservative. They do not patch the netlist; they only update
feature proxies.

### `tpi_jepa/dataset.py`

Produces two sample types:

- `TransitionSample` for one action step
- `RolloutSample` for cumulative multi-step training

It also loads hard-fault supervision from sidecar artifacts:

- node-level undetected fault CSVs
- graph-level hard-fault summary JSON

### `tpi_jepa/model.py`

Contains the world model:

- `NodeEncoder`
- `ActionEncoder`
- `DynamicsPredictor`
- `ResidualHardHead`
- `TPIWorldModel`

The model uses an online encoder plus EMA target encoder. Heads predict latent
state, SCOAP, hard-fault labels, hard-fault counts, reduction signals, and
reward-like values.

### `tpi_jepa/train.py`

Training entry point.

Typical behavior:

- load config JSON
- load and split labels by benchmark
- exclude benchmarks from a fixed evaluation protocol when configured
- build either `TPIDataset` or rollout dataset variants
- train for `epochs`
- write `latest.pt`, `best.pt`, `history.csv`, and optional epoch checkpoints

Common config fields:

- `labels`
- `run_dir`
- `seed`
- `feature_mode`
- `relation_mode`
- `rollout_training`
- `rollout_max_horizon`
- `lambda_hard`, `lambda_hard_count`, `lambda_hard_reduction`
- `exclude_eval_protocol`

### `tpi_jepa/plan.py`

Loads a trained checkpoint and generates plans for a benchmark.

Supported planner modes:

- `greedy`
- `beam`
- `beam_full`

Supported candidate strategies include:

- `netlist`
- `testability`
- `hard_fault`
- `hard_fault_cone`
- `hard_fault_ranked`
- `hard_fault_recall_union`
- `reconvergence`
- `ffr`
- `mixed`
- cached variants such as `cached_netlist` and `cached_hard_cone`

## Useful Commands

Run these from the project root.

```bash
python -m py_compile tpi_jepa/*.py
python -m tpi_jepa.inspect_data
python -m tpi_jepa.smoke_test
python -m tpi_jepa.train --config configs/aig_lowtc_100k_world_model_smoke.json
python -m tpi_jepa.plan --checkpoint runs/aig_lowtc_100k_smoke/best.pt --benchmark-id iscas89__s838 --budget 5
```

For the fixed final evaluation protocol, use:

```text
configs/eval_protocol_coverage_only.json
```

That protocol has strict restored DeepTPI Table-II `#TPs` budgets. Do not
recompute budgets from the current BENCH parser when comparing against Table-II
or historical 8-circuit results.

## Where Outputs Go

Training and sweep artifacts are typically written under:

- `runs/`
- `autoresearch/`
- `docs/`

Generated planning CSVs are written next to the checkpoint unless `--out` is
specified.

## Practical Notes

- `labels.py` is the best place to inspect if a benchmark fails to resolve.
- `dataset.py` is the place to inspect if a sample disappears during filtering.
- `features.py` is the place to inspect if a feature dimension changes.
- `plan.py` should be checked when a candidate strategy or scoring field is
  wrong.
- The smoke test is the fastest end-to-end sanity check because it exercises
  parsing, graph building, feature generation, dataset construction, model
  forward, and backward propagation.
