# TPI-my.3: AIG TPI World Model

This project is copied from `/data3/pengqingsong/DFT/TPI-my.2` and adapted
for the Atalanta_BIST low-TC AIG subcircuit dataset.

If you want the current codebase map and run-order, start with
[`docs/codebase_guide.md`](/data4/pengqingsong/DFT/TPI-my.3/docs/codebase_guide.md).

The first version intentionally keeps the existing TPI-JEPA world-model
backbone so that we have a clean baseline before changing the architecture:

- input circuit format: AIG-style `.bench`
- training labels: Atalanta_BIST 300k random-pattern sequence labels
- transition target: per-step `delta_test_coverage` with
  `delta_fault_coverage` as a backward-compatible fallback
- sequence horizon: up to 5 test-point insertions
- split rule: by `benchmark_id`, so the same subcircuit does not appear in
  both train and validation

## AIG Dataset

Default label file:

```text
/data3/pengqingsong/DFT/Dataset/atalanta_bist_lowtc_subckt_100k_labels/labels.csv
```

Default AIG BENCH root:

```text
/data3/pengqingsong/DFT/Dataset/atalanta_bist_lowtc_subckt_100k_labels/subcircuits/{benchmark_id}.bench
```

Dataset summary:

```text
baseline labels: 131
step labels:     100000
sequences:       20000
sequence length: 5
circuits:        131 low-TC AIG subcircuits
patterns:        300000
seed:            2026
backend:         atalanta_bist
```

## Quick Start

Run these checks from the project root:

```bash
cd /data3/pengqingsong/DFT/TPI-my.3

python -m py_compile tpi_jepa/*.py
python -m tpi_jepa.inspect_data
python -m tpi_jepa.labels
python -m tpi_jepa.dataset /data3/pengqingsong/DFT/Dataset/atalanta_bist_lowtc_subckt_100k_labels/labels.csv
python -m tpi_jepa.smoke_test
python -m tpi_jepa.train --config configs/aig_lowtc_100k_world_model_smoke.json
```

For a first full baseline run:

```bash
cd /data3/pengqingsong/DFT/TPI-my.3
CUDA_VISIBLE_DEVICES=0 python -m tpi_jepa.train \
  --config configs/aig_lowtc_100k_world_model_full.json
```

Outputs are written under:

```text
runs/aig_lowtc_100k_smoke
runs/aig_lowtc_100k_full
```

## Model V1

The current model is the conservative first design:

- node encoder: bidirectional message passing over the AIG graph
- action encoder: selected node latent plus one of `CP0`, `CP1`, `OP`
- dynamics: predicts next latent graph state after inserting one test point
- auxiliary head: predicts next-state SCOAP proxy
- reward head: predicts scaled per-step TC improvement
- rollout training: predicts 5-step TPI sequences in latent space

This gives us a runnable AIG-input baseline. The next architecture decisions
can be made one by one on top of this baseline, for example whether to add
undetected-fault priors, candidate-ranking heads, or a DeepGate-style
pretrained encoder.

# Original TPI-JEPA Minimal World Model

This project is a from-scratch, educational implementation of a minimal
TPI-JEPA-style world model for test point insertion.

## Legacy Data From TPI-my.2

The following paths are kept only as historical context from `TPI-my.2`.
They are not the default data source for `TPI-my.3`.

Old label file:

```text
/data3/pengqingsong/DFT/Dataset/results/random_labels/tmax_tc_lt_099_s1334_k5_300k_parallel/labels.csv
```

Old BENCH lookup order:

```text
/data3/pengqingsong/DFT/Dataset/stage1_jepa_pretrain_bench/bench/{benchmark_id}.bench
/data3/pengqingsong/DFT/Dataset/real_label_bench_standard/{benchmark_id}.bench
```

## Fixed Evaluation Protocol

Final coverage-only planner comparisons should use:

```text
configs/eval_protocol_coverage_only.json
```

This protocol fixes the evaluation task to the eight circuits in the selected
benchmark table:

```text
b15_C    -> iscas99__b15_1
b20_C    -> iscas99__b20
b21_C    -> iscas99__b21
b22_C    -> iscas99__b22
i2c      -> epfl__random_control__i2c__i2c
max      -> epfl__arithmetic__max__max
b17_C    -> iscas99__b17
mem_ctrl -> openabcd__mem_ctrl_orig
```

It also fixes `patterns=300000`, `seed=2026`, and `budget_mode=floor1pct`.
The TP budget rule is `max(1, floor(logic_gate_count * 0.01))`, so the number
of inserted test points never exceeds one percent of logic gates.
Baseline TC is measured by the local TMAX flow on the local BENCH files; the
TC values from the source table are kept only as references in the protocol.
The primary selection metric is `macro_mean_delta_tc`, with safe variants
preferred first and ties broken by `min_delta_tc`, `router_delta_tc`, then
`negative_count`.

Training configs now support strict train/eval circuit isolation:

```json
"exclude_eval_protocol": "configs/eval_protocol_coverage_only.json",
"exclude_protocol_auxiliary": true
```

This removes every benchmark listed in the fixed evaluation protocol, plus its
safety/development benchmarks, before the train/validation split. Additional
circuits can be excluded with `exclude_benchmarks`.

## Testability-Aware Sweep

The framework sweep exposes the main scalability ideas as named ablations:

```text
baseline       original features/relation, netlist-order candidates
region         hard-region / FFR / reconvergence / transparent-chain features
cone           action-conditioned cone relation features
sparse         fault-path weighted sparse message passing
hard_fault     hard-fault-region candidate generation
reconvergence  reconvergence-aware candidate generation
ffr            FFR-span-aware candidate generation
mixed          region + cone + mixed candidate scoring
diversity      mixed scoring with local-region diversity penalty
full_tac       region + cone + sparse fault-path MP + mixed/diverse planner
```

The AutoResearch-style control file is:

```text
autoresearch/tac_program.md
```

It plays the same role as Karpathy's `program.md`: it defines the research
goal, allowed edit scope, leakage rules, and metric. Humans edit that file
between runs; scripts produce measured artifacts under `autoresearch/`.

Run the 7-GPU sweep on cards 0-6:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python scripts/overnight_framework_search.py \
  --eval-protocol configs/eval_protocol_coverage_only.json \
  --framework-variants baseline,region,cone,sparse,hard_fault,reconvergence,ffr,mixed,diversity,full_tac \
  --parallel-jobs 7 \
  --devices 0,1,2,3,4,5,6 \
  --benchmark-id iscas89__s838 \
  --out-dir autoresearch/tac-framework-sweep
```

After a sweep finishes, summarize technique effectiveness with:

```bash
python scripts/summarize_tac_sweep.py \
  --results autoresearch/tac-framework-sweep/results.tsv
```

For a single final combined model:

```bash
CUDA_VISIBLE_DEVICES=0 python -m tpi_jepa.train --config configs/tmax50k_tac_full5.json
```

After a fixed-protocol run finishes, render the TC-improvement table with:

```bash
python scripts/report_table8.py \
  --results autoresearch/gmean-coverage-only-table8/results.tsv \
  --protocol configs/eval_protocol_coverage_only.json \
  --method-name TPI-JEPA \
  --out-md autoresearch/gmean-coverage-only-table8/table8_report.md
```

Example:

```bash
python scripts/run_gmean_sweep.py \
  --eval-protocol configs/eval_protocol_coverage_only.json \
  --checkpoint runs/tmax50k_coverage_only_full5/best.pt \
  --score-fields reward_pred \
  --planners beam \
  --beam-objectives cumulative,discounted,terminal \
  --beam-widths 4,8,16 \
  --lookahead-depths 3,5,7 \
  --max-candidates 64,96,128,192 \
  --candidate-strategies checkpoint,testability,mixed,hard_fault,reconvergence,ffr \
  --candidate-diversity-penalties 0.0,0.02 \
  --discount-gammas 0.8,0.9,0.95 \
  --out-dir autoresearch/gmean-coverage-only-fixed-v1
```

## Function Map

| File | Function/Class | Input | Output | Purpose | Why |
|---|---|---|---|---|---|
| `tpi_jepa/__init__.py` | package marker | import request | package version | Allows `import tpi_jepa` | Confirms the project is importable |
| `tpi_jepa/bench.py` | `Circuit` | parsed BENCH fields | structured netlist | Stores nodes, gates, fanins, inputs, outputs | Keeps parsing separate from tensors |
| `tpi_jepa/bench.py` | `parse_bench` | BENCH path | `Circuit` | Parses `INPUT`, `OUTPUT`, assignments | First step from raw circuit file to code |
| `tpi_jepa/bench.py` | `_decode_lut` | LUT params, args | gate, args, optional mask | Converts recognizable LUT truth tables to ordinary gates | Makes LUT-heavy BENCH files easier to interpret |
| `tpi_jepa/bench.py` | `_expand_lut_to_sop` | LUT lhs, args, mask | synthesized NOT/AND/OR nodes | Expands arbitrary LUT truth tables up to a guarded input limit, with minimal handling for common 2-input inverted-input LUTs | Lets complex LUTs participate in the normal graph representation |
| `tpi_jepa/bench.py` | `_parse_assignment` | one assignment line | lhs, gate, args, optional LUT mask | Supports gates, decoded LUTs, constants, aliases | Handles EPFL BENCH syntax |
| `tpi_jepa/graph.py` | `GraphData` | graph tensors/lists | reusable graph object | Stores edge tensors and masks | Gives every later module one graph format |
| `tpi_jepa/graph.py` | `build_graph` | `Circuit` | `GraphData` | Converts node names to integer ids and treats DFFs as scan-style PI/PO boundaries by default | Neural code needs integer tensors |
| `tpi_jepa/graph.py` | `compute_structural_features` | `GraphData` | tensor `[N, 6]` | Computes normalized local structure | Adds cheap topology signal before learning |
| `tpi_jepa/scoap.py` | `compute_scoap_proxy` | `GraphData` | tensor `[N, 3]` | Computes controllability/observability proxies | Gives the model testability features |
| `tpi_jepa/features.py` | `action_type_to_id` | action string | integer id | Normalizes action names | Keeps labels and model embeddings consistent |
| `tpi_jepa/features.py` | `make_base_node_features` | graph | tensor `[N, F-3]` | Builds action-independent features | Lets datasets cache expensive SCOAP/structure work |
| `tpi_jepa/features.py` | `make_state_features` | graph, inserted actions | tensor `[N, F]` | Builds node input features | Represents current TPI state without patched BENCH files |
| `tpi_jepa/features.py` | `make_action_relation_features` | graph, action node id | tensor `[N, 4]` | Marks nodes around the action | Lets dynamics know where the candidate action is |
| `tpi_jepa/labels.py` | `LabelRow` | CSV row fields | typed label row | Stores one usable action label | Filters bad rows early |
| `tpi_jepa/labels.py` | `TransitionSpec` | label row plus BENCH path | pre/action/post spec | Describes one world-model transition | Keeps CSV parsing separate from tensors |
| `tpi_jepa/labels.py` | `load_labels` | labels CSV | list of `LabelRow` | Loads valid action labels | Defines the training data subset |
| `tpi_jepa/labels.py` | `find_bench_path` | benchmark id | BENCH path | Resolves source circuit file | Makes missing data explicit |
| `tpi_jepa/labels.py` | `row_to_transition` | `LabelRow` | `TransitionSpec` | Builds pre/post action histories | Converts sequence labels into transitions |
| `tpi_jepa/dataset.py` | `TransitionSample` | graph transition tensors | one training sample | Holds model-ready data | Makes sample contents explicit |
| `tpi_jepa/dataset.py` | `TPIDataset` | label rows | lazy PyTorch dataset | Builds graph tensors from labels | Keeps memory use simple; `max_specs` and `max_nodes` keep tests small |
| `tpi_jepa/dataset.py` | `split_by_benchmark` | label rows, seed | train/val/test rows | Splits by circuit id | Prevents same-circuit leakage |
| `tpi_jepa/dataset.py` | `collate_one` | one sample | same sample | Batch-size-1 collate helper | Avoids padding variable-size graphs in v1 |
| `tpi_jepa/model.py` | `mean_aggregate` | edges, node states | aggregated states | Implements mean message passing | Avoids external graph libraries |
| `tpi_jepa/model.py` | `NodeEncoder` | node features, edges | node latent `z` | Encodes the circuit state | Produces the JEPA latent space |
| `tpi_jepa/model.py` | `ActionEncoder` | selected node latent, type id | action embedding | Encodes CP0/CP1/OP | Conditions dynamics on the action |
| `tpi_jepa/model.py` | `DynamicsPredictor` | `z`, action embedding, relations | predicted next latent state | Predicts the next JEPA state | Core world-model transition |
| `tpi_jepa/model.py` | `TPIWorldModel` | transition sample tensors | next-latent predictions and scores | Combines encoder, dynamics, heads | End-to-end trainable model |
| `tpi_jepa/model.py` | `predict_from_latent` | current latent state, action, relations | predicted next latent, SCOAP, and scores | Rolls the world model forward without re-encoding `x_pre` | Enables latent-space planning |
| `tpi_jepa/model.py` | `update_ema` | target, online, decay | in-place target update | Maintains target encoder | Stabilizes JEPA training |
| `tpi_jepa/train.py` | `load_config` | JSON path | config dict | Loads training settings | Keeps CLI simple |
| `tpi_jepa/train.py` | `compute_loss` | sample batch, model | loss and metrics | Combines JEPA/SCOAP/fc/pattern/score losses | Defines the learning objective |
| `tpi_jepa/train.py` | `train_one_epoch` | model, dataset, optimizer | metrics | Performs optimization steps | Smallest repeatable training loop |
| `tpi_jepa/train.py` | `evaluate` | model, dataset | metrics | Measures validation loss | Tracks whether training improves |
| `tpi_jepa/train.py` | `save_checkpoint` | path, model, config | checkpoint file | Saves weights and metadata | Enables planning after training |
| `tpi_jepa/plan.py` | `load_checkpoint` | checkpoint path | model, config | Loads trained model | Reuses training output |
| `tpi_jepa/plan.py` | `enumerate_candidates` | graph, selected actions | candidate list | Lists node/action pairs, skipping parser-synthesized LUT helper nodes | Defines search space |
| `tpi_jepa/plan.py` | `score_candidate_from_latent` | model, latent state, candidate | predicted scores and next latent | Scores one action | Connects model to planning |
| `tpi_jepa/plan.py` | `greedy_plan` | model, graph, budget | selected rows | Greedy action selection by iterating latent state | Uses the model as a rollout world model |
| `tpi_jepa/plan.py` | `write_plan_csv` | path, rows | CSV file | Saves selected actions | Makes plans inspectable |
| `tpi_jepa/inspect_data.py` | `inspect_labels` | labels CSV | stats dict | Summarizes label availability | Checks data before training |
| `tpi_jepa/smoke_test.py` | `main` | labels path | printed checks | Runs parser-to-backward sanity test | Confirms the whole minimal stack works |
| `tpi_jepa/train.py` | `load_config` | JSON path | config dict | Loads training settings | Keeps runs reproducible |
| `tpi_jepa/train.py` | `compute_loss` | sample, model, config | loss and metrics | Combines JEPA/SCOAP/fc/pattern/score losses | Defines the learning objective |
| `tpi_jepa/train.py` | `train_one_epoch` | model, dataset, optimizer | averaged metrics | Runs shuffled optimization steps | Updates online and EMA target encoders |
| `tpi_jepa/train.py` | `evaluate` | model, dataset | averaged metrics | Measures validation loss | Tracks training health |
| `tpi_jepa/train.py` | `save_checkpoint` | path, model, config | checkpoint file | Saves train state for planning | Makes learned model reusable |
