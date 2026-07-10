# Mainline Plan: Heuristic Recall + World Model + Small-K Rollout

## Scope

Use one project line only:

1. Generate high-recall heuristic candidates with `heuristic_recall_pool`.
2. Score a smaller candidate slice with the world model.
3. Roll out only a small top-K pool with beam search.
4. Validate with eval8.

Out of scope for this phase:

- learned proposer
- full hierarchy
- full action-space value
- Q-ranker/listwise/candidate losses
- calibration-focused work
- additional large proxy-loss stacks

## Code Changes

- `tpi_jepa/plan.py`
  - Added `heuristic_recall_pool` as the fixed candidate policy.
  - Candidate proportions are `hard_fault_cone=50%`, `FFR=20%`, `reconvergence=15%`, `testability=10%`, `diversity=5%`.
  - Added staged planner limits: `--k-recall`, `--k-model`, `--k-plan`.
  - Legacy `--max-candidates` behavior remains unchanged when staged K values are omitted.

- `scripts/run_gmean_sweep.py`
  - Added `--k-recalls`, `--k-models`, `--k-plans`.
  - Records staged K values in result and grouped TSV files.

- `configs/mainline_world_model_simplified.json`
  - Simplified loss:
    - `0.25 * JEPA`
    - `0.70 * hard_reduction`
    - `0.25 * reward`
    - `0.03 * return`
    - `0.03 * hard_aux`
    - `0.01 * delta_scoap`
  - Disabled Q/candidate/rank/pattern/hard-count/brier losses.

- `scripts/run_eval8_hard_fault_cone_budget5.sh`
  - Repointed to this repo and the simplified checkpoint.
  - Uses `heuristic_recall_pool`, `beam_width=2`, `depth=2`, `k=96/32/12`.

- `scripts/run_mainline_ablation_eval8.sh`
  - Runs the four intended ablations:
    - A: heuristic only
    - B: heuristic + world-model single-step rerank
    - C: heuristic + world-model + depth-2 rollout
    - D: depth-3 rollout on small graphs only

## Metrics

Primary eval8 metrics:

- eval8 completion count, target `8/8`
- `macro_mean_delta_tc`
- `negative_count`
- OOM or planner crash count

Candidate diagnostics:

- oracle-best hit@16
- oracle-best hit@32
- oracle-best hit@64

World-model diagnostics:

- hard_reduction MAE@1 / MAE@2
- reward MAE@1 / MAE@2

## Verify

Fast checks already run:

```bash
python -m py_compile tpi_jepa/plan.py scripts/run_gmean_sweep.py
python -m json.tool configs/mainline_world_model_simplified.json
bash -n scripts/run_mainline_ablation_eval8.sh scripts/run_eval8_hard_fault_cone_budget5.sh
```

Train simplified world model:

```bash
python -m tpi_jepa.train --config configs/mainline_world_model_simplified.json
```

Run eval8 mainline:

```bash
CHECKPOINT=runs/mainline_world_model_simplified/best.pt \
bash scripts/run_eval8_hard_fault_cone_budget5.sh
```

Run minimal ablation:

```bash
CHECKPOINT=runs/mainline_world_model_simplified/best.pt \
bash scripts/run_mainline_ablation_eval8.sh
```

Success predicate for this phase:

- eval8 finishes `8/8`
- no OOM on `b22`, `b17`, or `mem_ctrl`
- B beats A on `macro_mean_delta_tc`
- C beats B on `macro_mean_delta_tc`
- `negative_count` does not increase materially versus A
