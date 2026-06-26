# Posweight-30 World-Model Rollout Plan

## Objective

Use the best `posweight-30` checkpoint as the JEPA world model for TPI insertion decisions, then measure real test-coverage gain with the local Atalanta-BIST 300k-pattern simulator.

This plan evaluates 5 inserted test points per circuit on:

```text
/data4/pengqingsong/DFT/Dataset/deeptpi_official_aig_bench_standard
```

## Incumbent Checkpoint

Best `posweight-30` row:

```text
seed=2030
hard_macro_f1_tuned=0.7909739745915929
predictive_score=0.8170630360809562
checkpoint=autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt
```

## Benchmarks

The benchmark ids are the `.bench` stems:

```text
b15_C
b17_C
b20_C
b21_C
b22_C
i2c_aig
max_aig
mem_ctrl_aig
```

The run script sets:

```bash
TPI_BENCH_ROOT=/data4/pengqingsong/DFT/Dataset/deeptpi_official_aig_bench_standard
```

`tpi_jepa.labels.find_bench_path()` has been made portable by checking `TPI_BENCH_ROOT` before the training-label subcircuit directory.

## JEPA Rollout Rule

Use receding-horizon latent rollout, not one-shot full sequence selection:

1. Encode the current circuit state into latent state `z_state`.
2. Enumerate candidate TPI actions that have not been inserted.
3. For each candidate branch, roll the world model forward in latent space.
4. Score the candidate action sequence by predicted hard-fault reduction signal, using `reward_pred`.
5. Select the candidate sequence with highest latent objective.
6. Execute only the first action from that sequence.
7. Update `z_state` with the model-predicted next latent state.
8. Repeat until `budget=5` inserted test points.

This maps to `tpi_jepa.plan --planner beam`, whose `beam_rollout_plan()` repeats beam expansion at every committed step. Do not use `beam_full` for the primary result because that chooses the entire sequence once and does not re-plan after each committed action.

## Primary Variant

Use the current best understood planner setting:

```text
planner=beam
score_field=reward_pred
beam_objective=cumulative
beam_width=4
lookahead_depth=3
max_candidates=96
candidate_strategy=hard_fault_cone
candidate_diversity_penalty=0.0
candidate_diversity_depth=4
budget_mode=fixed
fixed_budget=5
patterns=300000
eval_backend=atalanta-bist
plan_device=cuda
```

Rationale:

- `reward_pred` is the direct world-model hard-reduction reward used for action choice.
- `beam + lookahead_depth=3` implements latent rollout over candidate action sequences.
- `hard_fault_cone` keeps the candidate pool focused on hard-fault reduction.
- `fixed_budget=5` matches the requested five TPI insertions.
- `atalanta-bist` with 300k patterns gives the actual TC/FC measurement.
- `plan_device=cuda` keeps the world-model rollout on GPU; the Atalanta-BIST simulation remains CPU-side and serial.

## Success Metrics

Primary metric:

```text
macro_mean_delta_tc
```

Secondary metrics:

```text
delta_fault_coverage
min_delta_tc
positive_count
negative_count
per-benchmark final delta_test_coverage
```

Promote the planner result if:

1. `status=ok` on all 8 benchmarks.
2. `macro_mean_delta_tc > 0`.
3. No benchmark has a large negative regression. Use `safety_min_delta=-0.005` as the initial guardrail.

## Run Command

Dry run:

```bash
DRY_RUN=1 bash autoresearch/plan-260626-posweight30-world-rollout/run_posweight30_world_rollout.sh
```

Fast smoke check:

```bash
DRY_RUN=1 BENCHMARKS=i2c_aig FIXED_BUDGET=1 CANDIDATE_STRATEGIES=testability MAX_CANDIDATES=1 LOOKAHEAD_DEPTHS=1 BEAM_WIDTHS=1 \
  OUT_DIR=autoresearch/posweight30-world-rollout-260626-smoke \
  bash autoresearch/plan-260626-posweight30-world-rollout/run_posweight30_world_rollout.sh
```

Formal run:

```bash
bash autoresearch/plan-260626-posweight30-world-rollout/run_posweight30_world_rollout.sh
```

Outputs are written to:

```text
autoresearch/posweight30-world-rollout-260626-run-*/
```

Key files:

```text
results.tsv
grouped_results.tsv
plans/*.csv
evals/*/labels.csv
logs/*.plan.log
logs/*.eval.log
```

## Interpretation

Read `grouped_results.tsv` first. The primary single row should report the macro mean TC lift for the 8 official circuits.

Then inspect `results.tsv` for per-circuit failures or negative deltas. If the macro mean is positive but one or more large circuits regress, the next plan should sweep:

```text
beam_width=4,8
lookahead_depth=3,5
max_candidates=64,96,128
candidate_diversity_penalty=0.0,0.02
beam_objective=cumulative,discounted
```

Do not change the checkpoint during this plan; the purpose is to validate whether the current `posweight-30` world model can produce useful real TPI decisions.
