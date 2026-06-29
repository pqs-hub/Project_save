# AutoResearch Plan: Oracle Action-Value Dataset Export and Checkpoint Gate

generated_at: `2026-06-29 03:11 Asia/Shanghai`

## Goal

Implement the first production-quality action-value alignment loop:

```text
1. make scripts/oracle_action_value_probe.py reusable as an oracle dataset builder
2. add a checkpoint comparison gate that scores checkpoints on a fixed oracle_actions.tsv
3. report PROMOTE / REJECT / INCONCLUSIVE using ranking, regret, and negative-top1 metrics
```

This directly follows:

```text
autoresearch/improve-260629-0244/improve_report.md
```

## Problem

The current oracle probe already evaluates candidate actions with the backend
and emits:

```text
oracle_actions.tsv
prediction_metrics.tsv
rank_metrics.tsv
state_summary.tsv
```

But it is still a probe, not a reusable dataset/gate:

```text
1. no manifest.json with reproducible input/config metadata
2. no oracle_groups.tsv summarizing per-state candidate groups
3. no resume/append behavior to avoid re-running expensive backend actions
4. no separate script to compare checkpoints on the same fixed oracle TSV
5. no machine-readable PROMOTE / REJECT / INCONCLUSIVE gate
```

Without the fixed oracle TSV gate, model changes can still improve proxy metrics
while making real action selection worse.

## Scope

In scope:

```text
scripts/oracle_action_value_probe.py
scripts/evaluate_oracle_action_values.py
autoresearch/plan-260629-0311/
```

Out of scope for this plan:

```text
training loss changes
new model heads
full 8-circuit oracle labeling
TMAX/Atalanta backend semantic changes
planner policy changes
large GPU runs
```

## Design

### Part A: Dataset Export Improvements

Extend `scripts/oracle_action_value_probe.py` with:

```text
--resume
--manifest PATH optional, default out_dir/manifest.json
```

Expected behavior:

```text
if --resume and out_dir/oracle_actions.tsv exists:
  load previous rows keyed by:
    benchmark_id, state_id, candidate_strategy, action_key
  reuse existing finite or failed oracle result
  only evaluate missing actions
```

Important detail:

```text
Resume must still recompute model score fields for the current checkpoint when
the checkpoint changes, unless the previous row came from the same checkpoint.
```

Recommended practical implementation:

```text
1. include checkpoint_path and checkpoint_id in manifest.json
2. for --resume, reuse oracle backend columns and eval_dir
3. always overwrite score columns from the current checkpoint
4. write final oracle_actions.tsv atomically enough for normal CLI use
```

Add `oracle_groups.tsv`.

Fields:

```text
benchmark_id
state_id
candidate_strategy
candidate_count
finite_count
ok_count
positive_count
zero_count
negative_count
oracle_best_action
oracle_best_delta_tc
oracle_worst_action
oracle_worst_delta_tc
mean_delta_tc
base_test_coverage
base_fault_coverage
```

Base coverage:

```text
For initial state, base coverage can be derived as:
  oracle_test_coverage - oracle_delta_tc
  oracle_fault_coverage - oracle_delta_fault_coverage

For non-initial future states, use the same derivation from any finite row in
the group.
```

Add `manifest.json`.

Fields:

```text
script
generated_at
checkpoint
checkpoint_id
benchmarks
candidate_strategies
states
max_nets
action_types
patterns
seed
backend
candidate_cache_dir
score_fields
top_ks
oracle_top_m
records
outputs
```

Update `handoff.json` to include:

```text
oracle_groups
manifest
resume_supported: true
```

### Part B: Fixed Oracle Checkpoint Comparison Gate

Add new script:

```text
scripts/evaluate_oracle_action_values.py
```

Purpose:

```text
Given an existing oracle_actions.tsv and one or more checkpoints, rescore the
same action groups with each checkpoint and compute ranking/calibration metrics.
No backend evaluation should run in this script.
```

Required CLI:

```bash
python scripts/evaluate_oracle_action_values.py \
  --oracle-actions PATH \
  --checkpoints NAME=PATH[,NAME=PATH...] \
  --bench-root PATH optional \
  --score-fields reward_pred,guarded_reward,hard_reduction_total_pred,hybrid_pred \
  --top-ks 8,16,32 \
  --oracle-top-m 5 \
  --plan-device cpu \
  --out-dir PATH \
  --baseline NAME optional
```

It may also accept repeated `--checkpoint NAME=PATH`; either comma-separated or
repeatable is fine if documented in help.

Implementation behavior:

```text
1. read oracle_actions.tsv
2. group by benchmark_id, state_id, candidate_strategy
3. for each checkpoint, rebuild graph and latent state for the action group
4. rescore exactly the row actions from oracle_actions.tsv
5. preserve oracle_delta_tc from TSV
6. compute metrics with the same formulas as oracle_action_value_probe.py
7. aggregate by checkpoint and score_field
8. compare against baseline if provided
```

Initial state only is acceptable for this implementation because
`oracle_action_value_probe.py` currently supports only `--states initial`.

Output files:

```text
rescored_oracle_actions.tsv
oracle_action_value_metrics.tsv
oracle_action_value_summary.tsv
oracle_action_value_report.md
handoff.json
```

Recommended `oracle_action_value_summary.tsv` fields:

```text
checkpoint_name
score_field
groups
mean_spearman
mean_kendall_tau
mean_pearson
mean_top1_real_delta_tc
mean_top1_regret
negative_top1_rate
mean_sign_accuracy
mean_calibration_slope
mean_calibration_intercept
verdict_vs_baseline
```

### Verdict Logic

If no baseline is provided:

```text
verdict = INCONCLUSIVE
```

If baseline is provided, compare each candidate checkpoint to baseline using
its best score field by mean Spearman, with guardrails:

Promotion thresholds:

```text
mean_spearman >= baseline_mean_spearman + 0.10
negative_top1_rate <= baseline_negative_top1_rate - 0.10
mean_top1_regret <= baseline_mean_top1_regret - 0.01
```

Rejection guardrail:

```text
REJECT if mean_spearman improves but negative_top1_rate increases
```

Otherwise:

```text
INCONCLUSIVE
```

These thresholds come from:

```text
autoresearch/improve-260629-0244/handoff.json
```

## Reuse Requirements

Do not duplicate metric math if avoidable.

Preferred approach:

```text
Move reusable helpers from scripts/oracle_action_value_probe.py into the new
script by import, or keep them in oracle_action_value_probe.py and import:
  parse_csv_values
  safe_float
  mean
  metric_rows_for_group
  write_tsv
  write_json
  action_key
  score_actions
```

If importing the script creates CLI side effects, it is safe because it already
guards `main()` under:

```python
if __name__ == "__main__":
    main()
```

## Validation Commands

Static validation:

```bash
python -m py_compile \
  scripts/oracle_action_value_probe.py \
  scripts/evaluate_oracle_action_values.py
```

Help validation:

```bash
python scripts/oracle_action_value_probe.py --help
python scripts/evaluate_oracle_action_values.py --help
```

Smoke gate validation using existing oracle TSV:

```bash
python scripts/evaluate_oracle_action_values.py \
  --oracle-actions autoresearch/oracle-action-probe-260629-smoke/oracle_actions.tsv \
  --checkpoints incumbent=autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt \
  --score-fields reward_pred,guarded_reward,hard_reduction_total_pred,hybrid_pred \
  --top-ks 8,16,32 \
  --oracle-top-m 5 \
  --plan-device cpu \
  --out-dir autoresearch/oracle-action-value-gate-260629-smoke \
  --baseline incumbent
```

Resume smoke:

```bash
TPI_BENCH_ROOT=/data4/pengqingsong/DFT/Dataset/deeptpi_official_aig_bench_standard \
python scripts/oracle_action_value_probe.py \
  --checkpoint autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt \
  --benchmarks b15_C \
  --candidate-cache-dir autoresearch/tp-candidates-260626-2047 \
  --candidate-strategies cached_stride \
  --score-fields reward_pred,guarded_reward,hard_reduction_total_pred,hybrid_pred \
  --states initial \
  --max-nets 1 \
  --action-types CP0 \
  --top-ks 1 \
  --oracle-top-m 1 \
  --patterns 10 \
  --seed 2026 \
  --backend atalanta-bist \
  --plan-device cpu \
  --timeout-sec 300 \
  --out-dir autoresearch/oracle-action-probe-260629-resume-smoke \
  --cleanup-workdir

TPI_BENCH_ROOT=/data4/pengqingsong/DFT/Dataset/deeptpi_official_aig_bench_standard \
python scripts/oracle_action_value_probe.py \
  --checkpoint autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt \
  --benchmarks b15_C \
  --candidate-cache-dir autoresearch/tp-candidates-260626-2047 \
  --candidate-strategies cached_stride \
  --score-fields reward_pred,guarded_reward,hard_reduction_total_pred,hybrid_pred \
  --states initial \
  --max-nets 1 \
  --action-types CP0 \
  --top-ks 1 \
  --oracle-top-m 1 \
  --patterns 10 \
  --seed 2026 \
  --backend atalanta-bist \
  --plan-device cpu \
  --timeout-sec 300 \
  --out-dir autoresearch/oracle-action-probe-260629-resume-smoke \
  --cleanup-workdir \
  --resume
```

Expected validation:

```text
1. py_compile passes
2. both --help commands pass
3. gate smoke writes all expected output files
4. gate smoke exits with status 0 and verdict INCONCLUSIVE for baseline-only run
5. resume smoke writes oracle_groups.tsv and manifest.json
6. second resume run does not re-evaluate already present action rows
```

## Acceptance Criteria

Implementation is complete when:

```text
scripts/oracle_action_value_probe.py supports --resume
scripts/oracle_action_value_probe.py writes oracle_groups.tsv
scripts/oracle_action_value_probe.py writes manifest.json
scripts/evaluate_oracle_action_values.py exists
scripts/evaluate_oracle_action_values.py scores fixed oracle action groups without backend calls
scripts/evaluate_oracle_action_values.py writes PROMOTE / REJECT / INCONCLUSIVE verdicts
all validation commands pass
```

## Risks and Mitigations

Risk:

```text
Rescoring a fixed oracle TSV requires reconstructing the same state.
```

Mitigation:

```text
Limit v1 to initial state. Fail fast for non-initial state_id until state action
serialization is added.
```

Risk:

```text
Resume with a different checkpoint could accidentally reuse stale model scores.
```

Mitigation:

```text
Only reuse backend oracle columns. Always recompute model score columns.
```

Risk:

```text
Action rows from old TSV may include candidates unavailable under current
candidate strategy.
```

Mitigation:

```text
The fixed gate should not enumerate candidates. It should score exactly the
actions listed in oracle_actions.tsv.
```

Risk:

```text
Baseline-only run cannot promote or reject.
```

Mitigation:

```text
Return INCONCLUSIVE but still emit metrics. This is useful as a sanity smoke.
```

## Next Step

Run:

```text
$autoresearch fix autoresearch/plan-260629-0311/plan.md
```

The fix should implement Part A and Part B, then run the validation commands
above.
