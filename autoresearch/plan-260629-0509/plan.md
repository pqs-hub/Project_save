# AutoResearch Plan: Eval-Like Oracle Action Groups for Action-Value Finetune

generated_at: `2026-06-29 05:09 Asia/Shanghai`

## Goal

Generate more eval-like backend-labeled oracle action groups, then rerun
action-value finetuning and the fixed oracle checkpoint gate.

This follows the negative result from:

```text
autoresearch/fix-260629-0444/fix_report.md
```

## Current Finding

The first oracle-supervised finetune worked mechanically but did not improve the
held-out gate:

```text
train oracle: small subcircuits
held-out gate: b15_C + i2c_aig
candidate verdict: INCONCLUSIVE
candidate hybrid Spearman: 0.318279
incumbent hybrid Spearman: 0.327398
```

The likely reason is not just loss implementation. The training oracle data was:

```text
12 groups from small subcircuits
```

The held-out gate has different circuit scale and a different negative-action
distribution.

## Design

Use eval-like circuits for oracle action-value training, but keep a held-out
gate.

Held-out gate remains:

```text
b15_C,i2c_aig
source: autoresearch/oracle-action-probe-260629-smoke/oracle_actions.tsv
```

New train oracle circuits:

```text
b20_C,b21_C,b22_C,max_aig,b17_C,mem_ctrl_aig
```

These are eval-like but not used in the current b15_C/i2c_aig gate. This gives
training data closer to eval distribution while avoiding direct gate leakage.

## Phase 1: Medium Oracle Expansion

Run a bounded oracle expansion:

```text
6 benchmarks
3 candidate strategies
8 nets per strategy
3 action types per net
```

Estimated action count:

```text
6 * 3 * 8 * 3 = 432 backend evaluations
```

Use `--resume` so interrupted runs can continue without redoing completed
actions.

Command:

```bash
TPI_BENCH_ROOT=/data4/pengqingsong/DFT/Dataset/deeptpi_official_aig_bench_standard \
python scripts/oracle_action_value_probe.py \
  --checkpoint autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt \
  --benchmarks b20_C,b21_C,b22_C,max_aig,b17_C,mem_ctrl_aig \
  --candidate-cache-dir autoresearch/tp-candidates-260626-2047 \
  --candidate-strategies cached_stride,cached_hard_cone,hard_fault_recall_union \
  --score-fields reward_pred,guarded_reward,hard_reduction_total_pred,hybrid_pred \
  --states initial \
  --max-nets 8 \
  --action-types CP0,CP1,OP \
  --top-ks 8,16,24 \
  --oracle-top-m 5 \
  --patterns 10000 \
  --seed 2026 \
  --backend atalanta-bist \
  --plan-device cpu \
  --timeout-sec 14400 \
  --out-dir autoresearch/oracle-action-probe-260629-evallike-train \
  --cleanup-workdir \
  --resume
```

Expected outputs:

```text
oracle_actions.tsv
oracle_groups.tsv
prediction_metrics.tsv
rank_metrics.tsv
state_summary.tsv
manifest.json
handoff.json
```

Minimum acceptable data:

```text
>= 250 finite oracle actions
>= 12 groups with >= 12 finite actions
at least 3 benchmarks represented
some negative actions present, if backend distribution provides them
```

## Phase 2: Finetune on Eval-Like Oracle Groups

Use the new oracle data as training input:

```bash
python scripts/finetune_oracle_action_values.py \
  --checkpoint autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt \
  --oracle-actions autoresearch/oracle-action-probe-260629-evallike-train/oracle_actions.tsv \
  --out-dir autoresearch/oracle-action-value-finetune-260629-evallike \
  --epochs 5 \
  --lr 1e-4 \
  --lambda-oracle-value 1.0 \
  --lambda-oracle-rank 1.0 \
  --pairwise-min-delta 0.001 \
  --pairwise-temperature 1.0 \
  --train-heads reward,return \
  --plan-device cpu
```

Why increase `lambda_oracle_rank`:

```text
Previous run reduced value loss but did not improve rank loss.
With more eval-like groups, ranking should receive a stronger signal.
```

## Phase 3: Held-Out Gate

Keep the original b15_C/i2c_aig oracle probe as held-out:

```bash
python scripts/evaluate_oracle_action_values.py \
  --oracle-actions autoresearch/oracle-action-probe-260629-smoke/oracle_actions.tsv \
  --checkpoints incumbent=autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt,candidate=autoresearch/oracle-action-value-finetune-260629-evallike/candidate.pt \
  --score-fields reward_pred,guarded_reward,hard_reduction_total_pred,hybrid_pred \
  --top-ks 8,16,32 \
  --oracle-top-m 5 \
  --plan-device cpu \
  --out-dir autoresearch/oracle-action-value-gate-260629-evallike \
  --baseline incumbent
```

Promotion criteria:

```text
mean_spearman >= incumbent + 0.10
negative_top1_rate <= incumbent - 0.10
mean_top1_regret <= incumbent - 0.01
```

Guardrail:

```text
REJECT if Spearman improves but negative_top1_rate increases.
```

## Phase 4: Optional Variant If Head-Only Still Fails

Only if Phase 2/3 is `INCONCLUSIVE` or `REJECT`:

```text
run a second finetune with --train-heads reward,return,hard_reduction
```

Reason:

```text
The current best planner score uses hybrid_pred, and hybrid_pred is dominated
partly by hard_reduction_total_pred. If reward-only calibration cannot move the
best score, training hard_reduction_head may be necessary.
```

Variant command:

```bash
python scripts/finetune_oracle_action_values.py \
  --checkpoint autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt \
  --oracle-actions autoresearch/oracle-action-probe-260629-evallike-train/oracle_actions.tsv \
  --out-dir autoresearch/oracle-action-value-finetune-260629-evallike-hardred \
  --epochs 5 \
  --lr 5e-5 \
  --lambda-oracle-value 1.0 \
  --lambda-oracle-rank 1.0 \
  --pairwise-min-delta 0.001 \
  --pairwise-temperature 1.0 \
  --train-heads reward,return,hard_reduction \
  --plan-device cpu
```

## Verification

Before running backend expansion:

```bash
python -m py_compile \
  scripts/oracle_action_value_probe.py \
  scripts/finetune_oracle_action_values.py \
  scripts/evaluate_oracle_action_values.py
```

After oracle expansion:

```bash
test -s autoresearch/oracle-action-probe-260629-evallike-train/oracle_actions.tsv
test -s autoresearch/oracle-action-probe-260629-evallike-train/oracle_groups.tsv
test -s autoresearch/oracle-action-probe-260629-evallike-train/manifest.json
```

After finetune:

```bash
test -s autoresearch/oracle-action-value-finetune-260629-evallike/candidate.pt
test -s autoresearch/oracle-action-value-finetune-260629-evallike/history.tsv
```

After gate:

```bash
test -s autoresearch/oracle-action-value-gate-260629-evallike/oracle_action_value_summary.tsv
test -s autoresearch/oracle-action-value-gate-260629-evallike/handoff.json
```

## Stop Criteria

Stop this iteration if:

```text
oracle expansion produces too few finite rows
backend failures dominate a benchmark
candidate gate is REJECT
candidate gate remains INCONCLUSIVE and reward/hybrid Spearman both decline
```

If stopped due to `INCONCLUSIVE`, inspect:

```text
per-benchmark negative action rate
whether rank loss decreased
whether reward_pred improved but hybrid_pred did not
whether hard_reduction head must be included
```

## Next Command

```text
$autoresearch fix autoresearch/plan-260629-0509/plan.md
```
