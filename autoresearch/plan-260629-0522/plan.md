# AutoResearch Plan Correction: Oracle Action Groups from the 131 Training Subgraphs

generated_at: `2026-06-29 05:22 Asia/Shanghai`

## Correction

This plan supersedes:

```text
autoresearch/plan-260629-0509/plan.md
```

The previous plan was wrong because it proposed using original large eval
circuits as oracle-action training data. Your model training distribution is
the 131 sampled/labeled subgraphs from:

```text
/data4/pengqingsong/DFT/Dataset/atalanta_bist_lowtc_subckt_100k_labels/labels.csv
```

Therefore the action-value finetune data should be generated from those 131
labeled subgraphs, not from the original full circuits.

## Goal

Generate backend-labeled oracle action groups on the same subgraph distribution
used for training, then finetune action-value heads and evaluate in two stages:

```text
1. held-out subgraph oracle gate: checks in-distribution action-value learning
2. full-circuit oracle gate: checks transfer to b15_C/i2c_aig
```

This separates two questions:

```text
Q1: Can the model learn action-value ranking on the sampled-subgraph distribution?
Q2: If yes, does that transfer to full original eval circuits?
```

The previous failed smallckt finetune did not answer Q1 strongly because it used
only 4 tiny subcircuits / 12 groups.

## Data Source

Training-distribution source:

```text
labels.csv benchmark_id set = 131 subckt IDs
bench root = /data4/pengqingsong/DFT/Dataset/atalanta_bist_lowtc_subckt_100k_labels/subcircuits
```

Important:

```text
Do not use all 405 bench files in subcircuits/ unless they appear in labels.csv.
The 131 labeled benchmark_ids define the actual training distribution.
```

## Split

Use a deterministic split over the 131 labeled subckt IDs:

```text
oracle_train_subckts: 24 subckts
oracle_val_subckts: 8 subckts
remaining labeled subckts: unused for this oracle round
```

Selection rule:

```text
sort benchmark_id
take a deterministic stride sample across the 131 IDs
reserve every 4th selected subckt for oracle_val
```

Reason:

```text
This is cheap, reproducible, and avoids training/eval leakage inside the oracle
subgraph experiment.
```

## Phase 1: Build Candidate Cache for Selected Subckts

Use only selected labeled subckts.

Candidate cache command template:

```bash
TPI_BENCH_ROOT=/data4/pengqingsong/DFT/Dataset/atalanta_bist_lowtc_subckt_100k_labels/subcircuits \
python scripts/build_tp_candidate_cache.py \
  --benchmarks <SELECTED_ORACLE_SUBCKTS> \
  --out-dir autoresearch/tp-candidates-labeled-subckt-260629
```

Expected:

```text
one JSON cache per selected subckt
```

## Phase 2: Generate Oracle Groups on Oracle-Train Subckts

Use moderate budget:

```text
24 train subckts
3 candidate strategies
6 nets per strategy
3 action types
```

Estimated backend evaluations:

```text
24 * 3 * 6 * 3 = 1296
```

If too expensive, reduce to:

```text
16 train subckts -> 864 evaluations
```

Command template:

```bash
TPI_BENCH_ROOT=/data4/pengqingsong/DFT/Dataset/atalanta_bist_lowtc_subckt_100k_labels/subcircuits \
python scripts/oracle_action_value_probe.py \
  --checkpoint autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt \
  --benchmarks <ORACLE_TRAIN_SUBCKTS> \
  --candidate-cache-dir autoresearch/tp-candidates-labeled-subckt-260629 \
  --candidate-strategies cached_stride,cached_hard_cone,cached_random \
  --score-fields reward_pred,guarded_reward,hard_reduction_total_pred,hybrid_pred \
  --states initial \
  --max-nets 6 \
  --action-types CP0,CP1,OP \
  --top-ks 8,16,18 \
  --oracle-top-m 5 \
  --patterns 10000 \
  --seed 2026 \
  --backend atalanta-bist \
  --plan-device cpu \
  --timeout-sec 3600 \
  --out-dir autoresearch/oracle-action-probe-260629-labeled-subckt-train \
  --cleanup-workdir \
  --resume
```

## Phase 3: Generate Held-Out Subckt Oracle Gate

Use oracle_val subckts, not used for finetune.

Budget:

```text
8 val subckts
3 candidate strategies
6 nets per strategy
3 action types
= 432 evaluations
```

Command template:

```bash
TPI_BENCH_ROOT=/data4/pengqingsong/DFT/Dataset/atalanta_bist_lowtc_subckt_100k_labels/subcircuits \
python scripts/oracle_action_value_probe.py \
  --checkpoint autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt \
  --benchmarks <ORACLE_VAL_SUBCKTS> \
  --candidate-cache-dir autoresearch/tp-candidates-labeled-subckt-260629 \
  --candidate-strategies cached_stride,cached_hard_cone,cached_random \
  --score-fields reward_pred,guarded_reward,hard_reduction_total_pred,hybrid_pred \
  --states initial \
  --max-nets 6 \
  --action-types CP0,CP1,OP \
  --top-ks 8,16,18 \
  --oracle-top-m 5 \
  --patterns 10000 \
  --seed 2026 \
  --backend atalanta-bist \
  --plan-device cpu \
  --timeout-sec 3600 \
  --out-dir autoresearch/oracle-action-probe-260629-labeled-subckt-val \
  --cleanup-workdir \
  --resume
```

## Phase 4: Finetune on Labeled-Subckt Oracle Train

```bash
python scripts/finetune_oracle_action_values.py \
  --checkpoint autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt \
  --oracle-actions autoresearch/oracle-action-probe-260629-labeled-subckt-train/oracle_actions.tsv \
  --out-dir autoresearch/oracle-action-value-finetune-260629-labeled-subckt \
  --epochs 5 \
  --lr 1e-4 \
  --lambda-oracle-value 1.0 \
  --lambda-oracle-rank 1.0 \
  --pairwise-min-delta 0.001 \
  --pairwise-temperature 1.0 \
  --train-heads reward,return \
  --plan-device cpu
```

## Phase 5: In-Distribution Gate on Held-Out Subckts

```bash
python scripts/evaluate_oracle_action_values.py \
  --oracle-actions autoresearch/oracle-action-probe-260629-labeled-subckt-val/oracle_actions.tsv \
  --checkpoints incumbent=autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt,candidate=autoresearch/oracle-action-value-finetune-260629-labeled-subckt/candidate.pt \
  --bench-root /data4/pengqingsong/DFT/Dataset/atalanta_bist_lowtc_subckt_100k_labels/subcircuits \
  --score-fields reward_pred,guarded_reward,hard_reduction_total_pred,hybrid_pred \
  --top-ks 8,16,18 \
  --oracle-top-m 5 \
  --plan-device cpu \
  --out-dir autoresearch/oracle-action-value-gate-260629-labeled-subckt-val \
  --baseline incumbent
```

This is the primary gate for this plan.

## Phase 6: Transfer Gate on Full Circuits

Only after Phase 5 is not worse, run:

```bash
python scripts/evaluate_oracle_action_values.py \
  --oracle-actions autoresearch/oracle-action-probe-260629-smoke/oracle_actions.tsv \
  --checkpoints incumbent=autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt,candidate=autoresearch/oracle-action-value-finetune-260629-labeled-subckt/candidate.pt \
  --score-fields reward_pred,guarded_reward,hard_reduction_total_pred,hybrid_pred \
  --top-ks 8,16,32 \
  --oracle-top-m 5 \
  --plan-device cpu \
  --out-dir autoresearch/oracle-action-value-gate-260629-labeled-subckt-transfer \
  --baseline incumbent
```

This checks transfer to original b15_C/i2c_aig, but it is not training data.

## Acceptance

Data acceptance:

```text
train oracle: >= 500 finite actions, >= 16 groups
val oracle: >= 150 finite actions, >= 6 groups
```

Model acceptance:

```text
held-out subckt gate improves mean Spearman or top1 regret without increasing negative_top1_rate
```

Strong success:

```text
transfer gate also improves over incumbent
```

## Why This Corrects the Previous Plan

Previous wrong plan:

```text
train action-value on original eval circuits
```

Corrected plan:

```text
train action-value on sampled/labeled training subgraphs
validate on held-out sampled/labeled subgraphs
only then test transfer on original full circuits
```

This matches the actual training distribution and avoids contaminating final
full-circuit evaluation.
