# AutoResearch Plan: Pairwise-Only Oracle Action Ranking

generated_at: `2026-06-29 10:04 Asia/Shanghai`

## Decision

Do not add sign/top1/regret losses in this iteration.

Do not add a dedicated action-value head in this iteration.

Only optimize the core objective:

```text
If oracle_delta_tc(action_i) > oracle_delta_tc(action_j),
then model_score(action_i) > model_score(action_j).
```

## Why

The actual goal is ranking consistency:

```text
model action-value ordering ~= backend oracle_delta_tc ordering
```

Adding four losses at once would make the result hard to interpret:

```text
pairwise rank
sign calibration
top1 negative penalty
top1 regret
```

If performance changes, we would not know which objective caused it.

Therefore use a clean first test:

```text
L = pairwise_rank_loss only
```

Then evaluate:

```text
Spearman
Kendall
top1 regret
negative_top1_rate as guardrail
```

## Current Script State

`scripts/finetune_oracle_action_values.py` already has:

```text
pointwise SmoothL1 value loss
pairwise logistic rank loss
```

Current total:

```text
L = lambda_oracle_value * value_loss
  + lambda_oracle_rank  * rank_loss
```

So pairwise-only training can be done without new model code by running:

```text
--lambda-oracle-value 0.0
--lambda-oracle-rank 1.0
```

This is preferred before adding new loss terms.

## Implementation Scope

No code changes are required unless the script does not correctly support:

```text
--lambda-oracle-value 0.0
```

If needed, only patch `scripts/finetune_oracle_action_values.py` to ensure:

```text
zero value weight is valid
history clearly records value/rank loss
```

## Training Command

```bash
python scripts/finetune_oracle_action_values.py \
  --checkpoint autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt \
  --oracle-actions autoresearch/oracle-action-probe-260629-labeled-subckt-train/oracle_actions.tsv \
  --out-dir autoresearch/oracle-action-value-finetune-260629-pairwise-only \
  --epochs 5 \
  --lr 1e-4 \
  --lambda-oracle-value 0.0 \
  --lambda-oracle-rank 1.0 \
  --pairwise-min-delta 0.001 \
  --pairwise-temperature 1.0 \
  --train-heads reward,return \
  --plan-device cpu
```

## Primary Gate

Held-out labeled subckt gate:

```bash
python scripts/evaluate_oracle_action_values.py \
  --oracle-actions autoresearch/oracle-action-probe-260629-labeled-subckt-val/oracle_actions.tsv \
  --checkpoints incumbent=autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt,candidate=autoresearch/oracle-action-value-finetune-260629-pairwise-only/candidate.pt \
  --bench-root /data4/pengqingsong/DFT/Dataset/atalanta_bist_lowtc_subckt_100k_labels/subcircuits \
  --score-fields reward_pred,guarded_reward,hard_reduction_total_pred,hybrid_pred \
  --top-ks 8,16,18 \
  --oracle-top-m 5 \
  --plan-device cpu \
  --out-dir autoresearch/oracle-action-value-gate-260629-pairwise-only-val \
  --baseline incumbent
```

## Transfer Gate

Only run after primary gate completes:

```bash
python scripts/evaluate_oracle_action_values.py \
  --oracle-actions autoresearch/oracle-action-probe-260629-smoke/oracle_actions.tsv \
  --checkpoints incumbent=autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt,candidate=autoresearch/oracle-action-value-finetune-260629-pairwise-only/candidate.pt \
  --score-fields reward_pred,guarded_reward,hard_reduction_total_pred,hybrid_pred \
  --top-ks 8,16,32 \
  --oracle-top-m 5 \
  --plan-device cpu \
  --out-dir autoresearch/oracle-action-value-gate-260629-pairwise-only-transfer \
  --baseline incumbent
```

## Acceptance

Progress if:

```text
mean Spearman or Kendall improves on held-out subckt gate
and negative_top1_rate does not increase
```

Reject if:

```text
negative_top1_rate increases
```

Do not require top1 safety improvement yet. This experiment is specifically to
test whether pairwise ranking alone improves ordering.

## Next If It Fails

If pairwise-only does not improve ranking:

```text
1. pairwise loss implementation/temperature may be ineffective
2. reward_head may not be the right score carrier
3. action_value_head becomes more justified
```

If pairwise improves ranking but negative_top1 worsens:

```text
then add sign/top1 safety loss as the next isolated experiment
```
