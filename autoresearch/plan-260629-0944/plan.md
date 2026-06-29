# AutoResearch Plan: Sign and Top1 Safety Loss for Oracle Action-Value Training

generated_at: `2026-06-29 09:44 Asia/Shanghai`

## Goal

Fix the observed oracle action-value failure mode:

```text
ranking can improve locally, but the model still selects harmful top1 actions
too often.
```

Implement explicit sign/top1 safety losses first. Treat a dedicated
action-value head as a second-stage option only if safety loss cannot fix the
current reward/hybrid scoring path.

## Evidence

Source:

```text
autoresearch/fix-260629-0532/fix_report.md
```

Held-out sampled-subckt gate:

| checkpoint | score_field | mean Spearman | negative top1 | top1 regret |
|---|---|---:|---:|---:|
| incumbent | hybrid_pred | 0.005693 | 0.458333 | 0.043452 |
| candidate | hybrid_pred | 0.004533 | 0.416667 | 0.043300 |
| incumbent | reward_pred | -0.137870 | 0.416667 | 0.023290 |
| candidate | reward_pred | -0.205044 | 0.541667 | 0.031650 |

Full-circuit transfer gate:

| checkpoint | score_field | mean Spearman | negative top1 | top1 regret |
|---|---|---:|---:|---:|
| incumbent | hybrid_pred | 0.327398 | 0.166667 | 0.012552 |
| candidate | hybrid_pred | 0.325323 | 0.166667 | 0.012557 |
| incumbent | reward_pred | 0.294742 | 0.500000 | 0.020223 |
| candidate | reward_pred | 0.348502 | 0.333333 | 0.022373 |

Interpretation:

```text
reward_pred improved transfer Spearman but still had worse negative_top1 than
incumbent best field. The gate correctly rejected it.
```

Therefore the next objective should not be just "more ranking loss". It should
explicitly optimize:

```text
1. sign calibration
2. top1 safety
3. top1 regret
```

## Stage A: Add Safety Losses to Existing Finetune Script

Modify:

```text
scripts/finetune_oracle_action_values.py
```

Do not change model architecture in Stage A.

### New CLI Args

Add:

```text
--lambda-oracle-sign default 0.5
--lambda-oracle-top1 default 1.0
--lambda-oracle-regret default 0.5
--positive-margin default 0.0
--negative-margin default 0.0
--top1-temperature default 0.5
--safety-score reward|return|mean default reward
```

### Sign Loss

For each action:

```text
target_delta = oracle_delta_tc
score = reward_pred / coverage_scale
```

For positive oracle actions:

```text
loss_pos = softplus(-(score - positive_margin) / temp)
```

For negative oracle actions:

```text
loss_neg = softplus((score - negative_margin) / temp)
```

Combined:

```text
L_sign = mean(loss_pos_or_neg)
```

Purpose:

```text
Make negative oracle actions receive low scores instead of only relative ranks.
```

### Soft Top1 Safety Loss

Within a group:

```text
w_i = softmax(score_i / top1_temperature)
```

Penalize probability mass assigned to negative oracle actions:

```text
L_top1_neg = sum_i w_i * relu(-oracle_delta_tc_i)
```

Purpose:

```text
Directly reduce negative_top1_rate, the guardrail that rejected the candidate.
```

### Soft Regret Loss

Within a group:

```text
oracle_best = max_i oracle_delta_tc_i
expected_delta = sum_i w_i * oracle_delta_tc_i
L_regret = relu(oracle_best - expected_delta)
```

Purpose:

```text
Move probability mass toward high-real-delta actions, not just away from
negative actions.
```

### Total Loss

Current:

```text
L = value + rank
```

New:

```text
L = lambda_oracle_value  * L_value
  + lambda_oracle_rank   * L_rank
  + lambda_oracle_sign   * L_sign
  + lambda_oracle_top1   * L_top1_neg
  + lambda_oracle_regret * L_regret
```

Add to `history.tsv`:

```text
train_sign_loss
train_top1_loss
train_regret_loss
train_negative_actions
```

## Stage A Commands

Static:

```bash
python -m py_compile scripts/finetune_oracle_action_values.py
python scripts/finetune_oracle_action_values.py --help
```

Safety finetune:

```bash
python scripts/finetune_oracle_action_values.py \
  --checkpoint autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt \
  --oracle-actions autoresearch/oracle-action-probe-260629-labeled-subckt-train/oracle_actions.tsv \
  --out-dir autoresearch/oracle-action-value-finetune-260629-safety \
  --epochs 5 \
  --lr 1e-4 \
  --lambda-oracle-value 1.0 \
  --lambda-oracle-rank 1.0 \
  --lambda-oracle-sign 0.5 \
  --lambda-oracle-top1 1.0 \
  --lambda-oracle-regret 0.5 \
  --top1-temperature 0.5 \
  --train-heads reward,return \
  --plan-device cpu
```

Primary held-out subckt gate:

```bash
python scripts/evaluate_oracle_action_values.py \
  --oracle-actions autoresearch/oracle-action-probe-260629-labeled-subckt-val/oracle_actions.tsv \
  --checkpoints incumbent=autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt,candidate=autoresearch/oracle-action-value-finetune-260629-safety/candidate.pt \
  --bench-root /data4/pengqingsong/DFT/Dataset/atalanta_bist_lowtc_subckt_100k_labels/subcircuits \
  --score-fields reward_pred,guarded_reward,hard_reduction_total_pred,hybrid_pred \
  --top-ks 8,16,18 \
  --oracle-top-m 5 \
  --plan-device cpu \
  --out-dir autoresearch/oracle-action-value-gate-260629-safety-val \
  --baseline incumbent
```

Transfer gate:

```bash
python scripts/evaluate_oracle_action_values.py \
  --oracle-actions autoresearch/oracle-action-probe-260629-smoke/oracle_actions.tsv \
  --checkpoints incumbent=autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt,candidate=autoresearch/oracle-action-value-finetune-260629-safety/candidate.pt \
  --score-fields reward_pred,guarded_reward,hard_reduction_total_pred,hybrid_pred \
  --top-ks 8,16,32 \
  --oracle-top-m 5 \
  --plan-device cpu \
  --out-dir autoresearch/oracle-action-value-gate-260629-safety-transfer \
  --baseline incumbent
```

## Stage A Acceptance

Do not require full PROMOTE immediately.

Accept Stage A as progress if:

```text
held-out subckt negative_top1_rate decreases without top1_regret increasing
or transfer negative_top1_rate decreases without Spearman collapsing
```

Promote only if the existing gate says:

```text
candidate: PROMOTE
```

Reject if:

```text
negative_top1_rate increases on either primary gate or transfer gate
```

## Stage B: Dedicated Action-Value Head

Only if Stage A is inconclusive:

```text
Add action_value_head to TPIWorldModel.
```

Required changes:

```text
tpi_jepa/model.py
tpi_jepa/plan.py
scripts/evaluate_oracle_action_values.py
scripts/finetune_oracle_action_values.py
```

New score field:

```text
action_value_pred
```

Why not Stage A:

```text
Adding a new head requires updating planner score fields and checkpoint
compatibility. It is more invasive than adding safety losses to the existing
reward score.
```

Stage B acceptance:

```text
action_value_pred must beat incumbent hybrid_pred on primary gate without
negative_top1 regression.
```

## Expected Outcome

The expected useful result is not necessarily higher Spearman first.

The immediate target is:

```text
reduce negative_top1_rate and top1_regret
```

Because the current system already showed that Spearman can improve while
top1 safety gets worse. That is not acceptable for a planner.

## Next Command

```text
$autoresearch fix autoresearch/plan-260629-0944/plan.md
```
