# AutoResearch Plan: Planner-Joint Hybrid Ranking With Frozen Hard Head

generated_at: `2026-06-29 10:47 Asia/Shanghai`

## Goal

Run a diagnostic experiment:

```text
planner_joint hybrid ranking with hard_reduction_head frozen
```

to test whether the previous full-circuit transfer regression came from
disturbing the hard-reduction signal.

## Background

Previous experiment:

```text
autoresearch/autoresearch-260629-1039/results.md
```

trained:

```text
action_encoder
dynamics
reward_head
return_head
hard_reduction_head
```

with pairwise ranking loss on:

```text
hybrid_pred = return_pred + reward_pred + hard_reduction_total_pred * coverage_scale
```

Result on held-out labeled subckt:

| checkpoint | hybrid Spearman | negative top1 | top1 regret |
|---|---:|---:|---:|
| incumbent | 0.005693 | 0.458333 | 0.043452 |
| candidate | 0.025509 | 0.458333 | 0.031754 |

Result on full-circuit transfer:

| checkpoint | hybrid Spearman | negative top1 | top1 regret |
|---|---:|---:|---:|
| incumbent | 0.327398 | 0.166667 | 0.012552 |
| candidate | 0.256683 | 0.166667 | 0.012558 |

Interpretation:

```text
The method learned something on held-out labeled subckts, but damaged transfer.
```

Since full-circuit transfer originally relied strongly on:

```text
hard_reduction_total_pred
hybrid_pred
```

the next diagnostic is to freeze `hard_reduction_head` and see whether transfer
recovers.

## Important Caveat

Freezing `hard_reduction_head` parameters does not guarantee
`hard_reduction_total_pred` is numerically unchanged.

Reason:

```text
action_encoder and dynamics still change z_pred, which is the input to
hard_reduction_head.
```

Therefore the gate must report both:

```text
hybrid_pred
hard_reduction_total_pred
```

If `hard_reduction_total_pred` still regresses while the head is frozen, the
disturbance is coming from shared latent/dynamics changes, not the hard head
weights.

## Scope

### In Scope

Modify:

```text
scripts/finetune_oracle_action_values.py
```

Add a new training scope:

```text
--train-scope planner_joint_frozen_hard
```

This scope should train:

```text
action_encoder
dynamics
reward_head
return_head
```

and freeze:

```text
hard_reduction_head
online_encoder
target_encoder
```

Keep the same loss:

```text
pairwise logistic ranking loss on hybrid_pred
```

Keep the same data and gates as the previous joint-hybrid experiment.

### Out Of Scope

Do not add new loss terms.

Do not add `action_value_head`.

Do not regenerate oracle labels.

Do not change candidate generation.

Do not change gate verdict logic.

## Implementation Steps

1. Edit `scripts/finetune_oracle_action_values.py`.
2. Extend CLI choices:

```text
--train-scope heads|planner_joint|planner_joint_frozen_hard
```

3. Extend `set_trainable_parts()`:

```text
planner_joint_frozen_hard:
  trainable prefixes:
    action_encoder
    dynamics
    reward_head
    return_head
  frozen prefixes:
    hard_reduction_head
    online_encoder
    target_encoder
```

4. Ensure `handoff.json` records:

```text
train_scope
trainable_prefixes
ranking_score_field
value_score_field
```

5. Run syntax check.
6. Run 1-epoch smoke with `--train-scope planner_joint_frozen_hard`.
7. Run full 5-epoch finetune.
8. Run held-out labeled-subckt gate.
9. Run full-circuit transfer gate.
10. Compare against both incumbent and previous joint-hybrid candidate.

## Commands

### Syntax Check

```bash
python -m py_compile scripts/finetune_oracle_action_values.py
```

### Smoke Finetune

```bash
python scripts/finetune_oracle_action_values.py \
  --checkpoint autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt \
  --oracle-actions autoresearch/oracle-action-probe-260629-labeled-subckt-train/oracle_actions.tsv \
  --out-dir autoresearch/oracle-action-value-finetune-260629-joint-freezehard-smoke \
  --epochs 1 \
  --lr 1e-5 \
  --lambda-oracle-value 0.0 \
  --lambda-oracle-rank 1.0 \
  --ranking-score-field hybrid_pred \
  --train-scope planner_joint_frozen_hard \
  --max-actions-per-group 8
```

### Full Finetune

```bash
python scripts/finetune_oracle_action_values.py \
  --checkpoint autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt \
  --oracle-actions autoresearch/oracle-action-probe-260629-labeled-subckt-train/oracle_actions.tsv \
  --out-dir autoresearch/oracle-action-value-finetune-260629-joint-freezehard \
  --epochs 5 \
  --lr 1e-5 \
  --lambda-oracle-value 0.0 \
  --lambda-oracle-rank 1.0 \
  --ranking-score-field hybrid_pred \
  --train-scope planner_joint_frozen_hard
```

Use the known valid incumbent checkpoint path:

```text
autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt
```

### Held-Out Labeled-Subckt Gate

```bash
python scripts/evaluate_oracle_action_values.py \
  --oracle-actions autoresearch/oracle-action-probe-260629-labeled-subckt-val/oracle_actions.tsv \
  --checkpoint incumbent=autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt \
  --checkpoint joint_hybrid=autoresearch/oracle-action-value-finetune-260629-joint-hybrid/candidate.pt \
  --checkpoint freezehard=autoresearch/oracle-action-value-finetune-260629-joint-freezehard/candidate.pt \
  --baseline incumbent \
  --score-fields reward_pred,guarded_reward,hard_reduction_total_pred,hybrid_pred \
  --out-dir autoresearch/oracle-action-value-gate-260629-joint-freezehard-val
```

### Full-Circuit Transfer Gate

```bash
python scripts/evaluate_oracle_action_values.py \
  --oracle-actions autoresearch/oracle-action-probe-260629-smoke/oracle_actions.tsv \
  --checkpoint incumbent=autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt \
  --checkpoint joint_hybrid=autoresearch/oracle-action-value-finetune-260629-joint-hybrid/candidate.pt \
  --checkpoint freezehard=autoresearch/oracle-action-value-finetune-260629-joint-freezehard/candidate.pt \
  --baseline incumbent \
  --score-fields reward_pred,guarded_reward,hard_reduction_total_pred,hybrid_pred \
  --out-dir autoresearch/oracle-action-value-gate-260629-joint-freezehard-transfer
```

## Primary Metrics

Primary comparison field:

```text
hybrid_pred
```

Diagnostic field:

```text
hard_reduction_total_pred
```

Metrics:

```text
mean_spearman
negative_top1_rate
mean_top1_regret
```

## Decision Table

| Outcome | Interpretation | Next action |
|---|---|---|
| held-out improves and transfer recovers | Updating hard head caused the previous transfer regression | Continue with frozen hard head and expand oracle data |
| held-out improves but transfer still regresses | Regression comes from action/dynamics latent shift or data mismatch | Try stricter scope or collect more oracle data |
| held-out no longer improves but transfer recovers | Hard head update was needed for in-distribution learning but unsafe for transfer | Need more data or separated action-value score |
| both held-out and transfer regress | Joint scope is too aggressive | Revert to narrower scope before adding losses |

## Acceptance Criteria

This is a diagnostic, not a promotion experiment.

Mark as successful diagnostic if it answers:

```text
Does freezing hard_reduction_head recover transfer hybrid_pred relative to the
previous joint_hybrid candidate?
```

Concrete transfer recovery threshold:

```text
freezehard transfer hybrid Spearman >= joint_hybrid transfer hybrid Spearman + 0.03
```

and:

```text
freezehard transfer negative_top1_rate <= joint_hybrid transfer negative_top1_rate
```

Candidate promotion remains stricter:

```text
held-out hybrid Spearman >= incumbent + 0.05
held-out negative_top1_rate <= incumbent
held-out top1_regret <= incumbent
transfer hybrid Spearman not materially below incumbent
```

## Next Step

Run:

```text
$autoresearch fix autoresearch/plan-260629-1047/plan.md
```
