# AutoResearch Results: Joint Hybrid Oracle Ranking

generated_at: `2026-06-29 10:39 Asia/Shanghai`

## Objective

Execute:

```text
autoresearch/plan-260629-1017/plan.md
```

Goal:

```text
Joint-train action_encoder/dynamics/reward/hard_reduction on oracle pairwise
planner-score ranking instead of head-only reward finetune.
```

## Code Changes

Modified:

```text
scripts/finetune_oracle_action_values.py
```

Added:

```text
--train-scope heads|planner_joint
--ranking-score-field
--value-score-field
```

New `planner_joint` trainable scope:

```text
action_encoder
dynamics
reward_head
return_head
hard_reduction_head
```

Frozen in `planner_joint`:

```text
online_encoder
target_encoder
```

`predict_group_scores()` now returns differentiable tensor scores:

```text
reward_pred
return_pred
guarded_reward
hard_reduction_total_pred
hybrid_pred
```

The training rank loss can now optimize:

```text
hybrid_pred = return_pred + reward_pred + hard_reduction_total_pred * coverage_scale
```

instead of being hardcoded to `reward_pred`.

## Verification

### Syntax Check

Command:

```bash
python -m py_compile scripts/finetune_oracle_action_values.py
```

Result:

```text
PASS
```

### Planner-Joint Smoke

Output:

```text
autoresearch/oracle-action-value-finetune-260629-joint-smoke/
```

Command properties:

```text
epochs: 1
max_actions_per_group: 8
train_scope: planner_joint
ranking_score_field: hybrid_pred
```

Result:

| epoch | train rank loss | pairs |
|---:|---:|---:|
| 1 | 0.713322 | 940 |

Status:

```text
PASS
```

### Backward-Compatible Heads Smoke

Output:

```text
autoresearch/oracle-action-value-finetune-260629-heads-compat-smoke/
```

Command properties:

```text
default train_scope: heads
default ranking_score_field: reward_pred
```

Result:

| epoch | train loss | rank loss | value loss | pairs |
|---:|---:|---:|---:|---:|
| 1 | 0.745121 | 0.472368 | 0.508937 | 214 |

Status:

```text
PASS
```

## Full Joint-Hybrid Training

Output:

```text
autoresearch/oracle-action-value-finetune-260629-joint-hybrid/
```

Configuration:

```text
epochs: 5
lr: 1e-5
lambda_oracle_value: 0.0
lambda_oracle_rank: 1.0
ranking_score_field: hybrid_pred
train_scope: planner_joint
```

Training history:

| epoch | train rank loss | pairs |
|---:|---:|---:|
| 1 | 0.796627 | 5032 |
| 2 | 0.733236 | 5032 |
| 3 | 0.686007 | 5032 |
| 4 | 0.658040 | 5032 |
| 5 | 0.638620 | 5032 |

Interpretation:

```text
The joint hybrid ranking objective is trainable and decreases materially.
This is stronger than the previous head-only pairwise run, whose loss barely
moved.
```

## Held-Out Labeled-Subckt Gate

Output:

```text
autoresearch/oracle-action-value-gate-260629-joint-hybrid-val/
```

Primary score field:

```text
hybrid_pred
```

Comparison:

| checkpoint | hybrid Spearman | negative top1 | top1 regret |
|---|---:|---:|---:|
| incumbent | 0.005693 | 0.458333 | 0.043452 |
| candidate | 0.025509 | 0.458333 | 0.031754 |

Acceptance check:

| criterion | result |
|---|---|
| Spearman >= incumbent + 0.05 | FAIL |
| negative_top1_rate <= incumbent | PASS |
| top1_regret <= incumbent | PASS |

Verdict:

```text
INCONCLUSIVE
```

Interpretation:

```text
The candidate improves held-out hybrid ranking slightly and improves top1
regret, but not enough to promote.
```

## Full-Circuit Transfer Gate

Output:

```text
autoresearch/oracle-action-value-gate-260629-joint-hybrid-transfer/
```

Primary score field:

```text
hybrid_pred
```

Comparison:

| checkpoint | hybrid Spearman | negative top1 | top1 regret |
|---|---:|---:|---:|
| incumbent | 0.327398 | 0.166667 | 0.012552 |
| candidate | 0.256683 | 0.166667 | 0.012558 |

Interpretation:

```text
The candidate regresses on full-circuit transfer. This blocks promotion even
though held-out subckt behavior slightly improved.
```

## Final Verdict

Implementation status:

```text
DONE
```

Checkpoint status:

```text
DO NOT PROMOTE
```

Reason:

```text
Joint hybrid training is a better optimization path than head-only pairwise,
but this 5-epoch candidate does not meet the held-out Spearman threshold and
regresses on transfer.
```

## What This Means

This result supports two conclusions:

1. Training the actual planner score is more effective than only training
   `reward_pred`.
2. The current oracle training set is probably too small or too narrow for
   stable transfer once `action_encoder/dynamics` are unfrozen.

The next experiment should not be another head-only loss tweak.

Recommended next branch:

```text
Increase oracle action groups from the 131 labeled subckt distribution, then
repeat joint-hybrid ranking with the same fixed gate.
```

Alternative diagnostic:

```text
Run planner_joint with hard_reduction_head frozen to test whether transfer
regression comes from disturbing the hard-reduction signal.
```

