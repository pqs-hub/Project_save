# AutoResearch Fix Report: Frozen Hard-Reduction Diagnostic

generated_at: `2026-06-29 10:53 Asia/Shanghai`

## Objective

Execute:

```text
autoresearch/plan-260629-1047/plan.md
```

Question:

```text
Did the previous full-circuit transfer regression come from updating
hard_reduction_head?
```

## Code Change

Modified:

```text
scripts/finetune_oracle_action_values.py
```

Added a new train scope:

```text
--train-scope planner_joint_frozen_hard
```

It trains:

```text
action_encoder
dynamics
reward_head
return_head
```

It freezes:

```text
hard_reduction_head
online_encoder
target_encoder
```

The ranking objective remains:

```text
pairwise logistic ranking loss on hybrid_pred
```

No new loss, head, candidate generation, or oracle labels were added.

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

### Smoke

Output:

```text
autoresearch/oracle-action-value-finetune-260629-joint-freezehard-smoke/
```

Configuration:

```text
epochs: 1
max_actions_per_group: 8
train_scope: planner_joint_frozen_hard
ranking_score_field: hybrid_pred
```

Result:

| epoch | train rank loss | pairs |
|---:|---:|---:|
| 1 | 0.723589 | 940 |

Smoke handoff confirmed trainable prefixes:

```text
action_encoder
dynamics
reward_head
return_head
```

## Full Training

Output:

```text
autoresearch/oracle-action-value-finetune-260629-joint-freezehard/
```

Training history:

| epoch | train rank loss | pairs |
|---:|---:|---:|
| 1 | 0.807767 | 5032 |
| 2 | 0.782869 | 5032 |
| 3 | 0.707618 | 5032 |
| 4 | 0.690929 | 5032 |
| 5 | 0.683221 | 5032 |

Comparison to previous joint-hybrid:

```text
previous joint_hybrid final rank loss: 0.638620
freezehard final rank loss:             0.683221
```

Interpretation:

```text
Freezing hard_reduction_head made the training objective harder to optimize.
```

## Held-Out Labeled-Subckt Gate

Output:

```text
autoresearch/oracle-action-value-gate-260629-joint-freezehard-val/
```

Hybrid comparison:

| checkpoint | hybrid Spearman | negative top1 | top1 regret |
|---|---:|---:|---:|
| incumbent | 0.005693 | 0.458333 | 0.043452 |
| joint_hybrid | 0.025509 | 0.458333 | 0.031754 |
| freezehard | -0.010977 | 0.416667 | 0.028037 |

Hard-reduction comparison:

| checkpoint | hard Spearman | negative top1 | top1 regret |
|---|---:|---:|---:|
| incumbent | 0.020105 | 0.458333 | 0.043452 |
| joint_hybrid | 0.068528 | 0.458333 | 0.031754 |
| freezehard | 0.015405 | 0.333333 | 0.026966 |

Interpretation:

```text
freezehard improves top1 safety/regret on held-out subckt, but loses ranking
correlation. It does not preserve the previous joint_hybrid held-out Spearman
gain.
```

## Full-Circuit Transfer Gate

Output:

```text
autoresearch/oracle-action-value-gate-260629-joint-freezehard-transfer/
```

Hybrid comparison:

| checkpoint | hybrid Spearman | negative top1 | top1 regret |
|---|---:|---:|---:|
| incumbent | 0.327398 | 0.166667 | 0.012552 |
| joint_hybrid | 0.256683 | 0.166667 | 0.012558 |
| freezehard | 0.182085 | 0.166667 | 0.017645 |

Hard-reduction comparison:

| checkpoint | hard Spearman | negative top1 | top1 regret |
|---|---:|---:|---:|
| incumbent | 0.324443 | 0.166667 | 0.012552 |
| joint_hybrid | 0.267081 | 0.166667 | 0.012558 |
| freezehard | 0.190038 | 0.166667 | 0.017645 |

Diagnostic success check:

| criterion | result |
|---|---|
| freezehard hybrid Spearman >= joint_hybrid + 0.03 | FAIL |
| freezehard negative_top1 <= joint_hybrid | PASS |

## Diagnosis

Freezing `hard_reduction_head` did not recover transfer.

More importantly:

```text
hard_reduction_total_pred also regressed while hard_reduction_head was frozen.
```

Therefore, the transfer damage is not mainly caused by changing hard-head
weights. It is caused by changing the upstream latent/action transition that
feeds the hard head:

```text
action_encoder
dynamics
```

This matches the caveat in the plan:

```text
freezing hard_reduction_head parameters does not freeze its input
representation.
```

## Verdict

Implementation:

```text
DONE
```

Diagnostic:

```text
ANSWERED
```

Candidate:

```text
DO NOT PROMOTE
```

Answer:

```text
The previous transfer regression is not primarily from updating
hard_reduction_head weights. It comes from latent/action-dynamics drift.
```

## Recommended Next Step

Do not keep training `action_encoder/dynamics` on the small oracle set.

The next diagnostic should be:

```text
train reward_head and return_head only, but rank hybrid_pred
```

That means:

```text
--train-scope heads
--train-heads reward,return
--ranking-score-field hybrid_pred
--lambda-oracle-value 0.0
--lambda-oracle-rank 1.0
```

Rationale:

```text
This keeps hard_reduction_total_pred and latent dynamics stable, while testing
whether reward/return can act as a residual correction to the already useful
hard score.
```

If that fails, the stronger conclusion is:

```text
the current 72 train oracle groups are insufficient; collect more oracle groups
from the 131 labeled subckt distribution before touching dynamics again.
```

