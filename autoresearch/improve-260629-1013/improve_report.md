# AutoResearch Improve Report: Oracle Ranking Alignment

generated_at: `2026-06-29 10:13 Asia/Shanghai`

## Problem

The actual goal is not to add another prediction head.

The goal is:

```text
For actions in the same circuit state, the model score should rank actions in
the same order as backend-measured true delta_tc.
```

The current experiments show that this is not solved yet.

## Current Evidence

### Fixed Oracle Gate Exists

The toolchain can now:

```text
1. generate action groups
2. label each action with backend true delta_tc
3. save the fixed oracle_actions.tsv
4. rescore the exact same actions with any checkpoint
5. compare ranking/safety metrics without backend rerun randomness
```

Primary gate script:

```text
scripts/evaluate_oracle_action_values.py
```

Primary oracle data:

```text
train: autoresearch/oracle-action-probe-260629-labeled-subckt-train/oracle_actions.tsv
val:   autoresearch/oracle-action-probe-260629-labeled-subckt-val/oracle_actions.tsv
```

### Correct Training Distribution Was Used

The corrected oracle experiment uses labeled sampled subcircuits, not the
original large circuits.

Source distribution:

```text
/data4/pengqingsong/DFT/Dataset/atalanta_bist_lowtc_subckt_100k_labels/labels.csv
```

Split:

```text
24 train labeled subckts
8 held-out labeled subckts
```

Oracle labels:

```text
train: 1296 actions, 72 groups
val:    432 actions, 24 groups
```

### Head-Only Value+Rank Finetune Failed

Run:

```text
autoresearch/oracle-action-value-finetune-260629-labeled-subckt/
```

Held-out subckt gate:

```text
autoresearch/oracle-action-value-gate-260629-labeled-subckt-val/
```

Key result:

| checkpoint | score_field | Spearman | negative top1 | top1 regret |
|---|---|---:|---:|---:|
| incumbent | hybrid_pred | 0.005693 | 0.458333 | 0.043452 |
| candidate | hybrid_pred | 0.004533 | 0.416667 | 0.043300 |
| incumbent | reward_pred | -0.137870 | 0.416667 | 0.023290 |
| candidate | reward_pred | -0.205044 | 0.541667 | 0.031650 |

Transfer gate:

```text
autoresearch/oracle-action-value-gate-260629-labeled-subckt-transfer/
```

Key result:

| checkpoint | score_field | Spearman | negative top1 | top1 regret |
|---|---|---:|---:|---:|
| incumbent | hybrid_pred | 0.327398 | 0.166667 | 0.012552 |
| candidate | hybrid_pred | 0.325323 | 0.166667 | 0.012557 |
| incumbent | reward_pred | 0.294742 | 0.500000 | 0.020223 |
| candidate | reward_pred | 0.348502 | 0.333333 | 0.022373 |

Interpretation:

```text
reward_pred Spearman can move, but this does not reliably improve the actual
planner score or top1 behavior.
```

### Pairwise-Only Loss Also Failed

Run:

```text
autoresearch/oracle-action-value-finetune-260629-pairwise-only/
```

Training loss:

| epoch | rank loss | pairs |
|---:|---:|---:|
| 1 | 0.587324 | 5032 |
| 2 | 0.587263 | 5032 |
| 3 | 0.586839 | 5032 |
| 4 | 0.586169 | 5032 |
| 5 | 0.585994 | 5032 |

Held-out subckt gate:

```text
autoresearch/oracle-action-value-gate-260629-pairwise-only-val/
```

| checkpoint | score_field | Spearman | negative top1 | top1 regret |
|---|---|---:|---:|---:|
| incumbent | reward_pred | -0.137870 | 0.416667 | 0.023290 |
| candidate | reward_pred | -0.160885 | 0.541667 | 0.035315 |
| incumbent | hybrid_pred | 0.005693 | 0.458333 | 0.043452 |
| candidate | hybrid_pred | -0.001395 | 0.458333 | 0.036230 |

Transfer gate:

```text
autoresearch/oracle-action-value-gate-260629-pairwise-only-transfer/
```

| checkpoint | score_field | Spearman | negative top1 | top1 regret |
|---|---|---:|---:|---:|
| incumbent | reward_pred | 0.294742 | 0.500000 | 0.020223 |
| candidate | reward_pred | 0.280641 | 0.500000 | 0.026983 |
| incumbent | hybrid_pred | 0.327398 | 0.166667 | 0.012552 |
| candidate | hybrid_pred | 0.323155 | 0.166667 | 0.012552 |

Interpretation:

```text
The clean pairwise objective is conceptually right, but applying it only to the
existing reward/return heads is not enough.
```

## Diagnosis

### Failure Mode 1: Wrong Trainable Scope

The current finetune mostly changes:

```text
reward_head
return_head
```

But the planner score also depends on features produced by:

```text
action_encoder
dynamics
hard_reduction_head
context/state representation
```

If these remain frozen, the rank loss can only reshape the last scalar heads.
That is too weak if the latent action/state representation does not already
separate high-delta_tc and low-delta_tc actions.

### Failure Mode 2: Wrong Score Carrier

The gate evaluates multiple fields:

```text
reward_pred
guarded_reward
hard_reduction_total_pred
hybrid_pred
```

But the actual planner usually relies on the safer combined behavior, not
`reward_pred` alone.

The current finetune optimizes reward/return fields, while the best historical
signal often comes from:

```text
hybrid_pred
hard_reduction_total_pred
```

Therefore, improving `reward_pred` alone can fail to improve the real decision
policy.

### Failure Mode 3: Adding More Losses May Hide the Main Problem

A sign/top1 safety loss can reduce bad top1 picks, but it does not solve the
central issue if the optimized score is not the score used by the planner.

So the next experiment should not be:

```text
add four losses at once
```

It should be:

```text
train the actual planner score to rank oracle delta_tc correctly, with a deeper
trainable scope.
```

## Recommended Next Improvement

### P0: Joint Oracle Ranking Finetune On Planner Score

Implement a new training mode in `scripts/finetune_oracle_action_values.py`.

Trainable modules:

```text
action_encoder
dynamics
reward_head
return_head
hard_reduction_head
```

Keep frozen at first:

```text
online_encoder
target_encoder
```

Reason:

```text
This allows action/state transition features and score heads to adapt, while
avoiding full encoder drift on only 1296 oracle actions.
```

Primary loss:

```text
pairwise logistic ranking loss on planner_score
```

where `planner_score` should initially be:

```text
hybrid_pred
```

or a differentiable equivalent matching the planner's current action ordering.

Do not optimize only `reward_pred`.

### P0 Acceptance Gate

Primary held-out gate:

```text
autoresearch/oracle-action-probe-260629-labeled-subckt-val/oracle_actions.tsv
```

Secondary transfer gate:

```text
autoresearch/oracle-action-probe-260629-smoke/oracle_actions.tsv
```

Promote only if held-out subckt improves:

```text
hybrid_pred Spearman >= incumbent + 0.05
negative_top1_rate <= incumbent
top1_regret <= incumbent
```

Reject if:

```text
negative_top1_rate worsens by any visible amount on held-out subckt
```

Transfer full-circuit should be reported but should not override the
in-distribution held-out gate unless it catastrophically regresses.

### P1: Add Safety Loss Only If Needed

If joint planner-score ranking improves Spearman but still picks negative
top1 actions, add one safety term:

```text
top1 negative margin loss
```

Do not add sign loss, top1 loss, regret loss, and value loss together in the
same first experiment. That makes attribution unclear.

### P1: Dedicated action_value_head Only If Planner-Score Training Is Messy

A dedicated head is useful only if we decide the planner needs a separate
oracle-value score field:

```text
action_value_pred
```

It is not required by the objective itself. The objective is ranking alignment.

So the dedicated head should be delayed until either:

```text
1. hybrid_pred is too non-differentiable or awkward to train cleanly
2. changing reward/hard heads damages existing planner behavior
3. the project needs oracle value and historical reward to remain separate
```

## Proposed Next Command

```text
$autoresearch plan Joint-train action_encoder/dynamics/reward/hard_reduction on oracle pairwise planner-score ranking instead of head-only reward finetune.
```

## Expected Implementation Tasks

1. Add `--train-scope planner_joint` to `scripts/finetune_oracle_action_values.py`.
2. Add `--ranking-score-field hybrid_pred` or equivalent differentiable planner score.
3. Compute pairwise logistic loss on that score inside each oracle action group.
4. Keep `online_encoder` frozen for the first experiment.
5. Train from the incumbent checkpoint on labeled-subckt train oracle data.
6. Gate on labeled-subckt val oracle actions.
7. Gate on full-circuit smoke oracle actions for transfer risk.

## Bottom Line

The next useful improvement is not "add a head" and not "add many losses".

The next useful improvement is:

```text
Make the trainable part of the model and the optimized score match the actual
planner decision: rank candidate actions by true backend delta_tc.
```

