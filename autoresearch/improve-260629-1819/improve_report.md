# AutoResearch Improve Report: Stop Proxy Chasing, Stabilize Ranking Pipeline

generated_at: `2026-06-29 18:19 Asia/Shanghai`

## Executive Decision

The next improvement should not be another SCOAP / delta-SCOAP tweak.

The highest-value next step is:

```text
Build a low-risk action-value scorer on top of fixed checkpoint predictions,
then only return to full retraining after the incumbent training recipe is
reproducible.
```

Reason:

```text
The current scratch retrain setup does not reproduce the incumbent. Even the
oracle-rank weight 0.0 control badly regresses hard F1. Therefore scratch
oracle-ranking sweeps are confounded.
```

In simpler terms:

```text
Before teaching the model a new skill, first make sure the normal training
recipe can reproduce the old strong model. Right now it cannot.
```

## Current Evidence

### 1. Oracle gate exists and is now the right judge

The framework can now compare checkpoints on the same backend-labeled action groups:

```text
scripts/evaluate_oracle_action_values.py
```

Useful gates:

```text
expanded oracle val: autoresearch/oracle-action-probe-260629-expanded-subckt-val/oracle_actions.tsv
transfer oracle:     autoresearch/oracle-action-probe-260629-smoke/oracle_actions.tsv
```

This is the correct direction because the real target is:

```text
Within the same circuit state, put higher true delta_tc actions earlier.
```

### 2. Scratch oracle-ranking sweep is not trustworthy yet

Source:

```text
autoresearch/autoresearch-260629-1450/final_report.md
```

Key result:

| checkpoint | expanded Spearman | transfer Spearman | hard F1 | predictive score | verdict |
|---|---:|---:|---:|---:|---|
| incumbent | 0.031476 | 0.327398 | 0.794805 | 0.820147 | baseline |
| scratch_0p00 | 0.008166 | -0.059028 | 0.165301 | 0.486414 | reject |
| scratch_0p05 | -0.072460 | -0.095552 | 0.213846 | 0.449052 | reject |
| scratch_0p10 | -0.080071 | 0.099302 | 0.160915 | 0.497586 | reject |
| scratch_0p20 | 0.013980 | 0.402667 | 0.187658 | 0.499720 | reject |

Important interpretation:

```text
scratch_0p00 has no oracle ranking loss, but it still collapses hard F1.
So the scratch training recipe itself is not equivalent to the incumbent recipe.
```

Therefore:

```text
Do not conclude "oracle ranking loss is bad" from this sweep.
Conclude "this scratch training setup is not a valid base for ablation."
```

### 3. Scheme A and Scheme B simplify losses but do not solve planner ranking

Scheme A:

```text
keep hard_reduction head
disable hard_count / FC / return
```

Result:

```text
hard F1 improves, but oracle ranking is bad.
```

Scheme B:

```text
only train node hard labels
derive hard_count and hard_reduction from node logits
```

Result:

```text
node hard prediction is okay, but derived hard_reduction is not reliable enough
as planner score.
```

Practical conclusion:

```text
hard_count loss is removable or reducible.
direct hard_reduction is still useful.
derived hard_count/reduction alone is not enough.
```

### 4. SCOAP and delta-SCOAP are redundant/conflicting for this goal

Scheme B ablation:

| variant | expanded Spearman | transfer Spearman | conclusion |
|---|---:|---:|---|
| B_base | -0.143447 | -0.007391 | weak |
| B_only_scoap | 0.094394 | -0.176001 | helps expanded, hurts transfer |
| B_only_delta_scoap | 0.016842 | 0.134591 | helps transfer, modest expanded |

Scheme A ablation:

| variant | expanded Spearman | expanded bad top1 | transfer Spearman | transfer bad top1 | conclusion |
|---|---:|---:|---:|---:|---|
| A_base | -0.040485 | 56.25% | -0.084822 | 50.00% | reference |
| A_only_scoap | 0.078398 | 27.08% | -0.012033 | 50.00% | mixed |
| A_only_delta_scoap | -0.104651 | 54.17% | -0.002604 | 50.00% | mixed |

Interpretation:

```text
SCOAP and delta-SCOAP are useful proxy tasks, but changing their weights alone
does not give a safe planner.
```

### 5. Head-only and local oracle finetunes are too weak

Already tried:

```text
reward_head / return_head finetune
pairwise-only reward ranking
planner_joint hybrid ranking
planner_joint with hard_reduction frozen
heads_hybrid
bounded residual hybrid
```

Observed pattern:

```text
Some metrics move, but no candidate consistently improves held-out oracle
ranking and transfer safety.
```

Main reason:

```text
Small oracle data can overfit or drift the score, while fixed heads are too weak
to repair the representation.
```

## Product Requirement: P0 Fixed-Checkpoint Action-Value Scorer

### Goal

Learn a small scorer that re-ranks candidate actions using existing model outputs,
without changing the checkpoint.

This avoids damaging:

```text
hard F1
latent prediction
hard_reduction head
existing planner behavior
```

The scorer should answer:

```text
Given action candidates in the same state, which action should be ranked first
by real backend delta_tc?
```

### Why this is now the best next step

Full retraining is currently confounded.

A fixed-checkpoint scorer has three advantages:

```text
1. It cannot hurt hard F1, because the neural checkpoint is unchanged.
2. It directly optimizes action ranking on oracle groups.
3. It tells us whether existing predictions contain enough information to rank
   true TC gain after a better combiner.
```

If this scorer fails, then the model representation itself lacks the needed
signal and we should collect more oracle data or change architecture.

If it succeeds, the framework gets an immediate planner improvement and a clean
target for later distillation into the neural model.

### Input features

Start with features already exported by oracle rescoring:

```text
reward_pred
return_pred
guarded_reward
hard_reduction_total_pred
hard_reduction_sa0_pred
hard_reduction_sa1_pred
hybrid_pred
derived_hard_reduction_total_pred
action_type one-hot
score rank within group
score mean/std normalized within group
```

Optional metadata if available:

```text
benchmark_id
node_count bucket
base_tc
candidate_strategy
```

Do not use backend `oracle_delta_tc` as a feature.

### First implementation

Use a simple linear or tiny MLP scorer:

```text
score = scorer(features)
loss = pairwise logistic ranking loss inside each oracle group
```

Do not start with a large model.

Recommended variants:

| variant | trainable model | reason |
|---|---|---|
| linear_ranker | linear weights | most interpretable |
| mlp_ranker | 2-layer small MLP | tests simple nonlinear interactions |
| per_action_linear | separate bias/scale by control0/control1/observe | tests action-type calibration |

### Train data

Use fixed oracle action groups:

```text
train: expanded subckt train oracle groups
val:   expanded subckt val oracle groups
test:  smoke transfer oracle groups
```

Do not sample new candidates during training.

### Promotion gate

Primary gate:

```text
expanded oracle val
```

Secondary gate:

```text
transfer oracle
```

Promote only if, against incumbent `hybrid_pred`:

```text
expanded Spearman >= incumbent + 0.05
expanded negative_top1_rate <= incumbent
expanded top1_regret <= incumbent
transfer negative_top1_rate <= incumbent + 0.10
```

Reject immediately if:

```text
expanded negative_top1_rate worsens
transfer top1 real delta_tc becomes negative when incumbent is positive
```

### Output files

New script:

```text
scripts/train_action_value_ranker.py
```

Expected outputs:

```text
ranker.pt
ranker_weights.tsv
val_metrics.tsv
transfer_metrics.tsv
ranker_report.md
handoff.json
```

### Why this is not "giving up" on world-model training

This is a diagnostic layer.

If the fixed-checkpoint scorer improves ranking, then:

```text
the neural model already contains useful signals, but hybrid_pred combines them
poorly.
```

If it does not improve ranking, then:

```text
the neural model does not expose enough action-value signal, and retraining or
more oracle data is necessary.
```

Both outcomes are useful.

## Product Requirement: P0 Incumbent Reproduction Gate

### Goal

Make scratch retraining meaningful again.

Before any future scratch oracle-loss sweep, reproduce incumbent-like quality
with `lambda_oracle_rank=0.0`.

### Acceptance

A scratch control is allowed as a base only if:

```text
hard_macro_f1_tuned >= incumbent - 0.03
predictive_score >= incumbent - 0.03
transfer hybrid Spearman >= incumbent - 0.05
transfer negative_top1_rate <= incumbent + 0.10
```

Current scratch control fails this badly:

```text
hard F1 0.165301 vs incumbent 0.794805
```

### Required checks

Compare incumbent and scratch control on:

```text
dataset path
train/val split
seed
epochs
checkpoint selection rule
hard loss type
positive weight
negative mining
state update mode
candidate/action sampling
label file version
device-specific nondeterminism
```

Do not run more oracle-loss scratch sweeps until this gate passes.

## Product Requirement: P1 Oracle Data Audit

### Goal

Check whether oracle train/val/transfer groups cover the hard cases needed for
ranking.

Add a report over each oracle TSV:

```text
number of groups
actions per group
positive / zero / negative delta_tc count
delta_tc range
best action gain per group
worst action loss per group
action_type distribution
candidate_strategy distribution
benchmark size bucket
```

Why:

```text
Several experiments move expanded validation and transfer in opposite
directions. This may be a data-distribution mismatch, not only a loss problem.
```

Acceptance:

```text
Each train/val split should contain enough negative or low-gain actions to teach
top1 safety. A split with almost all positive actions is not enough.
```

## Ranked Backlog

| priority | item | decision |
|---|---|---|
| P0 | fixed-checkpoint action-value ranker | do next |
| P0 | incumbent reproduction gate | do before more scratch retraining |
| P1 | oracle data audit by sign/action/size | do before collecting more data |
| P1 | collect more oracle groups with balanced negatives | only after audit |
| P1 | distill ranker into neural model | only if fixed ranker works |
| P2 | new action_value_head | delay until fixed ranker shows signal |
| P2 | more SCOAP/delta-SCOAP weight sweeps | stop for now |
| P2 | Scheme B-only derived planner | stop for now |

## Recommended Next Command

```text
$autoresearch plan Train a fixed-checkpoint action-value ranker over oracle-rescored model features, with pairwise ranking loss and expanded/transfer oracle gates.
```

## Bottom Line

The project now has enough evidence to stop treating proxy prediction quality as
the main target.

The next useful improvement is a direct action ranking layer that:

```text
uses fixed model predictions,
does not damage the checkpoint,
is judged only by backend-labeled oracle action ordering.
```

