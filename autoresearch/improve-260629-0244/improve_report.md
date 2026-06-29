# AutoResearch Improve Report: Action-Value Alignment for TPI-JEPA

generated_at: `2026-06-29 02:44 Asia/Shanghai`

## Executive Decision

The next framework improvement should be:

```text
Build a small but real backend-labeled action-value dataset, then train/evaluate
the world model against oracle delta_test_coverage ranking and sign calibration.
```

Do not spend the next iteration only on:

```text
historical-action recall
larger subcircuit training
more candidate heuristics
node-level hard-fault F1
```

Those may still help later, but the current failure is more direct: the planner
needs action-value ordering, while the model is trained mostly through proxy
targets.

## Evidence

### Big eval-circuit oracle probe

Source:

```text
autoresearch/evals-260629-0102/evals_report.md
```

Data:

```text
288 finite oracle actions
190 positive delta_tc
98 negative delta_tc
```

Best rank signal:

```text
hybrid_pred mean Spearman = 0.3274
negative top1 rate       = 0.1667
mean sign accuracy       = 0.1979
```

Interpretation:

```text
There is some relative ranking signal, but calibration/sign is unsafe.
The model can rank some groups weakly while still assigning high scores to
negative real-TC actions.
```

Concrete failure examples from `i2c_aig`:

```text
N158::control0  delta_tc=-0.03687  hybrid_pred=7.2699
N332::control0  delta_tc=-0.02968  hybrid_pred=8.1197
N295::control0  delta_tc=-0.02842  hybrid_pred=8.1162
```

### Small training-distribution subckt probe

Source:

```text
autoresearch/evals-260629-smallckt/smallckt_evals_report.md
```

Data:

```text
288 finite oracle actions
268 positive delta_tc
20 negative delta_tc
```

Best small-subckt score:

```text
hard_reduction_total_pred mean Spearman = 0.0748
negative top1 rate                    = 0.0833
mean sign accuracy                    = 0.9306
```

Interpretation:

```text
The simple "training subcircuits are too small" hypothesis is not supported.
Small circuits have easier sign distribution, but ranking high-gain actions is
still weak.
```

## User / ICP Challenge

The effective user is a DFT researcher trying to use a learned planner to
improve real test coverage under limited TP budget.

They do not primarily need:

```text
a model that recalls logged actions
a model that predicts hard-fault nodes in isolation
a candidate generator that contains many plausible actions
```

They need:

```text
a scorer that ranks backend-beneficial TP actions above harmful or low-gain TP
actions, with enough calibration to avoid negative-TC top choices.
```

## Root Cause Hypothesis

Current training/evaluation has an objective mismatch:

```text
training labels: logged action transitions, hard fault reductions, reward proxies
deployment target: choose the action with best backend-measured delta_test_coverage
```

The model architecture already exposes action-conditioned outputs:

```text
reward_pred
return_pred
hard_reduction_pred
```

The planner already scores candidate actions through:

```text
tpi_jepa.plan.score_candidate_from_latent()
```

The missing asset is not another head first. The missing asset is:

```text
oracle-labeled candidate groups: same circuit state, many valid candidate
actions, each labeled by real backend delta_tc.
```

Without grouped oracle labels, ranking loss and top-K acceptance cannot be
meaningfully optimized.

## PRD-A: Oracle Action-Value Dataset Builder

Priority: `P0`

### Goal

Create a reusable dataset of backend-evaluated candidate actions:

```text
(benchmark, state_id, candidate_action) -> oracle_delta_tc
```

Each state must contain multiple candidates so ranking metrics/losses are
well-defined.

### Scope

Build on the existing script:

```text
scripts/oracle_action_value_probe.py
```

Required additions:

```text
1. stable manifest format for oracle action groups
2. option to append/resume without re-evaluating existing actions
3. export format that can be consumed by training/eval
4. support multiple states, not only initial state
5. explicit split: train_oracle / val_oracle / heldout_oracle
```

Minimum dataset target:

```text
train_oracle:   8-16 subcircuits x 2 states x 24 actions
val_oracle:     4 subcircuits x 2 states x 24 actions
heldout_oracle: b15_C,i2c_aig initial x 48 actions
```

This is intentionally small. The goal is to establish the signal before paying
for full 8-circuit oracle labeling.

### Label Definition

Primary label:

```text
oracle_delta_tc = backend TC after S + [a] - backend TC after S
```

Secondary labels:

```text
oracle_delta_fault_coverage
oracle_delta_hard_fault_count
oracle_delta_undetected_fault_count
```

State key:

```text
benchmark_id + sorted pre_actions
```

Action key:

```text
node::action_type
```

### Acceptance

Dataset builder is acceptable when:

```text
python scripts/oracle_action_value_probe.py --dry-run ...
python scripts/oracle_action_value_probe.py --resume ...
python -m py_compile scripts/oracle_action_value_probe.py
```

and output contains:

```text
oracle_actions.tsv
oracle_groups.tsv
manifest.json
handoff.json
```

with at least:

```text
>= 24 finite oracle actions per group
>= 2 groups per train benchmark
no duplicate action_key within a group
base_tc recorded for each state
```

## PRD-B: Action-Value Evaluation Gate

Priority: `P0`

### Goal

Turn oracle action-value data into a promotion gate for checkpoints and planner
score fields.

### Metrics

Per `(dataset, benchmark, state_id, score_field)`:

```text
spearman(pred_score, oracle_delta_tc)
kendall(pred_score, oracle_delta_tc)
ndcg@8
top1_real_delta_tc
top1_regret
negative_top1
sign_accuracy
calibration_slope
calibration_intercept
```

Aggregate promotion gate:

```text
mean_spearman >= incumbent + 0.10
negative_top1_rate <= incumbent - 0.10
mean_top1_regret <= incumbent - 0.01
```

Guardrail:

```text
Do not promote a model that improves Spearman but increases negative_top1_rate.
```

### Scope

Build on:

```text
scripts/oracle_action_value_probe.py
scripts/evaluate_hard_checkpoints.py
tpi_jepa.plan.score_candidate_from_latent()
```

The gate should compare at least these score fields:

```text
reward_pred
guarded_reward
hard_reduction_total_pred
hybrid_pred
```

### Acceptance

The gate is acceptable when it can produce one TSV and one markdown summary:

```text
oracle_action_value_metrics.tsv
oracle_action_value_report.md
```

for two checkpoints:

```text
incumbent checkpoint
candidate checkpoint
```

and reports:

```text
PROMOTE / REJECT / INCONCLUSIVE
```

## PRD-C: Action-Value Training Loss

Priority: `P0 after PRD-A/B`

### Goal

Train the world model to score candidate actions by real backend gain, not only
by proxy hard-fault reduction.

### Minimal Design

Use grouped oracle labels:

```text
group = fixed benchmark + state
items = candidate actions in that state
target = oracle_delta_tc
score = selected model field or new action_value_head
```

Start with two losses:

```text
pointwise: SmoothL1(pred_delta_tc, oracle_delta_tc_scaled)
pairwise: margin/logistic loss over action pairs in same group
```

Recommended first formula:

```text
loss = existing_loss
     + lambda_oracle_value * smooth_l1(action_value_pred, oracle_delta_tc_scaled)
     + lambda_oracle_rank  * pairwise_logistic(action_value_pred_i - action_value_pred_j,
                                               oracle_delta_tc_i - oracle_delta_tc_j)
```

Initial weights:

```text
lambda_oracle_value = 0.1
lambda_oracle_rank  = 0.1
```

Scale:

```text
oracle_delta_tc_scaled = 100.0 * oracle_delta_tc
```

This matches the existing reward scale convention.

### Important Constraint

Do not train only on positive oracle actions.

The `i2c_aig` failure shows negative action examples are essential. The dataset
must preserve negative and low-gain actions, otherwise the model will keep
selecting unsafe top-1 actions.

### Acceptance

On held-out oracle groups:

```text
mean Spearman improves by >= 0.10 over incumbent
negative_top1_rate decreases by >= 0.10
mean top1_regret decreases by >= 0.01
```

and node-level guardrails hold:

```text
hard_f1 drop <= 0.03
predictive score drop <= 0.03
```

## PRD-D: Scale-Aware Generalization Check

Priority: `P1`

### Goal

Keep the size-mismatch hypothesis alive as a diagnostic, but do not treat it as
the primary fix.

### Evidence So Far

Small subcircuits did not improve ranking alignment:

```text
small best mean Spearman = 0.0748
big best mean Spearman   = 0.3274
```

But small circuits had far fewer negative actions:

```text
small negative actions = 20 / 288
big negative actions   = 98 / 288
```

So scale/distribution may affect action sign distribution, but it does not
explain the whole ranking problem.

### Scope

Add benchmark buckets to oracle reports:

```text
node_count
gate_count
depth
initial_tc
positive_action_rate
negative_action_rate
```

### Acceptance

A report can answer:

```text
Does action-value alignment degrade with size/depth/initial TC?
Does negative_top1_rate correlate with negative_action_rate?
```

## Ranked Improvement Backlog

| Priority | Improvement | Why | Effort | Risk |
|---|---|---|---:|---:|
| P0 | Oracle action-value dataset builder | Enables correct labels | M | M |
| P0 | Oracle action-value eval gate | Prevents proxy-metric regressions | S | L |
| P0 | Pairwise/listwise action-value training | Directly optimizes planner ranking | M | M |
| P1 | Negative-action curriculum | Fixes unsafe top-1 selection | M | M |
| P1 | Scale-aware oracle diagnostics | Tests size/distribution shift rigorously | S | L |
| P1 | Planner safety filter by calibrated sign | Cheaply suppresses harmful actions | S | M |
| P2 | Larger-subckt retraining | Only useful after proving scale effect | L | M |
| P2 | New GNN/global architecture | Expensive before objective alignment | L | H |

## Immediate Next Plan

Recommended next `$autoresearch plan` target:

```text
Implement oracle action-value dataset export and checkpoint comparison gate.
```

Concrete first implementation slice:

```text
1. extend scripts/oracle_action_value_probe.py with --resume and manifest.json
2. emit oracle_groups.tsv with per-state base_tc and action counts
3. add scripts/evaluate_oracle_action_values.py to score existing oracle TSVs
4. compare incumbent checkpoint vs candidate checkpoint on the same oracle TSV
5. report PROMOTE/REJECT/INCONCLUSIVE
```

Recommended verify:

```bash
python -m py_compile scripts/oracle_action_value_probe.py scripts/evaluate_oracle_action_values.py

python scripts/evaluate_oracle_action_values.py \
  --oracle-actions autoresearch/oracle-action-probe-260629-smoke/oracle_actions.tsv \
  --checkpoint autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt \
  --out-dir autoresearch/oracle-action-value-gate-260629-smoke
```

## Stop Criteria

Stop optimizing this route if:

```text
oracle pairwise/ranking training cannot improve held-out oracle Spearman by 0.10
after two reasonable loss-weight attempts
```

Then switch to:

```text
candidate generation bottleneck analysis or architecture/distribution shift
```

But do not switch before action-value supervision has been tested, because
current probes show the model score itself is not yet aligned with true TC gain.
