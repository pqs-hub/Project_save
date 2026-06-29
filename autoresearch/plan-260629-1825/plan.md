# Plan: Fixed-Checkpoint Action-Value Ranker

## Goal

Train a small ranker on top of fixed checkpoint predictions.

The ranker does not change the neural checkpoint. It only learns how to combine already exported action features so that actions with higher backend-measured `oracle_delta_tc` are ranked earlier.

Plain meaning:

```text
The old model gives several scores for each action.
We train a small second-stage scorer that learns which score mixture better
matches real backend TC gain.
```

## Why This Plan

Recent evidence says full scratch retraining is not a clean experiment yet:

```text
scratch_0p00 has no oracle ranking loss, but hard F1 still collapses to 0.165301
versus incumbent 0.794805.
```

So full retraining is currently confounded by the training recipe. A fixed-checkpoint ranker avoids that failure mode:

```text
1. original checkpoint stays unchanged
2. hard F1 cannot regress
3. latent/world-model predictions cannot drift
4. the experiment directly tests whether existing model scores contain enough
   signal for real TC ranking
```

## Current Data

Use existing fixed oracle action groups.

| split | file | rows | groups | positive | negative | actions per group |
|---|---|---:|---:|---:|---:|---:|
| train | `autoresearch/oracle-action-probe-260629-expanded-subckt-train/oracle_actions.tsv` | 5184 | 288 | 4468 | 716 | 18 |
| expanded val | `autoresearch/oracle-action-probe-260629-expanded-subckt-val/oracle_actions.tsv` | 864 | 48 | 505 | 359 | 18 |
| transfer | `autoresearch/oracle-action-probe-260629-smoke/oracle_actions.tsv` | 288 | 6 | 190 | 98 | 48 |

Primary checkpoint:

```text
autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt
```

## Scope

Implement one new script:

```text
scripts/train_action_value_ranker.py
```

The script should:

```text
1. load rescored oracle action TSVs
2. build numeric features from fixed checkpoint prediction columns
3. train a small ranker with pairwise ranking loss inside each action group
4. evaluate on train, expanded val, and transfer
5. write TSV metrics and a markdown report
```

Optional but useful:

```text
scripts/evaluate_action_value_ranker.py
```

Only add this second script if keeping train/eval in one script becomes messy.

## Out Of Scope

Do not change these in this plan:

```text
tpi_jepa/model.py
tpi_jepa/train.py
tpi_jepa/plan.py planner scoring for live insertion
neural checkpoint weights
oracle action generation
backend labeling
```

Do not add a new neural `action_value_head` in this step.

## Required Input Preparation

The ranker needs rescored files, because raw `oracle_actions.tsv` has true labels but may not contain all incumbent prediction features.

Generate fixed incumbent rescored TSVs:

```bash
python scripts/evaluate_oracle_action_values.py \
  --oracle-actions autoresearch/oracle-action-probe-260629-expanded-subckt-train/oracle_actions.tsv \
  --checkpoint incumbent=autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt \
  --score-fields reward_pred,guarded_reward,hard_reduction_total_pred,hybrid_pred,bounded_residual_hybrid_pred,derived_hard_reduction_total_pred,derived_hard_reduction_hybrid_pred \
  --plan-device cpu \
  --baseline incumbent \
  --out-dir autoresearch/action-value-ranker-260629/rescore_train

python scripts/evaluate_oracle_action_values.py \
  --oracle-actions autoresearch/oracle-action-probe-260629-expanded-subckt-val/oracle_actions.tsv \
  --checkpoint incumbent=autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt \
  --score-fields reward_pred,guarded_reward,hard_reduction_total_pred,hybrid_pred,bounded_residual_hybrid_pred,derived_hard_reduction_total_pred,derived_hard_reduction_hybrid_pred \
  --plan-device cpu \
  --baseline incumbent \
  --out-dir autoresearch/action-value-ranker-260629/rescore_val

python scripts/evaluate_oracle_action_values.py \
  --oracle-actions autoresearch/oracle-action-probe-260629-smoke/oracle_actions.tsv \
  --checkpoint incumbent=autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt \
  --score-fields reward_pred,guarded_reward,hard_reduction_total_pred,hybrid_pred,bounded_residual_hybrid_pred,derived_hard_reduction_total_pred,derived_hard_reduction_hybrid_pred \
  --plan-device cpu \
  --baseline incumbent \
  --out-dir autoresearch/action-value-ranker-260629/rescore_transfer
```

Expected input files after this step:

```text
autoresearch/action-value-ranker-260629/rescore_train/rescored_oracle_actions.tsv
autoresearch/action-value-ranker-260629/rescore_val/rescored_oracle_actions.tsv
autoresearch/action-value-ranker-260629/rescore_transfer/rescored_oracle_actions.tsv
```

## Feature Design

Use only model/action metadata features.

Allowed numeric score features:

```text
reward_pred
fc_pred
guarded_reward
return_pred
hard_reduction_total_pred
hard_reduction_sa0_pred
hard_reduction_sa1_pred
hybrid_pred
bounded_residual_hybrid_pred
derived_hard_reduction_total_pred
derived_hard_reduction_hybrid_pred
derived_hard_reduction_sa0_pred
derived_hard_reduction_sa1_pred
derived_hard_count_pre_total_pred
derived_hard_count_post_total_pred
candidate_rank
```

Allowed categorical features:

```text
action type: control0/control1/observe
candidate_strategy
benchmark_id
```

Group-normalized features:

For each numeric score column, add:

```text
raw_value
value_minus_group_mean
value_div_group_std
rank_percentile_within_group
```

This matters because different circuits can have different score scales.

Forbidden features:

```text
oracle_delta_tc
oracle_delta_fault_coverage
oracle_delta_pattern_count
oracle_test_coverage
oracle_fault_coverage
oracle_hard_fault_count
oracle_undetected_fault_count
status-derived label flags beyond filtering finite rows
```

These are labels or backend results. Using them as input would leak the answer.

## Ranker Variants

Train these variants first:

| variant | model | reason |
|---|---|---|
| `linear` | one linear layer | easiest to inspect, lowest overfit risk |
| `linear_l2` | linear layer with stronger L2 | tests whether regularization is enough |
| `mlp_small` | Linear -> ReLU -> Linear, hidden size 16 | checks simple nonlinear feature interactions |
| `action_type_linear` | linear plus action-type bias | checks whether control/observe actions need calibration |

Do not start with a larger model.

## Loss

Use pairwise logistic ranking loss inside the same group.

For two actions `i` and `j` in the same group:

```text
if oracle_delta_tc_i > oracle_delta_tc_j:
    ranker_score_i should be greater than ranker_score_j
```

Loss formula:

```text
loss_ij = softplus(-(score_i - score_j) / temperature)
```

Only include pairs where true gains differ enough:

```text
abs(oracle_delta_tc_i - oracle_delta_tc_j) >= pairwise_min_delta
```

Recommended defaults:

```text
pairwise_min_delta = 0.001
temperature = 1.0
epochs = 200
early_stopping_metric = val_top1_regret_then_spearman
patience = 20
batch_unit = group
seed = 2026
```

Why group-level loss:

```text
Actions from different states should not be compared directly because their
base TC and circuit context differ. Ranking is only meaningful inside the same
state/candidate group.
```

## Evaluation Metrics

For each split and variant, report:

```text
groups
mean_spearman
mean_kendall_tau
mean_pearson
mean_top1_real_delta_tc
mean_top1_regret
negative_top1_rate
mean_sign_accuracy
pairwise_accuracy
ndcg_at_5
ndcg_at_10
```

Also report incumbent baseline fields on the same TSV:

```text
hybrid_pred
hard_reduction_total_pred
reward_pred
bounded_residual_hybrid_pred
derived_hard_reduction_hybrid_pred
```

Primary comparison:

```text
ranker_score vs incumbent hybrid_pred
```

Secondary comparison:

```text
ranker_score vs incumbent hard_reduction_total_pred
```

## Promotion Gate

Primary gate:

```text
expanded val
```

Secondary gate:

```text
transfer
```

Promote a ranker only if it satisfies all primary conditions:

```text
expanded mean_spearman >= incumbent hybrid_pred + 0.05
expanded negative_top1_rate <= incumbent hybrid_pred
expanded mean_top1_regret <= incumbent hybrid_pred
```

And these transfer safety conditions:

```text
transfer negative_top1_rate <= incumbent hybrid_pred + 0.10
transfer mean_top1_real_delta_tc >= 0.0
transfer mean_top1_regret <= incumbent hybrid_pred + 0.01
```

Reject immediately if:

```text
expanded negative_top1_rate worsens
transfer mean_top1_real_delta_tc becomes negative while incumbent is positive
```

Why this gate:

```text
Spearman checks whole-list ordering.
negative_top1 checks whether the first chosen action is harmful.
top1_regret checks how much true gain is lost compared with the best available
oracle action in that group.
```

## Output Files

The fix should produce:

```text
autoresearch/action-value-ranker-260629/rankers/linear.pt
autoresearch/action-value-ranker-260629/rankers/linear_l2.pt
autoresearch/action-value-ranker-260629/rankers/mlp_small.pt
autoresearch/action-value-ranker-260629/rankers/action_type_linear.pt
autoresearch/action-value-ranker-260629/ranker_metrics.tsv
autoresearch/action-value-ranker-260629/ranker_predictions_train.tsv
autoresearch/action-value-ranker-260629/ranker_predictions_val.tsv
autoresearch/action-value-ranker-260629/ranker_predictions_transfer.tsv
autoresearch/action-value-ranker-260629/ranker_report.md
autoresearch/action-value-ranker-260629/handoff.json
```

The report should include:

```text
1. best ranker by expanded val gate
2. whether best ranker passes transfer safety
3. feature weights for linear variants
4. top 10 improved and top 10 worsened action groups
5. final verdict: PROMOTE / REJECT / INCONCLUSIVE
```

## Implementation Notes

Use PyTorch for simplicity.

Recommended script interface:

```bash
python scripts/train_action_value_ranker.py \
  --train-rescored autoresearch/action-value-ranker-260629/rescore_train/rescored_oracle_actions.tsv \
  --val-rescored autoresearch/action-value-ranker-260629/rescore_val/rescored_oracle_actions.tsv \
  --transfer-rescored autoresearch/action-value-ranker-260629/rescore_transfer/rescored_oracle_actions.tsv \
  --variants linear,linear_l2,mlp_small,action_type_linear \
  --baseline-score-field hybrid_pred \
  --pairwise-min-delta 0.001 \
  --temperature 1.0 \
  --epochs 200 \
  --patience 20 \
  --seed 2026 \
  --out-dir autoresearch/action-value-ranker-260629
```

Suggested internal functions:

```text
read_tsv(path)
group_rows(rows)
build_feature_table(rows, feature_config)
fit_feature_normalizer(train_features)
apply_feature_normalizer(features)
pairwise_rank_loss(scores, targets, group_ids)
evaluate_scores(rows, score_column)
write_predictions(rows, scores)
write_report(metrics, feature_weights)
```

## Verification

Code checks:

```bash
python -m py_compile scripts/train_action_value_ranker.py
```

Smoke run:

```bash
python scripts/train_action_value_ranker.py \
  --train-rescored autoresearch/action-value-ranker-260629/rescore_train/rescored_oracle_actions.tsv \
  --val-rescored autoresearch/action-value-ranker-260629/rescore_val/rescored_oracle_actions.tsv \
  --transfer-rescored autoresearch/action-value-ranker-260629/rescore_transfer/rescored_oracle_actions.tsv \
  --variants linear \
  --epochs 2 \
  --patience 1 \
  --out-dir autoresearch/action-value-ranker-260629-smoke
```

Full run:

```bash
python scripts/train_action_value_ranker.py \
  --train-rescored autoresearch/action-value-ranker-260629/rescore_train/rescored_oracle_actions.tsv \
  --val-rescored autoresearch/action-value-ranker-260629/rescore_val/rescored_oracle_actions.tsv \
  --transfer-rescored autoresearch/action-value-ranker-260629/rescore_transfer/rescored_oracle_actions.tsv \
  --variants linear,linear_l2,mlp_small,action_type_linear \
  --baseline-score-field hybrid_pred \
  --pairwise-min-delta 0.001 \
  --temperature 1.0 \
  --epochs 200 \
  --patience 20 \
  --seed 2026 \
  --out-dir autoresearch/action-value-ranker-260629
```

Manual output checks:

```bash
test -s autoresearch/action-value-ranker-260629/ranker_metrics.tsv
test -s autoresearch/action-value-ranker-260629/ranker_report.md
test -s autoresearch/action-value-ranker-260629/handoff.json
python -m json.tool autoresearch/action-value-ranker-260629/handoff.json >/dev/null
```

## Expected Outcomes

Possible result A:

```text
ranker improves expanded val and transfer safety.
```

Decision:

```text
Promote the fixed ranker as a planner re-ranking layer and later distill it into
the neural score.
```

Possible result B:

```text
ranker improves expanded val but fails transfer.
```

Decision:

```text
Do not promote. Audit oracle split mismatch and collect more balanced transfer-like
oracle groups.
```

Possible result C:

```text
ranker does not beat hybrid_pred on expanded val.
```

Decision:

```text
The existing exported model scores do not contain enough separable signal. Next
step should be oracle data audit or neural representation changes, not a bigger
ranker.
```

## Next Command

```text
$autoresearch fix autoresearch/plan-260629-1825/plan.md
```

