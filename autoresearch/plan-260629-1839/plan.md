# Plan: Oracle Split Audit And Negative-Action-Balanced Groups

## Goal

Audit why the fixed-checkpoint ranker improved `expanded val` but failed `transfer`, then build more useful oracle action groups before training another ranker.

Plain meaning:

```text
The ranker did better on the practice exam, but failed the transfer exam.
Before training again, check whether the practice data has the same kind of bad
actions as transfer. Then create training groups that contain enough bad actions.
```

## Current Evidence

Previous fixed ranker:

```text
autoresearch/action-value-ranker-260629/
```

Best variant:

```text
linear_l2
```

Result:

| split | score | Spearman | negative top1 | top1 real delta | top1 regret |
|---|---|---:|---:|---:|---:|
| expanded val | baseline `hybrid_pred` | 0.031476 | 0.375000 | -0.009015 | 0.035483 |
| expanded val | `linear_l2` | 0.102283 | 0.354167 | -0.005004 | 0.031472 |
| transfer | baseline `hybrid_pred` | 0.327398 | 0.166667 | 0.010638 | 0.012552 |
| transfer | `linear_l2` | 0.054477 | 0.500000 | 0.001328 | 0.021862 |

Conclusion:

```text
The ranker learned something useful for expanded val, but the learned rule is
unsafe on transfer.
```

## Split Mismatch Already Visible

Current oracle splits:

| split | rows | groups | positive | negative | negative rate | groups with no negative action |
|---|---:|---:|---:|---:|---:|---:|
| train | 5184 | 288 | 4468 | 716 | 13.81% | 168 |
| expanded val | 864 | 48 | 505 | 359 | 41.55% | 11 |
| transfer | 288 | 6 | 190 | 98 | 34.03% | 3 |

Key mismatch:

```text
train has too many all-positive groups.
expanded val and transfer have many more negative control actions.
```

Action-type mismatch:

| split | action type | negative rate |
|---|---|---:|
| train | control0 | 20.89% |
| train | control1 | 19.44% |
| train | observe | 1.10% |
| expanded val | control0 | 63.89% |
| expanded val | control1 | 57.99% |
| expanded val | observe | 2.78% |
| transfer | control0 | 50.00% |
| transfer | control1 | 50.00% |
| transfer | observe | 2.08% |

Important observation:

```text
Negative actions are mostly control0/control1, not observe.
```

Benchmark mismatch:

```text
transfer b15_C has 0% negative actions.
transfer i2c_aig has 68.06% negative actions.
```

So transfer safety mostly means:

```text
Do not pick bad control actions on i2c_aig-like groups.
```

## Scope

Implement two scripts.

### Script 1: Audit Existing Oracle Groups

Add:

```text
scripts/audit_oracle_action_groups.py
```

It should read one or more oracle/rescored TSV files and report:

```text
split-level stats
benchmark-level stats
candidate_strategy stats
action_type stats
group-level negative counts
group-level positive counts
delta_tc quantiles
best/worst action per group
all-positive groups
all-negative groups
groups with high negative rate
groups where ranker worsened baseline top1
```

Input examples:

```text
raw oracle_actions.tsv
rescored_oracle_actions.tsv
ranker top1_group_deltas.tsv
```

Output:

```text
oracle_group_audit_summary.tsv
oracle_group_audit_by_group.tsv
oracle_group_audit_by_action_type.tsv
oracle_group_audit_by_strategy.tsv
oracle_group_audit_by_benchmark.tsv
oracle_group_audit_report.md
handoff.json
```

### Script 2: Build Balanced Oracle Action Subsets

Add:

```text
scripts/build_balanced_oracle_action_subset.py
```

It should create balanced train/val subsets from already labeled oracle rows.

Important:

```text
This script should not duplicate rows by default.
It should select groups/actions and optionally write per-row weights.
```

Selection target:

```text
keep groups with at least 3 negative and 3 positive actions when possible
prefer control0/control1 negatives
keep observe positives as useful high-gain candidates
preserve full groups when possible, but allow action-level trimming for very large pools
```

Output:

```text
balanced_train_oracle_actions.tsv
balanced_val_oracle_actions.tsv
balanced_manifest.json
balance_report.md
handoff.json
```

## Out Of Scope

Do not train a new ranker in this plan.

Do not change:

```text
tpi_jepa/model.py
tpi_jepa/train.py
tpi_jepa/plan.py
```

Do not add a new neural head.

Do not call backend for new labels in the first fix pass unless existing labeled rows are insufficient.

## Phase 1: Audit Current Splits

Command:

```bash
python scripts/audit_oracle_action_groups.py \
  --oracle-tsv train=autoresearch/oracle-action-probe-260629-expanded-subckt-train/oracle_actions.tsv \
  --oracle-tsv expanded_val=autoresearch/oracle-action-probe-260629-expanded-subckt-val/oracle_actions.tsv \
  --oracle-tsv transfer=autoresearch/oracle-action-probe-260629-smoke/oracle_actions.tsv \
  --ranker-top1-deltas autoresearch/action-value-ranker-260629/top1_group_deltas.tsv \
  --out-dir autoresearch/oracle-split-audit-260629
```

Required report sections:

```text
1. split-level sign distribution
2. per-action-type sign distribution
3. per-strategy sign distribution
4. per-benchmark sign distribution
5. group negative-count histogram
6. groups where ranker made transfer worse
7. recommendation for balancing thresholds
```

Acceptance:

```text
python -m py_compile scripts/audit_oracle_action_groups.py
test -s autoresearch/oracle-split-audit-260629/oracle_group_audit_report.md
test -s autoresearch/oracle-split-audit-260629/oracle_group_audit_by_group.tsv
python -m json.tool autoresearch/oracle-split-audit-260629/handoff.json >/dev/null
```

## Phase 2: Build Balanced Subsets From Existing Labels

Command:

```bash
python scripts/build_balanced_oracle_action_subset.py \
  --train-oracle autoresearch/oracle-action-probe-260629-expanded-subckt-train/oracle_actions.tsv \
  --val-oracle autoresearch/oracle-action-probe-260629-expanded-subckt-val/oracle_actions.tsv \
  --transfer-oracle autoresearch/oracle-action-probe-260629-smoke/oracle_actions.tsv \
  --min-negatives-per-group 3 \
  --min-positives-per-group 3 \
  --prefer-negative-types control0,control1 \
  --max-actions-per-group 18 \
  --out-dir autoresearch/oracle-balanced-groups-260629
```

Expected behavior:

```text
1. keep all groups that satisfy the min positive/negative requirement
2. drop or mark all-positive groups as low-value for rank training
3. preserve transfer as evaluation-only, not training
4. write a manifest showing how many groups/rows were kept/dropped
```

Acceptance:

```text
python -m py_compile scripts/build_balanced_oracle_action_subset.py
test -s autoresearch/oracle-balanced-groups-260629/balanced_train_oracle_actions.tsv
test -s autoresearch/oracle-balanced-groups-260629/balanced_val_oracle_actions.tsv
test -s autoresearch/oracle-balanced-groups-260629/balance_report.md
python -m json.tool autoresearch/oracle-balanced-groups-260629/handoff.json >/dev/null
```

Minimum useful balanced data target:

```text
balanced_train groups >= 80
balanced_val groups >= 24
balanced_train negative rate between 25% and 55%
balanced_val negative rate between 25% and 60%
control0/control1 negative examples are present in >= 70% of balanced_train groups
```

If the existing data cannot meet this target, Phase 3 is required.

## Phase 3: Generate More Negative-Rich Oracle Groups If Needed

Only run this if Phase 2 reports insufficient balanced groups.

The current train data has many all-positive groups:

```text
168 / 288 train groups have no negative action
```

To collect more negative-rich labels, use a larger and more control-heavy candidate pool.

Recommended generation direction:

```text
1. use sampled subckts from the same training distribution
2. increase max_nets
3. keep CP0/CP1 heavily represented
4. include strategies that also appear in transfer when possible
5. label first, then select balanced groups
```

Suggested command shape:

```bash
python scripts/oracle_action_value_probe.py \
  --checkpoint autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt \
  --benchmarks <selected_subckt_ids> \
  --candidate-strategies cached_stride,cached_hard_cone,cached_random,hard_fault_recall_union \
  --action-types CP0,CP1,OP \
  --max-nets 16 \
  --states initial \
  --patterns 10000 \
  --seed 2026 \
  --backend atalanta-bist \
  --plan-device cpu \
  --resume \
  --out-dir autoresearch/oracle-action-probe-260629-negative-rich-subckt
```

If runtime is too high, use a staged collection:

```text
Stage 1: 16 subckts x 4 strategies x 16 nets x 3 actions
Stage 2: audit negative rate
Stage 3: add more subckts only where negative rate remains too low
```

Important:

```text
Do not train on transfer b15_C/i2c_aig labels.
Keep transfer evaluation-only.
```

## Phase 4: Rescore Balanced Sets For Ranker Training

After balanced raw oracle TSVs exist, rescore them with incumbent:

```bash
python scripts/evaluate_oracle_action_values.py \
  --oracle-actions autoresearch/oracle-balanced-groups-260629/balanced_train_oracle_actions.tsv \
  --checkpoint incumbent=autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt \
  --score-fields reward_pred,guarded_reward,hard_reduction_total_pred,hybrid_pred,bounded_residual_hybrid_pred,derived_hard_reduction_total_pred,derived_hard_reduction_hybrid_pred \
  --plan-device cpu \
  --baseline incumbent \
  --out-dir autoresearch/oracle-balanced-groups-260629/rescore_train

python scripts/evaluate_oracle_action_values.py \
  --oracle-actions autoresearch/oracle-balanced-groups-260629/balanced_val_oracle_actions.tsv \
  --checkpoint incumbent=autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt \
  --score-fields reward_pred,guarded_reward,hard_reduction_total_pred,hybrid_pred,bounded_residual_hybrid_pred,derived_hard_reduction_total_pred,derived_hard_reduction_hybrid_pred \
  --plan-device cpu \
  --baseline incumbent \
  --out-dir autoresearch/oracle-balanced-groups-260629/rescore_val
```

Acceptance:

```text
test -s autoresearch/oracle-balanced-groups-260629/rescore_train/rescored_oracle_actions.tsv
test -s autoresearch/oracle-balanced-groups-260629/rescore_val/rescored_oracle_actions.tsv
```

## Phase 5: Ranker Rerun Gate

This plan does not rerun the ranker, but it defines the next run.

Next ranker command:

```bash
python scripts/train_action_value_ranker.py \
  --train-rescored autoresearch/oracle-balanced-groups-260629/rescore_train/rescored_oracle_actions.tsv \
  --val-rescored autoresearch/oracle-balanced-groups-260629/rescore_val/rescored_oracle_actions.tsv \
  --transfer-rescored autoresearch/action-value-ranker-260629/rescore_transfer/rescored_oracle_actions.tsv \
  --variants linear,linear_l2,action_type_linear \
  --baseline-score-field hybrid_pred \
  --pairwise-min-delta 0.001 \
  --temperature 1.0 \
  --epochs 200 \
  --patience 20 \
  --seed 2026 \
  --out-dir autoresearch/action-value-ranker-balanced-260629
```

Do not include `mlp_small` first unless the linear models still underfit.

Reason:

```text
mlp_small overfit train and failed expanded/transfer safety.
```

Promotion gate:

```text
balanced val Spearman >= incumbent hybrid_pred + 0.05
balanced val negative_top1_rate <= incumbent hybrid_pred
balanced val top1_regret <= incumbent hybrid_pred
expanded original val negative_top1_rate does not regress
transfer negative_top1_rate <= incumbent hybrid_pred + 0.10
transfer mean_top1_real_delta_tc >= incumbent hybrid_pred - 0.005
```

## Key Design Decisions

### Do Not Train On Transfer

Transfer exists to detect distribution shift.

If we train on transfer:

```text
we no longer know whether the ranker generalizes.
```

### Balance By Groups, Not Just Rows

Pairwise ranking loss compares actions inside the same group.

A row-level negative rate is not enough. The group must contain both good and bad actions.

Bad example:

```text
many all-positive groups + a few all-negative groups
```

Good example:

```text
most groups contain positive and negative actions, especially control0/control1 negatives
```

### Keep Observe Actions But Do Not Let Them Dominate

Observe actions are usually positive:

```text
train observe negative rate = 1.10%
transfer observe negative rate = 2.08%
```

They are useful as positive anchors, but they do not teach the model how to avoid bad control actions.

### Control0/Control1 Need More Safety Coverage

Transfer control negative rates:

```text
control0 = 50%
control1 = 50%
```

So the balanced data must explicitly include negative control actions.

## Final Verify Checklist

The implementation is complete when these pass:

```bash
python -m py_compile scripts/audit_oracle_action_groups.py scripts/build_balanced_oracle_action_subset.py

python scripts/audit_oracle_action_groups.py \
  --oracle-tsv train=autoresearch/oracle-action-probe-260629-expanded-subckt-train/oracle_actions.tsv \
  --oracle-tsv expanded_val=autoresearch/oracle-action-probe-260629-expanded-subckt-val/oracle_actions.tsv \
  --oracle-tsv transfer=autoresearch/oracle-action-probe-260629-smoke/oracle_actions.tsv \
  --ranker-top1-deltas autoresearch/action-value-ranker-260629/top1_group_deltas.tsv \
  --out-dir autoresearch/oracle-split-audit-260629

python scripts/build_balanced_oracle_action_subset.py \
  --train-oracle autoresearch/oracle-action-probe-260629-expanded-subckt-train/oracle_actions.tsv \
  --val-oracle autoresearch/oracle-action-probe-260629-expanded-subckt-val/oracle_actions.tsv \
  --transfer-oracle autoresearch/oracle-action-probe-260629-smoke/oracle_actions.tsv \
  --min-negatives-per-group 3 \
  --min-positives-per-group 3 \
  --prefer-negative-types control0,control1 \
  --max-actions-per-group 18 \
  --out-dir autoresearch/oracle-balanced-groups-260629

test -s autoresearch/oracle-split-audit-260629/oracle_group_audit_report.md
test -s autoresearch/oracle-balanced-groups-260629/balance_report.md
python -m json.tool autoresearch/oracle-split-audit-260629/handoff.json >/dev/null
python -m json.tool autoresearch/oracle-balanced-groups-260629/handoff.json >/dev/null
```

## Expected Outcome

Expected best case:

```text
Balanced train/val groups are enough, and the next ranker improves transfer safety.
```

Expected likely case:

```text
Existing train labels are insufficient because too many train groups are all-positive.
Then Phase 3 must collect more negative-rich oracle labels before retraining.
```

Failure case:

```text
Even balanced groups do not improve transfer.
```

Then the issue is probably not just data balance. It may require:

```text
1. transfer-like subckt sampling
2. richer structural features
3. neural representation changes
```

## Next Command

```text
$autoresearch fix autoresearch/plan-260629-1839/plan.md
```

