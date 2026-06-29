# AutoResearch Fix Report: Labeled-Subckt Oracle Action-Value Round

generated_at: `2026-06-29 05:32 Asia/Shanghai`

## Objective

Execute corrected plan:

```text
autoresearch/plan-260629-0522/plan.md
```

This superseded the incorrect original-full-circuit training plan:

```text
autoresearch/plan-260629-0509/plan.md
```

Reason:

```text
The model training distribution is the 131 sampled/labeled subgraphs from
labels.csv. Oracle action-value finetune data must come from that distribution,
not from original full eval circuits.
```

## Split

Source:

```text
/data4/pengqingsong/DFT/Dataset/atalanta_bist_lowtc_subckt_100k_labels/labels.csv
```

Found:

```text
131 labeled benchmark_id values
```

Deterministic selected split:

```text
24 oracle_train subckts
8 oracle_val subckts
```

Split files:

```text
autoresearch/fix-260629-labeled-subckt-split/labeled_subckt_ids.txt
autoresearch/fix-260629-labeled-subckt-split/oracle_train_subckts.txt
autoresearch/fix-260629-labeled-subckt-split/oracle_val_subckts.txt
autoresearch/fix-260629-labeled-subckt-split/split.json
```

## Candidate Cache

Built cache for all 32 selected labeled subckts:

```text
autoresearch/tp-candidates-labeled-subckt-260629/
```

## Oracle Data Generated

### Train Oracle

Output:

```text
autoresearch/oracle-action-probe-260629-labeled-subckt-train/
```

Records:

```text
oracle_actions: 1296
oracle_groups: 72
prediction_metrics: 288
rank_metrics: 864
```

### Held-Out Val Oracle

Output:

```text
autoresearch/oracle-action-probe-260629-labeled-subckt-val/
```

Records:

```text
oracle_actions: 432
oracle_groups: 24
prediction_metrics: 96
rank_metrics: 288
```

This satisfies the data acceptance thresholds:

```text
train oracle >= 500 actions and >= 16 groups
val oracle >= 150 actions and >= 6 groups
```

## Finetune

Output:

```text
autoresearch/oracle-action-value-finetune-260629-labeled-subckt/
```

Training:

```text
train_heads: reward,return
epochs: 5
lambda_oracle_value: 1.0
lambda_oracle_rank: 1.0
groups: 72
pairs: 5032
```

History:

| epoch | loss | value loss | rank loss |
|---:|---:|---:|---:|
| 1 | 1.063273 | 0.475988 | 0.587285 |
| 2 | 1.056870 | 0.469602 | 0.587268 |
| 3 | 1.054374 | 0.467249 | 0.587124 |
| 4 | 1.054118 | 0.467738 | 0.586380 |
| 5 | 1.051867 | 0.465461 | 0.586406 |

Interpretation:

```text
The finetune learned a small amount on both value and rank losses.
The rank-loss movement is still weak.
```

## Primary Gate: Held-Out Labeled Subckts

Output:

```text
autoresearch/oracle-action-value-gate-260629-labeled-subckt-val/
```

Verdict:

```text
candidate: INCONCLUSIVE
incumbent: INCONCLUSIVE
```

Key rows:

| checkpoint | score_field | mean Spearman | negative top1 | top1 regret |
|---|---|---:|---:|---:|
| incumbent | hybrid_pred | 0.005693 | 0.458333 | 0.043452 |
| candidate | hybrid_pred | 0.004533 | 0.416667 | 0.043300 |
| incumbent | reward_pred | -0.137870 | 0.416667 | 0.023290 |
| candidate | reward_pred | -0.205044 | 0.541667 | 0.031650 |

Interpretation:

```text
Hybrid top1 safety improved slightly, but ranking did not improve.
Reward_pred got worse on held-out labeled subckts.
This is not a promotion-quality in-distribution improvement.
```

## Transfer Gate: Full b15_C/i2c_aig Oracle

Output:

```text
autoresearch/oracle-action-value-gate-260629-labeled-subckt-transfer/
```

Verdict:

```text
candidate: REJECT
incumbent: INCONCLUSIVE
```

Key rows:

| checkpoint | score_field | mean Spearman | negative top1 | top1 regret |
|---|---|---:|---:|---:|
| incumbent | hybrid_pred | 0.327398 | 0.166667 | 0.012552 |
| candidate | hybrid_pred | 0.325323 | 0.166667 | 0.012557 |
| incumbent | reward_pred | 0.294742 | 0.500000 | 0.020223 |
| candidate | reward_pred | 0.348502 | 0.333333 | 0.022373 |

Why REJECT:

```text
The gate selects the candidate's best score field by mean Spearman. Candidate
reward_pred improves Spearman over the incumbent's best field, but its
negative_top1_rate is worse than the incumbent best field. This triggers the
safety guardrail.
```

The important lesson:

```text
Better rank correlation alone is insufficient. The model still chooses harmful
top1 actions too often under the best-looking reward score.
```

## Conclusion

This corrected experiment answered the distribution question:

```text
Even with oracle action-value training data from the actual 131-subckt training
distribution, head-only reward/return finetune is not enough.
```

Observed failure mode:

```text
reward_pred can improve Spearman on full-circuit transfer, but top1 safety and
regret do not improve enough.
```

## Recommended Next Step

Do not promote this candidate.

The next method should not be just "more epochs on reward_head".

More defensible next options:

```text
1. Add explicit top1/negative-action penalty or sign-calibration loss.
2. Train a dedicated action_value_head instead of overloading reward_pred.
3. Include hard_reduction_head in finetune, because hybrid/hard_reduction remain the safer planner signals.
4. Generate oracle groups for non-initial states, if final planner uses rolled-out states.
```

Best immediate plan:

```text
Add a sign/top1 safety loss on oracle groups and compare against this candidate.
```
