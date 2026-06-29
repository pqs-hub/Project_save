# AutoResearch Fix Report: Expanded Oracle Groups + Heads-Hybrid Early Stopping

generated_at: `2026-06-29 11:10 Asia/Shanghai`

## Objective

Execute the next step:

```text
Export more oracle action groups from the 131 labeled subckt distribution,
then rerun heads_hybrid with validation early stopping.
```

## Code Changes

### Validation Early Stopping

Modified:

```text
scripts/finetune_oracle_action_values.py
```

Added:

```text
--val-oracle-actions
```

The finetune script now records per-epoch validation metrics:

```text
val_rank_loss
val_spearman
val_negative_top1_rate
val_top1_regret
```

and saves:

```text
best.pt
candidate.pt
```

Best checkpoint selection order:

```text
1. higher val_spearman
2. lower val_negative_top1_rate
3. lower val_top1_regret
4. lower val_rank_loss
```

### Planner Cache Bugfix

Modified:

```text
tpi_jepa/plan.py
scripts/oracle_action_value_probe.py
```

Added:

```text
clear_planner_caches()
```

and called it before each benchmark in `oracle_action_value_probe.py`.

Reason:

```text
Long multi-subckt oracle probes can reuse Python object ids for graph objects.
The planner relation cache was keyed by id(graph), so a later graph could
receive stale relation features from an earlier graph.
```

Observed failure before the fix:

```text
RuntimeError: Sizes of tensors must match except in dimension 1.
Expected size 318 but got size 404 for tensor number 2.
```

## Expanded Oracle Split

Output:

```text
autoresearch/fix-260629-expanded-subckt-split/
```

Split:

| split | subckts | groups | actions |
|---|---:|---:|---:|
| train | 96 | 288 | 5184 |
| val | 16 | 48 | 864 |

This expands from the previous:

| split | old subckts | old groups | old actions |
|---|---:|---:|---:|
| train | 24 | 72 | 1296 |
| val | 8 | 24 | 432 |

Candidate cache:

```text
autoresearch/tp-candidates-labeled-subckt-expanded-260629/
```

## Expanded Oracle Export

Train oracle:

```text
autoresearch/oracle-action-probe-260629-expanded-subckt-train/
```

Records:

```text
oracle_actions: 5184
oracle_groups: 288
reused_oracle_actions: 1296
evaluated_oracle_actions: 3888
```

Val oracle:

```text
autoresearch/oracle-action-probe-260629-expanded-subckt-val/
```

Records:

```text
oracle_actions: 864
oracle_groups: 48
reused_oracle_actions: 432
evaluated_oracle_actions: 432
```

## Training

Output:

```text
autoresearch/oracle-action-value-finetune-260629-expanded-heads-hybrid/
```

Configuration:

```text
--train-scope heads
--train-heads reward,return
--ranking-score-field hybrid_pred
--lambda-oracle-value 0.0
--lambda-oracle-rank 1.0
--val-oracle-actions autoresearch/oracle-action-probe-260629-expanded-subckt-val/oracle_actions.tsv
```

History:

| epoch | train rank loss | val Spearman | val negative top1 | val top1 regret | best |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.720058 | 0.017916 | 0.375000 | 0.035528 | 1 |
| 2 | 0.642586 | 0.000954 | 0.500000 | 0.037870 | 0 |
| 3 | 0.619700 | -0.038414 | 0.479167 | 0.037661 | 0 |
| 4 | 0.577001 | -0.064821 | 0.500000 | 0.033576 | 0 |
| 5 | 0.558705 | -0.092895 | 0.520833 | 0.033625 | 0 |
| 6 | 0.559712 | -0.122485 | 0.541667 | 0.034191 | 0 |
| 7 | 0.554147 | -0.110674 | 0.520833 | 0.032797 | 0 |
| 8 | 0.552198 | -0.117140 | 0.520833 | 0.033149 | 0 |

Interpretation:

```text
The model overfits quickly. Train rank loss keeps improving, but validation
Spearman degrades after epoch 1.
```

Selected checkpoint:

```text
autoresearch/oracle-action-value-finetune-260629-expanded-heads-hybrid/best.pt
```

## Expanded Val Gate

Output:

```text
autoresearch/oracle-action-value-gate-260629-expanded-heads-hybrid-val/
```

Hybrid comparison:

| checkpoint | hybrid Spearman | negative top1 | top1 regret |
|---|---:|---:|---:|
| incumbent | 0.031476 | 0.375000 | 0.035483 |
| heads_hybrid_old | 0.021053 | 0.416667 | 0.035649 |
| expanded_best | 0.017519 | 0.375000 | 0.035528 |
| expanded_final | -0.117110 | 0.520833 | 0.033149 |

Interpretation:

```text
expanded_best does not improve over incumbent on expanded val.
expanded_final is clearly overfit and unsafe by negative_top1.
```

## Full-Circuit Transfer Gate

Output:

```text
autoresearch/oracle-action-value-gate-260629-expanded-heads-hybrid-transfer/
```

Hybrid comparison:

| checkpoint | hybrid Spearman | negative top1 | top1 regret |
|---|---:|---:|---:|
| incumbent | 0.327398 | 0.166667 | 0.012552 |
| heads_hybrid_old | 0.334122 | 0.166667 | 0.012552 |
| expanded_best | 0.331447 | 0.166667 | 0.012552 |
| expanded_final | 0.327306 | 0.166667 | 0.017673 |

Interpretation:

```text
heads_hybrid remains transfer-safe because hard_reduction and dynamics are
unchanged. expanded_best is transfer-safe but not better than the old
heads_hybrid transfer score.
```

## Verdict

Implementation:

```text
DONE
```

Data export:

```text
DONE
```

Candidate:

```text
DO NOT PROMOTE
```

Reason:

```text
expanded_best is transfer-safe but does not improve expanded held-out subckt
ranking over incumbent.
```

## Diagnosis

The larger oracle dataset changed the conclusion:

```text
The previous 8-subckt val was too small/noisy.
```

On the 16-subckt expanded val:

```text
incumbent hybrid_pred is already stronger than heads_hybrid variants.
```

The head-only residual correction is safe but too weak:

```text
reward_head/return_head cannot consistently reorder actions against the
hard_reduction-dominated hybrid score.
```

The joint variants can move ranking more, but they damage transfer by moving
action_encoder/dynamics.

## Recommended Next Step

Do not promote any oracle-finetuned checkpoint yet.

Next useful experiment should change the score composition, not just the loss:

```text
learn a bounded residual score from reward/return and add it to fixed
hard_reduction_total_pred
```

Concretely:

```text
planner_score = hard_reduction_total_pred * coverage_scale
              + alpha * residual_reward_return
```

with:

```text
action_encoder/dynamics/hard_reduction frozen
alpha small or learned with clamp
validation early stopping
```

This addresses the observed issue:

```text
hard_reduction gives transfer safety, but reward/return residual needs bounded
influence to avoid corrupting held-out ranking/top1.
```

