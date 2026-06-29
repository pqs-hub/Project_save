# AutoResearch Fix Report: Bounded Residual Planner Score

generated_at: `2026-06-29 11:35 Asia/Shanghai`

## Objective

Implement and test:

```text
keep action_encoder/dynamics/hard_reduction frozen,
train only a bounded residual term,
use validation early stopping
```

## Implementation

Modified:

```text
tpi_jepa/plan.py
scripts/oracle_action_value_probe.py
scripts/finetune_oracle_action_values.py
```

Added score field:

```text
bounded_residual_hybrid_pred
```

Formula:

```text
bounded_residual_hybrid_pred =
    hard_reduction_total_pred * coverage_scale
  + alpha * (reward_pred + return_pred)
```

where:

```text
alpha = bounded_residual_alpha
alpha is clamped by bounded_residual_alpha_bound
```

Training mode:

```text
--train-scope bounded_residual
```

This freezes all model weights and trains only:

```text
bounded_residual_alpha
```

Checkpoint config now stores:

```text
bounded_residual_alpha
bounded_residual_alpha_bound
```

## Verification

Syntax:

```text
python -m py_compile tpi_jepa/plan.py scripts/oracle_action_value_probe.py scripts/finetune_oracle_action_values.py scripts/evaluate_oracle_action_values.py
PASS
```

Smoke:

```text
autoresearch/oracle-action-value-finetune-260629-bounded-residual-smoke/
```

Smoke confirmed:

```text
trainable_prefixes: bounded_residual_alpha
```

## Training

Output:

```text
autoresearch/oracle-action-value-finetune-260629-bounded-residual/
```

Configuration:

```text
train_scope: bounded_residual
ranking_score_field: bounded_residual_hybrid_pred
alpha_init: 0.0
alpha_bound: 0.25
epochs: 8
validation: expanded 16-subckt val
```

History:

| epoch | alpha | train rank loss | val Spearman | val negative top1 | val top1 regret | best |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | -0.112052 | 0.745477 | 0.045261 | 0.375000 | 0.035483 | 1 |
| 2 | -0.155299 | 0.709610 | 0.043534 | 0.375000 | 0.035483 | 0 |
| 3 | -0.192880 | 0.738092 | 0.044134 | 0.375000 | 0.035483 | 0 |
| 4 | -0.211235 | 0.717643 | 0.044134 | 0.375000 | 0.035483 | 0 |
| 5 | -0.221634 | 0.684805 | 0.044486 | 0.375000 | 0.035483 | 0 |
| 6 | -0.230898 | 0.718056 | 0.044461 | 0.375000 | 0.035483 | 0 |
| 7 | -0.234959 | 0.699464 | 0.044461 | 0.375000 | 0.035483 | 0 |
| 8 | -0.237840 | 0.702245 | 0.044461 | 0.375000 | 0.035483 | 0 |

Selected checkpoint:

```text
autoresearch/oracle-action-value-finetune-260629-bounded-residual/best.pt
```

Best alpha:

```text
-0.11205179989337921
```

## Expanded Val Gate

Output:

```text
autoresearch/oracle-action-value-gate-260629-bounded-residual-val/
```

Comparison on expanded held-out subckt val:

| checkpoint | score field | Spearman | negative top1 | top1 regret |
|---|---|---:|---:|---:|
| incumbent | hybrid_pred | 0.031476 | 0.375000 | 0.035483 |
| incumbent | hard_reduction_total_pred | 0.044034 | 0.375000 | 0.035483 |
| incumbent | bounded_residual_hybrid_pred | 0.042002 | 0.375000 | 0.035483 |
| bounded_best | bounded_residual_hybrid_pred | 0.045258 | 0.375000 | 0.035483 |
| bounded_final | bounded_residual_hybrid_pred | 0.044458 | 0.375000 | 0.035483 |

Interpretation:

```text
bounded_best is the first variant that slightly beats hard-only and the default
bounded-residual baseline on expanded val without worsening top1 safety/regret.
```

## Full-Circuit Transfer Gate

Output:

```text
autoresearch/oracle-action-value-gate-260629-bounded-residual-transfer/
```

Comparison on full-circuit transfer:

| checkpoint | score field | Spearman | negative top1 | top1 regret |
|---|---|---:|---:|---:|
| incumbent | hybrid_pred | 0.327398 | 0.166667 | 0.012552 |
| incumbent | hard_reduction_total_pred | 0.324443 | 0.166667 | 0.012552 |
| incumbent | bounded_residual_hybrid_pred | 0.324835 | 0.166667 | 0.012552 |
| bounded_best | bounded_residual_hybrid_pred | 0.324331 | 0.166667 | 0.012552 |
| bounded_final | bounded_residual_hybrid_pred | 0.323770 | 0.166667 | 0.012552 |

Interpretation:

```text
bounded_best preserves transfer top1 safety/regret, but its transfer Spearman is
slightly below incumbent hybrid_pred.
```

## Verdict

Implementation:

```text
DONE
```

Checkpoint:

```text
DO NOT PROMOTE AS DEFAULT PLANNER CHECKPOINT
```

Reason:

```text
The new score field improves expanded val slightly and is transfer-safe by top1,
but it does not beat incumbent hybrid_pred on full-circuit transfer Spearman.
```

## What This Means

This is the best structural direction so far:

```text
Do not move action_encoder/dynamics/hard_reduction.
Use a bounded residual on top of fixed hard_reduction.
```

It avoids the transfer collapse seen in joint training, and it improves
expanded-val rank slightly.

But the effect size is small:

```text
expanded val Spearman gain over hard-only: 0.044034 -> 0.045258
```

So the next work should tune the residual design, not unfreeze dynamics.

## Recommended Next Step

Run a small alpha/bound sweep without retraining model weights:

```text
alpha_bound in {0.05, 0.10, 0.20, 0.25, 0.50}
alpha_init in {-bound, -bound/2, 0, bound/2, bound}
```

or directly grid-search alpha on the fixed expanded train groups and select by
expanded val:

```text
score = hard_reduction_total_pred * coverage_scale
      + alpha * (reward_pred + return_pred)
```

This is cheaper and more stable than gradient training a single scalar through
all graphs.

