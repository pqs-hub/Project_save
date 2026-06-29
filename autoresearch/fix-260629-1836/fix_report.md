# Fix Report: Fixed-Checkpoint Action-Value Ranker

## Objective

Execute:

```text
$autoresearch fix autoresearch/plan-260629-1825/plan.md
```

Goal:

```text
Train a small ranker on top of fixed incumbent checkpoint predictions, without
changing the neural checkpoint, and evaluate it with expanded/transfer oracle
gates.
```

## Code Change

Added:

```text
scripts/train_action_value_ranker.py
```

The script:

```text
1. reads rescored oracle action TSVs
2. builds model-output and action-metadata features
3. trains linear / linear_l2 / mlp_small / action_type_linear rankers
4. uses pairwise ranking loss inside each oracle action group
5. writes metrics, predictions, ranker checkpoints, feature weights, report, and handoff
```

The script does not modify:

```text
tpi_jepa/model.py
tpi_jepa/train.py
tpi_jepa/plan.py
any neural checkpoint weights
```

## Data Generated

Rescored incumbent features:

```text
autoresearch/action-value-ranker-260629/rescore_train/rescored_oracle_actions.tsv
autoresearch/action-value-ranker-260629/rescore_val/rescored_oracle_actions.tsv
autoresearch/action-value-ranker-260629/rescore_transfer/rescored_oracle_actions.tsv
```

Rows:

```text
train: 5184
expanded val: 864
transfer: 288
```

## Verification

Passed:

```text
python -m py_compile scripts/train_action_value_ranker.py
python scripts/train_action_value_ranker.py ... --variants linear --epochs 2 --patience 1 --out-dir autoresearch/action-value-ranker-260629-smoke
python scripts/train_action_value_ranker.py ... --variants linear,linear_l2,mlp_small,action_type_linear --epochs 200 --patience 20 --out-dir autoresearch/action-value-ranker-260629
test -s autoresearch/action-value-ranker-260629/ranker_metrics.tsv
test -s autoresearch/action-value-ranker-260629/ranker_report.md
python -m json.tool autoresearch/action-value-ranker-260629/handoff.json
```

## Result

Final verdict:

```text
INCONCLUSIVE
```

Best ranker:

```text
linear_l2
```

Reason not promoted:

```text
transfer negative_top1_rate safety failed
```

## Key Metrics

| split | score | Spearman | negative top1 | top1 real delta | top1 regret |
|---|---|---:|---:|---:|---:|
| expanded val | baseline hybrid_pred | 0.031476 | 0.375000 | -0.009015 | 0.035483 |
| expanded val | linear_l2 ranker | 0.102283 | 0.354167 | -0.005004 | 0.031472 |
| transfer | baseline hybrid_pred | 0.327398 | 0.166667 | 0.010638 | 0.012552 |
| transfer | linear_l2 ranker | 0.054477 | 0.500000 | 0.001328 | 0.021862 |

Interpretation:

```text
The fixed ranker can improve the in-distribution expanded validation gate.
But it learns a rule that does not transfer safely to the smoke circuits.
```

## Important Implementation Correction

The first run selected `mlp_small` because early stopping over-prioritized top1 regret.
That selected an unsafe model with worse negative top1.

The selection key was corrected to:

```text
1. lower negative_top1_rate
2. higher Spearman
3. lower top1_regret
```

After correction, the best model became `linear_l2`.

## Artifacts

```text
autoresearch/action-value-ranker-260629/ranker_report.md
autoresearch/action-value-ranker-260629/ranker_metrics.tsv
autoresearch/action-value-ranker-260629/feature_weights.tsv
autoresearch/action-value-ranker-260629/top1_group_deltas.tsv
autoresearch/action-value-ranker-260629/handoff.json
autoresearch/action-value-ranker-260629/rankers/
```

## Recommended Next Step

Do not promote the ranker yet.

Next useful step:

```text
$autoresearch plan Audit oracle train/expanded/transfer split mismatch and add
negative-action-balanced oracle groups before training another fixed ranker.
```

Reason:

```text
The ranker improves expanded validation but fails transfer safety. This points
to split mismatch or insufficient negative-action coverage, not just model
capacity.
```

