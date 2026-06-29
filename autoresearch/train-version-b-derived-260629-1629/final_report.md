# Version B Training Report

## Verdict

Version B is **not promoted**.

It can still learn node-level hard labels, but the derived hard-reduction score is too weak to use for planner ranking.

## Training

Config:

```text
lambda_hard_count = 0.0
lambda_hard_reduction = 0.0
lambda_fc = 0.0
lambda_return = 0.0
lambda_oracle_rank = 0.0
hard_value_mode = derived_from_node_hard
```

Training completed 4 epochs:

```text
epoch 1 train_loss=0.752731 val_loss=0.235351
epoch 2 train_loss=0.592576 val_loss=0.196425
epoch 3 train_loss=0.506286 val_loss=0.175646
epoch 4 train_loss=0.407765 val_loss=0.160915
```

## Hard Gate

| checkpoint | hard F1 tuned | old hard-reduction score | derived hard-reduction score | derived sign acc | predictive score |
|---|---:|---:|---:|---:|---:|
| incumbent | 0.794805 | 0.798620 | 0.692207 | 0.769531 | 0.820147 |
| version_B | 0.802648 | 0.096044 | 0.591637 | 0.769531 | 0.655775 |

Interpretation:

- Node hard classifier is fine: `0.802648 > 0.794805`.
- Old hard-reduction head is intentionally not trained, so old score collapses.
- Derived hard-reduction score is lower than incumbent: `0.591637 < 0.692207`.
- Predictive score drops strongly: `0.655775 < 0.820147`.

## Expanded Oracle Gate

| checkpoint | score | Spearman | negative top1 | top1 regret |
|---|---|---:|---:|---:|
| incumbent | derived_hard_reduction_total_pred | 0.086327 | 0.479167 | 0.019714 |
| version_B | derived_hard_reduction_total_pred | -0.143447 | 0.583333 | 0.038149 |
| incumbent | hard_reduction_total_pred | 0.044034 | 0.375000 | 0.035483 |
| version_B | hard_reduction_total_pred | 0.143597 | 0.145833 | 0.017511 |

Interpretation:

- The actual Version B score is derived hard reduction, and it is bad.
- The untrained old hard-reduction head happens to rank well on expanded val, but it is not a valid Version B mechanism.

## Transfer Oracle Gate

| checkpoint | score | Spearman | negative top1 | top1 regret |
|---|---|---:|---:|---:|
| incumbent | derived_hard_reduction_total_pred | 0.046113 | 0.500000 | 0.021963 |
| version_B | derived_hard_reduction_total_pred | -0.007391 | 0.500000 | 0.022142 |
| incumbent | hard_reduction_total_pred | 0.324443 | 0.166667 | 0.012552 |
| version_B | hard_reduction_total_pred | -0.069130 | 0.000000 | 0.020537 |

Interpretation:

- Derived transfer ranking is near zero.
- Version B does not produce a useful action-value score from node hard labels alone.

## Conclusion

The idea is partially validated:

```text
node-level hard labels alone can train a good hard classifier
```

But the key assumption fails:

```text
hard_reduction can be safely derived from node hard probabilities
```

At least with the current formula, derived hard reduction is too noisy for action ranking.

## Next Step

Do not promote Version B.

Recommended next experiment:

```text
Version A plus remove hard_count loss is acceptable for hard classification,
but planner ranking still needs a direct action-level signal.
```

Possible next directions:

1. Keep direct `hard_reduction_head`, remove only `hard_count_head`.
2. Add a consistency loss: `hard_reduction_head` should match derived reduction.
3. Use derived count only as auxiliary diagnostic, not as planner score.
