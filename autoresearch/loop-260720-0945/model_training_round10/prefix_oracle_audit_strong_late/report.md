# Prefix-oracle ranking audit

This fixed-label diagnostic did not run ATPG and was not used for target-circuit selection.

| checkpoint | score | groups | top-1 | type | within-type top-1 | regret (pp) | within regret | negative | Spearman | pairwise | same-type pairwise | CP0/CP1/OP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| round8_incumbent | `typed_marginal_pred` | 30 | 13.333% | 83.333% | 23.333% | 0.1508 | 0.0674 | 0.000% | 0.6836 | 0.8116 | 0.5531 | 0/0/30 |
| round8_incumbent | `typed_return_pred` | 30 | 26.667% | 83.333% | 30.000% | 0.1449 | 0.1283 | 3.333% | 0.5167 | 0.7389 | 0.4390 | 1/0/29 |
| return_within_lr2e4 | `typed_marginal_pred` | 30 | 13.333% | 83.333% | 23.333% | 0.1508 | 0.0674 | 0.000% | 0.6836 | 0.8116 | 0.5531 | 0/0/30 |
| return_within_lr2e4 | `typed_return_pred` | 30 | 33.333% | 83.333% | 36.667% | 0.1421 | 0.1286 | 0.000% | 0.5728 | 0.7761 | 0.4927 | 0/0/30 |
| return_within_lr5e4 | `typed_marginal_pred` | 30 | 13.333% | 83.333% | 23.333% | 0.1508 | 0.0674 | 0.000% | 0.6836 | 0.8116 | 0.5531 | 0/0/30 |
| return_within_lr5e4 | `typed_return_pred` | 30 | 26.667% | 83.333% | 30.000% | 0.1415 | 0.1057 | 0.000% | 0.6932 | 0.8313 | 0.5838 | 0/0/30 |
| return_dual_lr2e4 | `typed_marginal_pred` | 30 | 13.333% | 83.333% | 23.333% | 0.1508 | 0.0674 | 0.000% | 0.6836 | 0.8116 | 0.5531 | 0/0/30 |
| return_dual_lr2e4 | `typed_return_pred` | 30 | 33.333% | 83.333% | 36.667% | 0.1421 | 0.1286 | 0.000% | 0.5519 | 0.7613 | 0.5016 | 0/0/30 |
