# Prefix-oracle ranking audit

This fixed-label diagnostic did not run ATPG and was not used for target-circuit selection.

| checkpoint | score | groups | top-1 | type | within-type top-1 | regret (pp) | within regret | negative | Spearman | pairwise | same-type pairwise | CP0/CP1/OP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| round8_incumbent | `typed_marginal_pred` | 30 | 13.333% | 83.333% | 23.333% | 0.1508 | 0.0674 | 0.000% | 0.6836 | 0.8116 | 0.5531 | 0/0/30 |
| round8_incumbent | `typed_return_pred` | 30 | 26.667% | 83.333% | 30.000% | 0.1449 | 0.1283 | 3.333% | 0.5145 | 0.7379 | 0.4348 | 1/0/29 |
| return_within_lr5e5 | `typed_marginal_pred` | 30 | 13.333% | 83.333% | 23.333% | 0.1508 | 0.0674 | 0.000% | 0.6836 | 0.8116 | 0.5531 | 0/0/30 |
| return_within_lr5e5 | `typed_return_pred` | 30 | 26.667% | 83.333% | 30.000% | 0.1449 | 0.1283 | 3.333% | 0.5223 | 0.7420 | 0.4473 | 1/0/29 |
| return_within_lr1e4 | `typed_marginal_pred` | 30 | 13.333% | 83.333% | 23.333% | 0.1508 | 0.0674 | 0.000% | 0.6848 | 0.8121 | 0.5551 | 0/0/30 |
| return_within_lr1e4 | `typed_return_pred` | 30 | 30.000% | 83.333% | 33.333% | 0.1393 | 0.1283 | 0.000% | 0.5339 | 0.7491 | 0.4640 | 0/0/30 |
| return_dual_lr5e5 | `typed_marginal_pred` | 30 | 13.333% | 83.333% | 23.333% | 0.1508 | 0.0674 | 0.000% | 0.6836 | 0.8116 | 0.5531 | 0/0/30 |
| return_dual_lr5e5 | `typed_return_pred` | 30 | 26.667% | 83.333% | 30.000% | 0.1393 | 0.1283 | 0.000% | 0.5214 | 0.7411 | 0.4453 | 0/0/30 |
