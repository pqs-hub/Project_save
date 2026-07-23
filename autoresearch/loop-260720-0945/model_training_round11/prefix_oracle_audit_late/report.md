# Prefix-oracle ranking audit

This fixed-label diagnostic did not run ATPG and was not used for target-circuit selection.

| checkpoint | score | groups | top-1 | type | within-type top-1 | regret (pp) | within regret | negative | Spearman | pairwise | same-type pairwise | CP0/CP1/OP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| round8_incumbent | `typed_marginal_pred` | 30 | 13.333% | 83.333% | 23.333% | 0.1508 | 0.0674 | 0.000% | 0.6836 | 0.8116 | 0.5531 | 0/0/30 |
| round8_incumbent | `typed_return_pred` | 30 | 26.667% | 83.333% | 30.000% | 0.1449 | 0.1283 | 3.333% | 0.5167 | 0.7389 | 0.4390 | 1/0/29 |
| marginal_horizon_lr2e5 | `typed_marginal_pred` | 30 | 13.333% | 83.333% | 23.333% | 0.1508 | 0.0674 | 0.000% | 0.6869 | 0.8134 | 0.5551 | 0/0/30 |
| marginal_horizon_lr2e5 | `typed_return_pred` | 30 | 26.667% | 83.333% | 30.000% | 0.1449 | 0.1283 | 3.333% | 0.5167 | 0.7389 | 0.4390 | 1/0/29 |
| marginal_horizon_lr5e5 | `typed_marginal_pred` | 30 | 13.333% | 83.333% | 23.333% | 0.1508 | 0.0674 | 0.000% | 0.6856 | 0.8129 | 0.5531 | 0/0/30 |
| marginal_horizon_lr5e5 | `typed_return_pred` | 30 | 26.667% | 83.333% | 30.000% | 0.1449 | 0.1283 | 3.333% | 0.5145 | 0.7379 | 0.4348 | 1/0/29 |
| marginal_horizon_lr1e4 | `typed_marginal_pred` | 30 | 20.000% | 83.333% | 30.000% | 0.1508 | 0.0674 | 0.000% | 0.6836 | 0.8116 | 0.5447 | 0/0/30 |
| marginal_horizon_lr1e4 | `typed_return_pred` | 30 | 26.667% | 83.333% | 30.000% | 0.1449 | 0.1283 | 3.333% | 0.5156 | 0.7384 | 0.4369 | 1/0/29 |
