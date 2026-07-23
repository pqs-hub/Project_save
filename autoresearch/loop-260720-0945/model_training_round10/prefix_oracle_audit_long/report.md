# Prefix-oracle ranking audit

This fixed-label diagnostic did not run ATPG and was not used for target-circuit selection.

| checkpoint | score | groups | top-1 | type | within-type top-1 | regret (pp) | within regret | negative | Spearman | pairwise | same-type pairwise | CP0/CP1/OP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| round8_incumbent | `typed_marginal_pred` | 116 | 31.897% | 91.379% | 37.069% | 0.1843 | 0.1471 | 5.172% | 0.6783 | 0.8280 | 0.5544 | 2/4/110 |
| round8_incumbent | `typed_return_pred` | 116 | 19.828% | 80.172% | 27.586% | 0.2142 | 0.1840 | 8.621% | 0.5310 | 0.7591 | 0.5174 | 14/7/95 |
| return_within_lr5e5 | `typed_marginal_pred` | 116 | 31.897% | 91.379% | 37.069% | 0.1843 | 0.1471 | 5.172% | 0.6783 | 0.8280 | 0.5544 | 2/4/110 |
| return_within_lr5e5 | `typed_return_pred` | 116 | 20.690% | 79.310% | 28.448% | 0.2115 | 0.1840 | 8.621% | 0.5366 | 0.7615 | 0.5108 | 13/7/96 |
| return_within_lr1e4 | `typed_marginal_pred` | 116 | 31.897% | 91.379% | 37.069% | 0.1843 | 0.1471 | 5.172% | 0.6782 | 0.8280 | 0.5544 | 2/4/110 |
| return_within_lr1e4 | `typed_return_pred` | 116 | 20.690% | 80.172% | 28.448% | 0.2114 | 0.1840 | 8.621% | 0.5469 | 0.7666 | 0.5135 | 12/7/97 |
| return_dual_lr5e5 | `typed_marginal_pred` | 116 | 31.897% | 91.379% | 37.069% | 0.1843 | 0.1471 | 5.172% | 0.6783 | 0.8280 | 0.5544 | 2/4/110 |
| return_dual_lr5e5 | `typed_return_pred` | 116 | 18.103% | 80.172% | 25.862% | 0.2114 | 0.1840 | 8.621% | 0.5441 | 0.7656 | 0.5108 | 12/7/97 |
