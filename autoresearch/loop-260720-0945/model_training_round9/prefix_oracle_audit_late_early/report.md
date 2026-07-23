# Prefix-oracle ranking audit

This fixed-label diagnostic did not run ATPG and was not used for target-circuit selection.

| checkpoint | score | groups | top-1 | type | within-type top-1 | regret (pp) | within regret | negative | Spearman | pairwise | same-type pairwise | CP0/CP1/OP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| round8_incumbent | `typed_marginal_pred` | 30 | 13.333% | 83.333% | 23.333% | 0.1508 | 0.0674 | 0.000% | 0.6836 | 0.8116 | 0.5531 | 0/0/30 |
| horizon_only | `typed_marginal_pred` | 30 | 13.333% | 83.333% | 23.333% | 0.1508 | 0.0674 | 0.000% | 0.6856 | 0.8129 | 0.5531 | 0/0/30 |
| horizon_joint | `typed_marginal_pred` | 30 | 23.333% | 86.667% | 30.000% | 0.1285 | 0.0645 | 0.000% | 0.7077 | 0.8248 | 0.5506 | 0/1/29 |
