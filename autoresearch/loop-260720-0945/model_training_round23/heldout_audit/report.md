# Prefix-oracle ranking audit

This fixed-label diagnostic did not run ATPG and was not used for target-circuit selection.

| checkpoint | score | groups | top-1 | type | within-type top-1 | regret (pp) | within regret | negative | Spearman | pairwise | same-type pairwise | CP0/CP1/OP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| r21 | `typed_return_pred` | 10 | 10.000% | 80.000% | 10.000% | 0.0473 | 0.0615 | 20.000% | 0.4484 | 0.7073 | 0.3951 | 2/0/8 |
| r23 | `typed_return_pred` | 10 | 10.000% | 80.000% | 10.000% | 0.0473 | 0.0615 | 20.000% | 0.4378 | 0.7010 | 0.3888 | 2/0/8 |
