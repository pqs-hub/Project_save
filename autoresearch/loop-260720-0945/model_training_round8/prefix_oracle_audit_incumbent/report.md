# Prefix-oracle ranking audit

This fixed-label diagnostic did not run ATPG and was not used for target-circuit selection.

| checkpoint | score | groups | top-1 | type | within-type top-1 | regret (pp) | within regret | negative | Spearman | pairwise | same-type pairwise | CP0/CP1/OP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| round7_incumbent | `typed_marginal_pred` | 116 | 26.724% | 92.241% | 30.172% | 0.1775 | 0.1546 | 4.310% | 0.6624 | 0.8184 | 0.5248 | 3/3/110 |
| round7_incumbent | `typed_return_pred` | 116 | 19.828% | 79.310% | 27.586% | 0.2139 | 0.1819 | 9.483% | 0.5244 | 0.7562 | 0.5258 | 15/7/94 |
| round7_incumbent | `typed_sa_reduction_total_pred` | 116 | 29.310% | 89.655% | 31.897% | 0.1639 | 0.1489 | 6.034% | 0.6114 | 0.7896 | 0.4686 | 1/7/108 |
