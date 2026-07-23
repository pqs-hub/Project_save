# Prefix-oracle ranking audit

This fixed-label diagnostic did not run ATPG and was not used for target-circuit selection.

| checkpoint | score | groups | top-1 | type | within-type top-1 | regret (pp) | within regret | negative | Spearman | pairwise | same-type pairwise | CP0/CP1/OP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| round8_incumbent | `typed_marginal_pred` | 116 | 31.897% | 91.379% | 37.069% | 0.1843 | 0.1471 | 5.172% | 0.6782 | 0.8280 | 0.5544 | 2/4/110 |
| round8_incumbent | `typed_return_pred` | 116 | 18.103% | 80.172% | 25.862% | 0.2142 | 0.1840 | 8.621% | 0.5310 | 0.7591 | 0.5174 | 14/7/95 |
| round8_incumbent | `typed_sa_reduction_total_pred` | 116 | 30.172% | 88.793% | 33.621% | 0.1449 | 0.1185 | 6.897% | 0.5976 | 0.7803 | 0.4510 | 1/8/107 |
| horizon_only_late | `typed_marginal_pred` | 116 | 31.034% | 91.379% | 36.207% | 0.1843 | 0.1471 | 5.172% | 0.6789 | 0.8286 | 0.5616 | 2/4/110 |
| horizon_only_late | `typed_return_pred` | 116 | 19.828% | 80.172% | 27.586% | 0.2142 | 0.1840 | 8.621% | 0.5351 | 0.7611 | 0.5198 | 14/7/95 |
| horizon_only_late | `typed_sa_reduction_total_pred` | 116 | 30.172% | 88.793% | 33.621% | 0.1449 | 0.1185 | 6.897% | 0.5976 | 0.7803 | 0.4510 | 1/8/107 |
| horizon_joint_late | `typed_marginal_pred` | 116 | 27.586% | 88.793% | 35.345% | 0.1871 | 0.1454 | 5.172% | 0.6688 | 0.8227 | 0.5667 | 3/6/107 |
| horizon_joint_late | `typed_return_pred` | 116 | 18.103% | 79.310% | 27.586% | 0.2184 | 0.1840 | 9.483% | 0.5301 | 0.7597 | 0.5266 | 15/7/94 |
| horizon_joint_late | `typed_sa_reduction_total_pred` | 116 | 30.172% | 89.655% | 32.759% | 0.1429 | 0.1185 | 6.034% | 0.5974 | 0.7810 | 0.4492 | 1/7/108 |
