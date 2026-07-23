# Prefix-oracle ranking audit

This fixed-label diagnostic did not run ATPG and was not used for target-circuit selection.

| checkpoint | score | groups | top-1 | type | within-type top-1 | regret (pp) | within regret | negative | Spearman | pairwise | same-type pairwise | CP0/CP1/OP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| round7_incumbent | `typed_marginal_pred` | 116 | 26.724% | 92.241% | 30.172% | 0.1775 | 0.1546 | 4.310% | 0.6624 | 0.8184 | 0.5248 | 3/3/110 |
| round7_incumbent | `typed_return_pred` | 116 | 20.690% | 79.310% | 28.448% | 0.2139 | 0.1819 | 9.483% | 0.5245 | 0.7562 | 0.5258 | 15/7/94 |
| round7_incumbent | `typed_sa_reduction_total_pred` | 116 | 29.310% | 89.655% | 31.897% | 0.1639 | 0.1489 | 6.034% | 0.6114 | 0.7896 | 0.4686 | 1/7/108 |
| moe_experts_within_best_final_horizon | `typed_marginal_pred` | 116 | 26.724% | 92.241% | 30.172% | 0.1775 | 0.1546 | 4.310% | 0.6627 | 0.8186 | 0.5272 | 3/3/110 |
| moe_experts_within_best_final_horizon | `typed_return_pred` | 116 | 19.828% | 79.310% | 27.586% | 0.2160 | 0.1840 | 9.483% | 0.5253 | 0.7559 | 0.5227 | 15/7/94 |
| moe_experts_within_best_final_horizon | `typed_sa_reduction_total_pred` | 116 | 28.448% | 89.655% | 31.034% | 0.1639 | 0.1489 | 6.034% | 0.6114 | 0.7896 | 0.4686 | 1/7/108 |
| moe_experts_within_epoch_010 | `typed_marginal_pred` | 116 | 27.586% | 92.241% | 31.034% | 0.1775 | 0.1546 | 4.310% | 0.6628 | 0.8186 | 0.5272 | 3/3/110 |
| moe_experts_within_epoch_010 | `typed_return_pred` | 116 | 20.690% | 79.310% | 28.448% | 0.2160 | 0.1840 | 9.483% | 0.5253 | 0.7559 | 0.5227 | 15/7/94 |
| moe_experts_within_epoch_010 | `typed_sa_reduction_total_pred` | 116 | 29.310% | 89.655% | 31.897% | 0.1639 | 0.1489 | 6.034% | 0.6115 | 0.7896 | 0.4686 | 1/7/108 |
| moe_joint_within_best_final_horizon | `typed_marginal_pred` | 116 | 31.897% | 91.379% | 37.069% | 0.1843 | 0.1471 | 5.172% | 0.6783 | 0.8280 | 0.5544 | 2/4/110 |
| moe_joint_within_best_final_horizon | `typed_return_pred` | 116 | 19.828% | 80.172% | 27.586% | 0.2142 | 0.1840 | 8.621% | 0.5310 | 0.7591 | 0.5174 | 14/7/95 |
| moe_joint_within_best_final_horizon | `typed_sa_reduction_total_pred` | 116 | 30.172% | 88.793% | 33.621% | 0.1449 | 0.1185 | 6.897% | 0.5976 | 0.7803 | 0.4510 | 1/8/107 |
| moe_joint_within_epoch_010 | `typed_marginal_pred` | 116 | 20.690% | 87.931% | 27.586% | 0.1849 | 0.1470 | 5.172% | 0.6753 | 0.8284 | 0.5697 | 3/7/106 |
| moe_joint_within_epoch_010 | `typed_return_pred` | 116 | 20.690% | 80.172% | 27.586% | 0.2155 | 0.1840 | 8.621% | 0.5244 | 0.7562 | 0.5180 | 14/7/95 |
| moe_joint_within_epoch_010 | `typed_sa_reduction_total_pred` | 116 | 31.034% | 88.793% | 34.483% | 0.1490 | 0.1226 | 6.897% | 0.5986 | 0.7820 | 0.4557 | 1/8/107 |
