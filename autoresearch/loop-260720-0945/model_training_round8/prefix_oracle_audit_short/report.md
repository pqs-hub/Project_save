# Prefix-oracle ranking audit

This fixed-label diagnostic did not run ATPG and was not used for target-circuit selection.

| checkpoint | score | groups | top-1 | type | within-type top-1 | regret (pp) | within regret | negative | Spearman | pairwise | same-type pairwise | CP0/CP1/OP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| round7_incumbent | `typed_marginal_pred` | 192 | 24.479% | 64.062% | 36.979% | 0.1735 | 0.1151 | 4.167% | 0.5346 | 0.7678 | 0.6355 | 9/56/127 |
| round7_incumbent | `typed_return_pred` | 192 | 8.333% | 48.958% | 28.646% | 0.6667 | 0.2638 | 11.458% | 0.3204 | 0.6632 | 0.5822 | 33/34/125 |
| round7_incumbent | `typed_sa_reduction_total_pred` | 192 | 20.312% | 59.896% | 31.771% | 0.5593 | 0.4271 | 3.646% | 0.4327 | 0.7096 | 0.4890 | 2/4/186 |
| moe_experts_within_best_final_horizon | `typed_marginal_pred` | 192 | 24.479% | 64.062% | 36.458% | 0.1718 | 0.1151 | 3.646% | 0.5366 | 0.7687 | 0.6341 | 10/54/128 |
| moe_experts_within_best_final_horizon | `typed_return_pred` | 192 | 8.854% | 49.479% | 29.688% | 0.6658 | 0.2626 | 10.938% | 0.3297 | 0.6681 | 0.5879 | 32/34/126 |
| moe_experts_within_best_final_horizon | `typed_sa_reduction_total_pred` | 192 | 20.833% | 59.896% | 32.292% | 0.5593 | 0.4271 | 3.646% | 0.4327 | 0.7096 | 0.4890 | 2/4/186 |
| moe_experts_within_epoch_010 | `typed_marginal_pred` | 192 | 23.958% | 64.062% | 35.938% | 0.1718 | 0.1151 | 3.646% | 0.5366 | 0.7687 | 0.6341 | 10/54/128 |
| moe_experts_within_epoch_010 | `typed_return_pred` | 192 | 8.854% | 49.479% | 29.167% | 0.6658 | 0.2626 | 10.938% | 0.3296 | 0.6681 | 0.5879 | 32/34/126 |
| moe_experts_within_epoch_010 | `typed_sa_reduction_total_pred` | 192 | 20.312% | 59.896% | 32.292% | 0.5593 | 0.4271 | 3.646% | 0.4327 | 0.7096 | 0.4890 | 2/4/186 |
| moe_joint_within_best_final_horizon | `typed_marginal_pred` | 192 | 24.479% | 63.021% | 39.062% | 0.1911 | 0.1172 | 4.167% | 0.5346 | 0.7713 | 0.6510 | 11/50/131 |
| moe_joint_within_best_final_horizon | `typed_return_pred` | 192 | 6.250% | 50.000% | 27.604% | 0.6149 | 0.2608 | 12.500% | 0.3476 | 0.6763 | 0.5944 | 30/34/128 |
| moe_joint_within_best_final_horizon | `typed_sa_reduction_total_pred` | 192 | 22.917% | 59.896% | 34.896% | 0.5066 | 0.3871 | 3.646% | 0.4293 | 0.7084 | 0.4814 | 2/4/186 |
| moe_joint_within_epoch_010 | `typed_marginal_pred` | 192 | 20.833% | 58.333% | 36.458% | 0.2379 | 0.1494 | 6.250% | 0.5095 | 0.7562 | 0.6512 | 13/66/113 |
| moe_joint_within_epoch_010 | `typed_return_pred` | 192 | 6.250% | 48.958% | 27.083% | 0.6127 | 0.2608 | 11.458% | 0.3436 | 0.6739 | 0.5963 | 29/37/126 |
| moe_joint_within_epoch_010 | `typed_sa_reduction_total_pred` | 192 | 25.000% | 60.417% | 35.938% | 0.5192 | 0.4156 | 3.646% | 0.4223 | 0.7055 | 0.4755 | 3/4/185 |
