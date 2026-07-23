# Prefix-oracle ranking audit

This fixed-label diagnostic did not run ATPG and was not used for target-circuit selection.

| checkpoint | score | groups | top-1 | type | regret (pp) | negative | Spearman | pairwise | CP0/CP1/OP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| type_balanced_rank_best_final_horizon | `typed_marginal_pred` | 192 | 24.479% | 65.104% | 0.2016 | 5.729% | 0.5382 | 0.7721 | 7/49/136 |
| type_balanced_rank_best_final_horizon | `typed_return_pred` | 192 | 10.417% | 48.958% | 0.6602 | 12.500% | 0.2754 | 0.6402 | 39/32/121 |
| type_balanced_rank_best_final_horizon | `typed_sa_reduction_total_pred` | 192 | 19.271% | 60.417% | 0.5782 | 4.167% | 0.4367 | 0.7131 | 1/3/188 |
| type_balanced_rank_epoch_008 | `typed_marginal_pred` | 192 | 25.000% | 66.667% | 0.1633 | 5.208% | 0.5470 | 0.7763 | 6/53/133 |
| type_balanced_rank_epoch_008 | `typed_return_pred` | 192 | 10.417% | 50.000% | 0.6540 | 11.458% | 0.2878 | 0.6471 | 38/31/123 |
| type_balanced_rank_epoch_008 | `typed_sa_reduction_total_pred` | 192 | 20.312% | 59.896% | 0.5718 | 4.167% | 0.4323 | 0.7101 | 2/3/187 |
| type_balanced_toplist_best_final_horizon | `typed_marginal_pred` | 192 | 25.000% | 66.667% | 0.1630 | 4.688% | 0.5360 | 0.7685 | 6/52/134 |
| type_balanced_toplist_best_final_horizon | `typed_return_pred` | 192 | 11.458% | 50.521% | 0.6464 | 10.417% | 0.2965 | 0.6504 | 38/30/124 |
| type_balanced_toplist_best_final_horizon | `typed_sa_reduction_total_pred` | 192 | 19.271% | 60.938% | 0.5892 | 6.250% | 0.4268 | 0.7064 | 0/3/189 |
| type_balanced_toplist_epoch_008 | `typed_marginal_pred` | 192 | 26.562% | 67.708% | 0.1407 | 3.125% | 0.5370 | 0.7680 | 5/55/132 |
| type_balanced_toplist_epoch_008 | `typed_return_pred` | 192 | 11.458% | 52.604% | 0.6396 | 8.854% | 0.3083 | 0.6577 | 35/29/128 |
| type_balanced_toplist_epoch_008 | `typed_sa_reduction_total_pred` | 192 | 17.188% | 60.417% | 0.5924 | 6.250% | 0.4264 | 0.7066 | 0/4/188 |
