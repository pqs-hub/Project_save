# Prefix-oracle ranking audit

This fixed-label diagnostic did not run ATPG and was not used for target-circuit selection.

| checkpoint | score | groups | top-1 | type | regret (pp) | negative | Spearman | pairwise | CP0/CP1/OP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cone_rank_best_final_horizon | `typed_marginal_pred` | 192 | 20.312% | 66.146% | 0.2199 | 4.167% | 0.5165 | 0.7612 | 0/40/152 |
| cone_rank_best_final_horizon | `typed_return_pred` | 192 | 13.021% | 45.312% | 0.6700 | 14.062% | 0.1914 | 0.5976 | 45/36/111 |
| cone_rank_best_final_horizon | `typed_sa_reduction_total_pred` | 192 | 22.396% | 61.979% | 0.5232 | 4.688% | 0.4441 | 0.7181 | 1/0/191 |
| cone_rank_epoch_012 | `typed_marginal_pred` | 192 | 20.833% | 65.625% | 0.2274 | 5.208% | 0.5549 | 0.7801 | 2/35/155 |
| cone_rank_epoch_012 | `typed_return_pred` | 192 | 12.500% | 51.042% | 0.6731 | 12.500% | 0.2683 | 0.6377 | 35/30/127 |
| cone_rank_epoch_012 | `typed_sa_reduction_total_pred` | 192 | 19.792% | 59.896% | 0.5847 | 5.208% | 0.4351 | 0.7128 | 2/3/187 |
| cone_toplist_best_final_horizon | `typed_marginal_pred` | 192 | 16.667% | 65.625% | 0.2333 | 4.167% | 0.5123 | 0.7554 | 1/33/158 |
| cone_toplist_best_final_horizon | `typed_return_pred` | 192 | 14.583% | 47.917% | 0.6610 | 11.979% | 0.2366 | 0.6185 | 33/37/122 |
| cone_toplist_best_final_horizon | `typed_sa_reduction_total_pred` | 192 | 19.792% | 61.979% | 0.5230 | 5.208% | 0.4503 | 0.7196 | 2/0/190 |
| cone_toplist_epoch_012 | `typed_marginal_pred` | 192 | 19.792% | 66.667% | 0.2282 | 4.167% | 0.5443 | 0.7724 | 1/35/156 |
| cone_toplist_epoch_012 | `typed_return_pred` | 192 | 13.021% | 52.083% | 0.6363 | 9.896% | 0.2948 | 0.6496 | 28/31/133 |
| cone_toplist_epoch_012 | `typed_sa_reduction_total_pred` | 192 | 17.188% | 60.417% | 0.5925 | 6.250% | 0.4312 | 0.7116 | 1/3/188 |
| cone_toplist_sa_best_final_horizon | `typed_marginal_pred` | 192 | 17.708% | 65.625% | 0.2333 | 4.167% | 0.5112 | 0.7549 | 1/33/158 |
| cone_toplist_sa_best_final_horizon | `typed_return_pred` | 192 | 14.583% | 47.917% | 0.6628 | 11.979% | 0.2283 | 0.6142 | 37/36/119 |
| cone_toplist_sa_best_final_horizon | `typed_sa_reduction_total_pred` | 192 | 22.917% | 61.979% | 0.4592 | 4.167% | 0.4698 | 0.7295 | 1/1/190 |
| cone_toplist_sa_epoch_012 | `typed_marginal_pred` | 192 | 19.271% | 66.667% | 0.2318 | 4.167% | 0.5423 | 0.7712 | 1/33/158 |
| cone_toplist_sa_epoch_012 | `typed_return_pred` | 192 | 13.021% | 52.083% | 0.6363 | 9.896% | 0.2923 | 0.6481 | 28/31/133 |
| cone_toplist_sa_epoch_012 | `typed_sa_reduction_total_pred` | 192 | 23.958% | 61.458% | 0.4405 | 4.167% | 0.4588 | 0.7230 | 1/2/189 |
