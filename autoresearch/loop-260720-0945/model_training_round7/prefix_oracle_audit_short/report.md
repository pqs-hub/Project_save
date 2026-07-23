# Prefix-oracle ranking audit

This fixed-label diagnostic did not run ATPG and was not used for target-circuit selection.

| checkpoint | score | groups | top-1 | type | regret (pp) | negative | Spearman | pairwise | CP0/CP1/OP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cone_long_rank_best_final_horizon | `typed_marginal_pred` | 192 | 24.479% | 64.062% | 0.1735 | 4.167% | 0.5345 | 0.7678 | 9/56/127 |
| cone_long_rank_best_final_horizon | `typed_return_pred` | 192 | 8.333% | 48.958% | 0.6667 | 11.458% | 0.3204 | 0.6632 | 33/34/125 |
| cone_long_rank_best_final_horizon | `typed_sa_reduction_total_pred` | 192 | 20.312% | 59.896% | 0.5593 | 3.646% | 0.4328 | 0.7096 | 2/4/186 |
| cone_long_rank_epoch_010 | `typed_marginal_pred` | 192 | 26.562% | 65.104% | 0.1746 | 4.167% | 0.5425 | 0.7724 | 10/57/125 |
| cone_long_rank_epoch_010 | `typed_return_pred` | 192 | 8.333% | 50.521% | 0.6647 | 10.938% | 0.3479 | 0.6772 | 32/30/130 |
| cone_long_rank_epoch_010 | `typed_sa_reduction_total_pred` | 192 | 23.958% | 59.375% | 0.5053 | 3.125% | 0.4309 | 0.7095 | 3/4/185 |
| cone_long_toplist_best_final_horizon | `typed_marginal_pred` | 192 | 25.521% | 64.583% | 0.1569 | 4.688% | 0.5103 | 0.7550 | 11/61/120 |
| cone_long_toplist_best_final_horizon | `typed_return_pred` | 192 | 8.854% | 49.479% | 0.6503 | 10.938% | 0.3358 | 0.6701 | 33/37/122 |
| cone_long_toplist_best_final_horizon | `typed_sa_reduction_total_pred` | 192 | 22.917% | 60.938% | 0.5863 | 5.208% | 0.4358 | 0.7112 | 0/3/189 |
| cone_long_toplist_epoch_010 | `typed_marginal_pred` | 192 | 26.042% | 64.583% | 0.1680 | 5.208% | 0.5222 | 0.7593 | 10/59/123 |
| cone_long_toplist_epoch_010 | `typed_return_pred` | 192 | 8.854% | 52.083% | 0.6635 | 9.896% | 0.3595 | 0.6815 | 27/30/135 |
| cone_long_toplist_epoch_010 | `typed_sa_reduction_total_pred` | 192 | 22.917% | 60.938% | 0.5276 | 3.125% | 0.4308 | 0.7086 | 1/3/188 |
