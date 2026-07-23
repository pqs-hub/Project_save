# Prefix-oracle ranking audit

This fixed-label diagnostic did not run ATPG and was not used for target-circuit selection.

| checkpoint | score | groups | top-1 | type | regret (pp) | negative | Spearman | pairwise | CP0/CP1/OP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| round4 | `typed_marginal_pred` | 192 | 5.729% | 31.771% | 0.7970 | 29.167% | 0.0334 | 0.5199 | 62/61/69 |
| round4 | `typed_return_pred` | 192 | 11.979% | 44.792% | 0.6226 | 21.354% | 0.1630 | 0.5885 | 42/49/101 |
| round4 | `typed_sa_reduction_total_pred` | 192 | 9.375% | 40.625% | 0.6546 | 8.333% | 0.2240 | 0.6161 | 40/37/115 |
| prefix_rank_best | `typed_marginal_pred` | 192 | 15.104% | 57.812% | 0.5610 | 8.333% | 0.3852 | 0.6884 | 11/21/160 |
| prefix_rank_best | `typed_return_pred` | 192 | 12.500% | 44.271% | 0.6076 | 19.792% | 0.2262 | 0.6163 | 35/45/112 |
| prefix_rank_best | `typed_sa_reduction_total_pred` | 192 | 4.688% | 25.000% | 0.7971 | 28.125% | -0.1077 | 0.4529 | 119/38/35 |
| prefix_rank_e12 | `typed_marginal_pred` | 192 | 15.104% | 58.854% | 0.5596 | 7.812% | 0.3968 | 0.6940 | 10/19/163 |
| prefix_rank_e12 | `typed_return_pred` | 192 | 13.021% | 46.354% | 0.6087 | 18.750% | 0.2293 | 0.6169 | 32/42/118 |
| prefix_rank_e12 | `typed_sa_reduction_total_pred` | 192 | 4.688% | 24.479% | 0.7979 | 29.167% | -0.1194 | 0.4472 | 120/40/32 |
| prefix_rank_sa_best | `typed_marginal_pred` | 192 | 14.062% | 57.812% | 0.5628 | 8.854% | 0.3853 | 0.6883 | 12/21/159 |
| prefix_rank_sa_best | `typed_return_pred` | 192 | 12.500% | 44.792% | 0.6063 | 19.271% | 0.2268 | 0.6162 | 35/44/113 |
| prefix_rank_sa_best | `typed_sa_reduction_total_pred` | 192 | 5.729% | 19.792% | 0.8817 | 34.896% | -0.1422 | 0.4379 | 154/22/16 |
| prefix_rank_sa_e12 | `typed_marginal_pred` | 192 | 14.583% | 58.854% | 0.5596 | 7.812% | 0.3964 | 0.6936 | 11/19/162 |
| prefix_rank_sa_e12 | `typed_return_pred` | 192 | 13.021% | 46.354% | 0.6087 | 18.750% | 0.2302 | 0.6174 | 32/42/118 |
| prefix_rank_sa_e12 | `typed_sa_reduction_total_pred` | 192 | 5.208% | 18.229% | 0.9187 | 35.938% | -0.1657 | 0.4277 | 163/16/13 |
| prefix_cql_sa_best | `typed_marginal_pred` | 192 | 16.146% | 59.375% | 0.5571 | 7.812% | 0.4397 | 0.7184 | 9/19/164 |
| prefix_cql_sa_best | `typed_return_pred` | 192 | 13.021% | 47.396% | 0.6100 | 17.708% | 0.2475 | 0.6262 | 28/42/122 |
| prefix_cql_sa_best | `typed_sa_reduction_total_pred` | 192 | 4.688% | 18.229% | 0.9048 | 39.062% | -0.1820 | 0.4159 | 167/11/14 |
| prefix_cql_sa_e12 | `typed_marginal_pred` | 192 | 17.188% | 61.458% | 0.4850 | 6.771% | 0.4529 | 0.7221 | 5/18/169 |
| prefix_cql_sa_e12 | `typed_return_pred` | 192 | 13.021% | 46.875% | 0.5913 | 17.188% | 0.2541 | 0.6292 | 25/41/126 |
| prefix_cql_sa_e12 | `typed_sa_reduction_total_pred` | 192 | 4.688% | 16.667% | 0.9118 | 38.542% | -0.1924 | 0.4135 | 163/14/15 |
| prefix_toplist_sa_best | `typed_marginal_pred` | 192 | 18.229% | 61.979% | 0.4771 | 6.771% | 0.4684 | 0.7308 | 4/18/170 |
| prefix_toplist_sa_best | `typed_return_pred` | 192 | 13.021% | 47.396% | 0.5858 | 16.146% | 0.2739 | 0.6391 | 24/39/129 |
| prefix_toplist_sa_best | `typed_sa_reduction_total_pred` | 192 | 6.250% | 20.312% | 0.8961 | 36.979% | -0.1555 | 0.4323 | 156/10/26 |
| prefix_toplist_sa_e12 | `typed_marginal_pred` | 192 | 17.188% | 62.500% | 0.4777 | 6.771% | 0.4780 | 0.7359 | 4/17/171 |
| prefix_toplist_sa_e12 | `typed_return_pred` | 192 | 11.458% | 47.917% | 0.5912 | 16.667% | 0.2821 | 0.6414 | 24/38/130 |
| prefix_toplist_sa_e12 | `typed_sa_reduction_total_pred` | 192 | 6.250% | 19.792% | 0.9205 | 38.021% | -0.1680 | 0.4264 | 153/12/27 |
