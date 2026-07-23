# Prefix-oracle ranking audit

This fixed-label diagnostic did not run ATPG and was not used for target-circuit selection.

| checkpoint | score | groups | top-1 | type | within-type top-1 | regret (pp) | within regret | negative | Spearman | pairwise | same-type pairwise | CP0/CP1/OP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| round8_incumbent | `typed_return_pred` | 90 | 26.667% | 87.778% | 27.778% | 0.1919 | 0.2035 | 14.444% | 0.5617 | 0.7708 | 0.5244 | 8/5/77 |
| round10_b15_selected | `typed_return_pred` | 90 | 22.222% | 86.667% | 25.556% | 0.2258 | 0.1951 | 16.667% | 0.5652 | 0.7599 | 0.5352 | 2/10/78 |
| return_pairwise_expanded_e006 | `typed_return_pred` | 90 | 26.667% | 86.667% | 27.778% | 0.1919 | 0.2035 | 14.444% | 0.5622 | 0.7710 | 0.5221 | 7/6/77 |
| return_pairwise_expanded_e008 | `typed_return_pred` | 90 | 27.778% | 86.667% | 28.889% | 0.1925 | 0.2031 | 14.444% | 0.5643 | 0.7718 | 0.5207 | 7/6/77 |
| return_pairwise_expanded_e010 | `typed_return_pred` | 90 | 28.889% | 86.667% | 31.111% | 0.1925 | 0.1998 | 14.444% | 0.5648 | 0.7712 | 0.5220 | 7/6/77 |
| return_hybrid_listwise_e006 | `typed_return_pred` | 90 | 27.778% | 86.667% | 28.889% | 0.1915 | 0.2031 | 14.444% | 0.5659 | 0.7731 | 0.5221 | 7/6/77 |
| return_hybrid_listwise_e008 | `typed_return_pred` | 90 | 27.778% | 86.667% | 30.000% | 0.1925 | 0.1998 | 14.444% | 0.5639 | 0.7711 | 0.5214 | 7/6/77 |
| return_hybrid_listwise_e010 | `typed_return_pred` | 90 | 28.889% | 86.667% | 31.111% | 0.1925 | 0.1998 | 14.444% | 0.5642 | 0.7707 | 0.5225 | 7/6/77 |
| return_top_listwise_e006 | `typed_return_pred` | 90 | 27.778% | 86.667% | 28.889% | 0.1915 | 0.2031 | 14.444% | 0.5656 | 0.7725 | 0.5249 | 7/6/77 |
| return_top_listwise_e008 | `typed_return_pred` | 90 | 27.778% | 86.667% | 30.000% | 0.1915 | 0.1998 | 14.444% | 0.5634 | 0.7707 | 0.5243 | 7/6/77 |
| return_top_listwise_e010 | `typed_return_pred` | 90 | 28.889% | 86.667% | 31.111% | 0.1925 | 0.1998 | 14.444% | 0.5615 | 0.7691 | 0.5226 | 7/6/77 |
