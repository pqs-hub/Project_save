# Prefix-oracle ranking audit

This fixed-label diagnostic did not run ATPG and was not used for target-circuit selection.

| checkpoint | score | groups | top-1 | type | within-type top-1 | regret (pp) | within regret | negative | Spearman | pairwise | same-type pairwise | CP0/CP1/OP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| round8_incumbent | `typed_return_pred` | 10 | 10.000% | 70.000% | 10.000% | 0.0503 | 0.0615 | 20.000% | 0.4788 | 0.7145 | 0.4122 | 1/1/8 |
| round10_b15_selected | `typed_return_pred` | 10 | 10.000% | 80.000% | 10.000% | 0.0473 | 0.0615 | 20.000% | 0.4491 | 0.7054 | 0.3985 | 2/0/8 |
| horizon_r8_pairwise_e006 | `typed_return_pred` | 10 | 10.000% | 70.000% | 10.000% | 0.0503 | 0.0615 | 20.000% | 0.4752 | 0.7123 | 0.4175 | 1/1/8 |
| horizon_r8_pairwise_e008 | `typed_return_pred` | 10 | 10.000% | 70.000% | 10.000% | 0.0503 | 0.0615 | 20.000% | 0.4627 | 0.7076 | 0.4045 | 1/1/8 |
| horizon_r8_pairwise_e010 | `typed_return_pred` | 10 | 10.000% | 70.000% | 10.000% | 0.0503 | 0.0615 | 20.000% | 0.4754 | 0.7155 | 0.4075 | 1/1/8 |
| horizon_r8_hybrid_e006 | `typed_return_pred` | 10 | 10.000% | 70.000% | 10.000% | 0.0503 | 0.0615 | 20.000% | 0.4752 | 0.7123 | 0.4175 | 1/1/8 |
| horizon_r8_hybrid_e008 | `typed_return_pred` | 10 | 10.000% | 70.000% | 10.000% | 0.0503 | 0.0615 | 20.000% | 0.4722 | 0.7129 | 0.4045 | 1/1/8 |
| horizon_r8_hybrid_e010 | `typed_return_pred` | 10 | 10.000% | 70.000% | 10.000% | 0.0503 | 0.0615 | 20.000% | 0.4910 | 0.7241 | 0.4127 | 1/1/8 |
| horizon_r10_pairwise_e006 | `typed_return_pred` | 10 | 10.000% | 80.000% | 10.000% | 0.0473 | 0.0615 | 20.000% | 0.4545 | 0.7078 | 0.3985 | 2/0/8 |
| horizon_r10_pairwise_e008 | `typed_return_pred` | 10 | 10.000% | 80.000% | 10.000% | 0.0473 | 0.0615 | 20.000% | 0.4496 | 0.7073 | 0.3951 | 2/0/8 |
| horizon_r10_pairwise_e010 | `typed_return_pred` | 10 | 10.000% | 80.000% | 10.000% | 0.0473 | 0.0615 | 20.000% | 0.4484 | 0.7073 | 0.3951 | 2/0/8 |
| horizon_r10_hybrid_e006 | `typed_return_pred` | 10 | 10.000% | 80.000% | 10.000% | 0.0473 | 0.0615 | 20.000% | 0.4531 | 0.7076 | 0.3951 | 2/0/8 |
| horizon_r10_hybrid_e008 | `typed_return_pred` | 10 | 10.000% | 80.000% | 10.000% | 0.0473 | 0.0615 | 20.000% | 0.4507 | 0.7086 | 0.3951 | 2/0/8 |
| horizon_r10_hybrid_e010 | `typed_return_pred` | 10 | 10.000% | 80.000% | 10.000% | 0.0473 | 0.0615 | 20.000% | 0.4540 | 0.7100 | 0.3951 | 2/0/8 |
