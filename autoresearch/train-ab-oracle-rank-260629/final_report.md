# Scheme A/B Oracle-Ranking Main-Model Experiment

generated_at: `2026-06-29T22:03:49`

## Summary

| variant | verdict | score | expanded Spearman | expanded neg top1 | transfer Spearman | transfer neg top1 | transfer top1 delta | transfer regret | hard F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| incumbent | BASELINE | `hybrid_pred` | 0.078579 | 0.486486 | 0.326661 | 0.166667 | 0.010638 | 0.012552 | nan |
| A_no_oracle | BASELINE | `hard_reduction_total_pred` | 0.129149 | 0.351351 | -0.010474 | 0.500000 | -0.004248 | 0.027438 | nan |
| B_no_oracle | BASELINE | `derived_hard_reduction_hybrid_pred` | -0.005741 | 0.594595 | 0.134791 | 0.333333 | 0.005812 | 0.017378 | nan |
| A_oracle_0p01 | REJECT | `hard_reduction_total_pred` | 0.074083 | 0.621622 | -0.092248 | 0.500000 | 0.001635 | 0.021555 | 0.811399 |
| A_oracle_0p03 | REJECT | `hard_reduction_total_pred` | -0.002181 | 0.486486 | 0.098460 | 0.000000 | 0.015928 | 0.007262 | 0.761439 |
| A_oracle_0p05 | REJECT | `hard_reduction_total_pred` | 0.015609 | 0.486486 | 0.154136 | 0.500000 | 0.004125 | 0.019065 | 0.818108 |
| B_oracle_0p01 | REJECT | `derived_hard_reduction_hybrid_pred` | 0.244340 | 0.216216 | 0.000919 | 0.000000 | 0.000582 | 0.022608 | 0.804093 |
| B_oracle_0p03 | REJECT | `derived_hard_reduction_hybrid_pred` | 0.336100 | 0.297297 | 0.074052 | 0.333333 | 0.000698 | 0.022492 | 0.805229 |
| B_oracle_0p05 | PROMOTE_GUARDED_RERANK | `derived_hard_reduction_hybrid_pred` | 0.424585 | 0.162162 | 0.081119 | 0.000000 | 0.005678 | 0.017512 | 0.769712 |

## Notes

- `A_no_oracle` and `B_no_oracle` are prior no-oracle baselines.
- `incumbent` uses `hybrid_pred`.
