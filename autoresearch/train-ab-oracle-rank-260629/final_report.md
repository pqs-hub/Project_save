# Scheme A/B Oracle-Ranking Main-Model Experiment

generated_at: `2026-06-29T21:35:22`

## Summary

| variant | verdict | score | expanded Spearman | expanded neg top1 | transfer Spearman | transfer neg top1 | transfer top1 delta | transfer regret | hard F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| incumbent | BASELINE | `hybrid_pred` | 0.078579 | 0.486486 | 0.326508 | 0.166667 | 0.010638 | 0.012552 | nan |
| A_no_oracle | BASELINE | `hard_reduction_total_pred` | 0.129159 | 0.351351 | -0.011152 | 0.500000 | -0.004255 | 0.027445 | nan |
| B_no_oracle | BASELINE | `derived_hard_reduction_hybrid_pred` | -0.005741 | 0.594595 | 0.134708 | 0.333333 | 0.005812 | 0.017378 | nan |
| A_oracle_0p03 | REJECT | `hard_reduction_total_pred` | -0.002035 | 0.459459 | 0.098569 | 0.000000 | 0.015928 | 0.007262 | 0.761439 |
| B_oracle_0p03 | REJECT | `derived_hard_reduction_hybrid_pred` | 0.336100 | 0.297297 | 0.073852 | 0.333333 | 0.000698 | 0.022492 | 0.805229 |

## Notes

- `A_no_oracle` and `B_no_oracle` are prior no-oracle baselines.
- `incumbent` uses `hybrid_pred`.
