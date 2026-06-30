# Oracle Action-Value Checkpoint Gate

generated_at: `2026-06-29T22:01:34`

baseline: `incumbent`

## Verdicts

| checkpoint | verdict |
|---|---|
| `A_no_oracle` | `INCONCLUSIVE` |
| `A_oracle_0p01` | `INCONCLUSIVE` |
| `A_oracle_0p03` | `INCONCLUSIVE` |
| `A_oracle_0p05` | `INCONCLUSIVE` |
| `B_no_oracle` | `PROMOTE` |
| `B_oracle_0p01` | `PROMOTE` |
| `B_oracle_0p03` | `PROMOTE` |
| `B_oracle_0p05` | `PROMOTE` |
| `incumbent` | `INCONCLUSIVE` |

## Summary

| checkpoint | score_field | groups | mean Spearman | negative top1 rate | mean top1 regret | verdict |
|---|---|---:|---:|---:|---:|---|
| `A_no_oracle` | `derived_hard_reduction_hybrid_pred` | 37 | -0.175655 | 0.810811 | 0.04292 | `INCONCLUSIVE` |
| `A_no_oracle` | `derived_hard_reduction_total_pred` | 37 | -0.175655 | 0.810811 | 0.04292 | `INCONCLUSIVE` |
| `A_no_oracle` | `hard_reduction_total_pred` | 37 | 0.129149 | 0.351351 | 0.0233368 | `INCONCLUSIVE` |
| `A_no_oracle` | `hybrid_pred` | 37 | 0.0996899 | 0.351351 | 0.0232473 | `INCONCLUSIVE` |
| `A_oracle_0p01` | `derived_hard_reduction_hybrid_pred` | 37 | 0.153791 | 0.459459 | 0.0240035 | `INCONCLUSIVE` |
| `A_oracle_0p01` | `derived_hard_reduction_total_pred` | 37 | 0.153791 | 0.459459 | 0.0240035 | `INCONCLUSIVE` |
| `A_oracle_0p01` | `hard_reduction_total_pred` | 37 | 0.0740831 | 0.621622 | 0.043257 | `INCONCLUSIVE` |
| `A_oracle_0p01` | `hybrid_pred` | 37 | 0.065595 | 0.648649 | 0.0440349 | `INCONCLUSIVE` |
| `A_oracle_0p03` | `derived_hard_reduction_hybrid_pred` | 37 | -0.301557 | 0.783784 | 0.0406784 | `INCONCLUSIVE` |
| `A_oracle_0p03` | `derived_hard_reduction_total_pred` | 37 | -0.301557 | 0.783784 | 0.0406784 | `INCONCLUSIVE` |
| `A_oracle_0p03` | `hard_reduction_total_pred` | 37 | -0.00218066 | 0.486486 | 0.037473 | `INCONCLUSIVE` |
| `A_oracle_0p03` | `hybrid_pred` | 37 | -0.0128738 | 0.486486 | 0.0377949 | `INCONCLUSIVE` |
| `A_oracle_0p05` | `derived_hard_reduction_hybrid_pred` | 37 | 0.133032 | 0.405405 | 0.0255854 | `INCONCLUSIVE` |
| `A_oracle_0p05` | `derived_hard_reduction_total_pred` | 37 | 0.133032 | 0.405405 | 0.0255854 | `INCONCLUSIVE` |
| `A_oracle_0p05` | `hard_reduction_total_pred` | 37 | 0.0156087 | 0.486486 | 0.0310435 | `INCONCLUSIVE` |
| `A_oracle_0p05` | `hybrid_pred` | 37 | 0.000751894 | 0.513514 | 0.0296116 | `INCONCLUSIVE` |
| `B_no_oracle` | `derived_hard_reduction_hybrid_pred` | 37 | -0.0057408 | 0.594595 | 0.0369008 | `PROMOTE` |
| `B_no_oracle` | `derived_hard_reduction_total_pred` | 37 | -0.0057408 | 0.594595 | 0.0369008 | `PROMOTE` |
| `B_no_oracle` | `hard_reduction_total_pred` | 37 | 0.218268 | 0.243243 | 0.0292386 | `PROMOTE` |
| `B_no_oracle` | `hybrid_pred` | 37 | 0.21925 | 0.216216 | 0.0286686 | `PROMOTE` |
| `B_oracle_0p01` | `derived_hard_reduction_hybrid_pred` | 37 | 0.24434 | 0.216216 | 0.0172773 | `PROMOTE` |
| `B_oracle_0p01` | `derived_hard_reduction_total_pred` | 37 | 0.24434 | 0.216216 | 0.0172773 | `PROMOTE` |
| `B_oracle_0p01` | `hard_reduction_total_pred` | 37 | -0.0249371 | 0.405405 | 0.0234722 | `PROMOTE` |
| `B_oracle_0p01` | `hybrid_pred` | 37 | -0.0242122 | 0.405405 | 0.0234722 | `PROMOTE` |
| `B_oracle_0p03` | `derived_hard_reduction_hybrid_pred` | 37 | 0.3361 | 0.297297 | 0.0201462 | `PROMOTE` |
| `B_oracle_0p03` | `derived_hard_reduction_total_pred` | 37 | 0.3361 | 0.297297 | 0.0201462 | `PROMOTE` |
| `B_oracle_0p03` | `hard_reduction_total_pred` | 37 | -0.087198 | 0.675676 | 0.0399995 | `PROMOTE` |
| `B_oracle_0p03` | `hybrid_pred` | 37 | -0.0917675 | 0.675676 | 0.0399995 | `PROMOTE` |
| `B_oracle_0p05` | `derived_hard_reduction_hybrid_pred` | 37 | 0.424585 | 0.162162 | 0.0193024 | `PROMOTE` |
| `B_oracle_0p05` | `derived_hard_reduction_total_pred` | 37 | 0.424585 | 0.162162 | 0.0193024 | `PROMOTE` |
| `B_oracle_0p05` | `hard_reduction_total_pred` | 37 | -0.216666 | 0.702703 | 0.0503149 | `PROMOTE` |
| `B_oracle_0p05` | `hybrid_pred` | 37 | -0.219102 | 0.702703 | 0.0500473 | `PROMOTE` |
| `incumbent` | `derived_hard_reduction_hybrid_pred` | 37 | 0.0801993 | 0.621622 | 0.0212805 | `INCONCLUSIVE` |
| `incumbent` | `derived_hard_reduction_total_pred` | 37 | 0.0801993 | 0.621622 | 0.0212805 | `INCONCLUSIVE` |
| `incumbent` | `hard_reduction_total_pred` | 37 | 0.0976427 | 0.486486 | 0.0408516 | `INCONCLUSIVE` |
| `incumbent` | `hybrid_pred` | 37 | 0.0785791 | 0.486486 | 0.0408516 | `INCONCLUSIVE` |
