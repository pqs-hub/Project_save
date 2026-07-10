# Oracle Action-Value Checkpoint Gate

generated_at: `2026-06-30T18:59:35`

baseline: `incumbent`

## Verdicts

| checkpoint | verdict |
|---|---|
| `B_oracle_0p05` | `PROMOTE` |
| `incumbent` | `INCONCLUSIVE` |
| `q_v2_pairwise_listwise_hardred` | `INCONCLUSIVE` |

## Summary

| checkpoint | score_field | groups | mean Spearman | negative top1 rate | mean top1 regret | verdict |
|---|---|---:|---:|---:|---:|---|
| `B_oracle_0p05` | `derived_hard_reduction_hybrid_pred` | 37 | 0.424585 | 0.162162 | 0.0193024 | `PROMOTE` |
| `B_oracle_0p05` | `hybrid_pred` | 37 | -0.219102 | 0.702703 | 0.0500473 | `PROMOTE` |
| `B_oracle_0p05` | `q_pred` | 37 | -0.167415 | 0.621622 | 0.0377808 | `PROMOTE` |
| `B_oracle_0p05` | `score_pred` | 37 | -0.167415 | 0.621622 | 0.0377808 | `PROMOTE` |
| `incumbent` | `derived_hard_reduction_hybrid_pred` | 37 | 0.0802077 | 0.621622 | 0.0212805 | `INCONCLUSIVE` |
| `incumbent` | `hybrid_pred` | 37 | 0.0785791 | 0.486486 | 0.0408516 | `INCONCLUSIVE` |
| `incumbent` | `q_pred` | 37 | 0.153704 | 0.351351 | 0.0297049 | `INCONCLUSIVE` |
| `incumbent` | `score_pred` | 37 | 0.153704 | 0.351351 | 0.0297049 | `INCONCLUSIVE` |
| `q_v2_pairwise_listwise_hardred` | `derived_hard_reduction_hybrid_pred` | 37 | -0.210196 | 0.648649 | 0.0423586 | `INCONCLUSIVE` |
| `q_v2_pairwise_listwise_hardred` | `hybrid_pred` | 37 | 0.290399 | 0.216216 | 0.0136105 | `INCONCLUSIVE` |
| `q_v2_pairwise_listwise_hardred` | `q_pred` | 37 | 0.462178 | 0.216216 | 0.0217532 | `INCONCLUSIVE` |
| `q_v2_pairwise_listwise_hardred` | `score_pred` | 37 | 0.462178 | 0.216216 | 0.0217532 | `INCONCLUSIVE` |
