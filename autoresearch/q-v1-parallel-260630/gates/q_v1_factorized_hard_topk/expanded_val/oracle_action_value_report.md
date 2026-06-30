# Oracle Action-Value Checkpoint Gate

generated_at: `2026-06-30T12:23:04`

baseline: `incumbent`

## Verdicts

| checkpoint | verdict |
|---|---|
| `B_oracle_0p05` | `PROMOTE` |
| `incumbent` | `INCONCLUSIVE` |
| `q_v1_factorized_hard_topk` | `INCONCLUSIVE` |

## Summary

| checkpoint | score_field | groups | mean Spearman | negative top1 rate | mean top1 regret | verdict |
|---|---|---:|---:|---:|---:|---|
| `B_oracle_0p05` | `derived_hard_reduction_hybrid_pred` | 37 | 0.424585 | 0.162162 | 0.0193024 | `PROMOTE` |
| `B_oracle_0p05` | `hybrid_pred` | 37 | -0.218089 | 0.702703 | 0.0500473 | `PROMOTE` |
| `B_oracle_0p05` | `q_pred` | 37 | 0.103686 | 0.405405 | 0.0427381 | `PROMOTE` |
| `B_oracle_0p05` | `score_pred` | 37 | 0.103686 | 0.405405 | 0.0427381 | `PROMOTE` |
| `incumbent` | `derived_hard_reduction_hybrid_pred` | 37 | 0.0802077 | 0.621622 | 0.0212805 | `INCONCLUSIVE` |
| `incumbent` | `hybrid_pred` | 37 | 0.0785791 | 0.486486 | 0.0408516 | `INCONCLUSIVE` |
| `incumbent` | `q_pred` | 37 | 0.139985 | 0.351351 | 0.0363708 | `INCONCLUSIVE` |
| `incumbent` | `score_pred` | 37 | 0.139985 | 0.351351 | 0.0363708 | `INCONCLUSIVE` |
| `q_v1_factorized_hard_topk` | `derived_hard_reduction_hybrid_pred` | 37 | -0.211971 | 0.864865 | 0.0434824 | `INCONCLUSIVE` |
| `q_v1_factorized_hard_topk` | `hybrid_pred` | 37 | -0.00990297 | 0.594595 | 0.0414608 | `INCONCLUSIVE` |
| `q_v1_factorized_hard_topk` | `q_pred` | 37 | 0.413918 | 0.243243 | 0.0291119 | `INCONCLUSIVE` |
| `q_v1_factorized_hard_topk` | `score_pred` | 37 | 0.413918 | 0.243243 | 0.0291119 | `INCONCLUSIVE` |
