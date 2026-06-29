# Oracle Action-Value Checkpoint Gate

generated_at: `2026-06-29T17:26:05`

baseline: `B_base`

## Verdicts

| checkpoint | verdict |
|---|---|
| `B_base` | `INCONCLUSIVE` |
| `B_only_delta_scoap` | `INCONCLUSIVE` |
| `B_only_scoap` | `PROMOTE` |

## Summary

| checkpoint | score_field | groups | mean Spearman | negative top1 rate | mean top1 regret | verdict |
|---|---|---:|---:|---:|---:|---|
| `B_base` | `derived_hard_reduction_hybrid_pred` | 48 | -0.143447 | 0.583333 | 0.0381487 | `INCONCLUSIVE` |
| `B_base` | `derived_hard_reduction_total_pred` | 48 | -0.143447 | 0.583333 | 0.0381487 | `INCONCLUSIVE` |
| `B_only_delta_scoap` | `derived_hard_reduction_hybrid_pred` | 48 | 0.016842 | 0.458333 | 0.0319625 | `INCONCLUSIVE` |
| `B_only_delta_scoap` | `derived_hard_reduction_total_pred` | 48 | 0.016842 | 0.458333 | 0.0319625 | `INCONCLUSIVE` |
| `B_only_scoap` | `derived_hard_reduction_hybrid_pred` | 48 | 0.0943941 | 0.354167 | 0.0279498 | `PROMOTE` |
| `B_only_scoap` | `derived_hard_reduction_total_pred` | 48 | 0.0943941 | 0.354167 | 0.0279498 | `PROMOTE` |
