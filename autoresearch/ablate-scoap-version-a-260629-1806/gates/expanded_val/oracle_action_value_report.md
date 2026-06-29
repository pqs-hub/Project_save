# Oracle Action-Value Checkpoint Gate

generated_at: `2026-06-29T18:14:52`

baseline: `A_base`

## Verdicts

| checkpoint | verdict |
|---|---|
| `A_base` | `INCONCLUSIVE` |
| `A_only_delta_scoap` | `INCONCLUSIVE` |
| `A_only_scoap` | `INCONCLUSIVE` |

## Summary

| checkpoint | score_field | groups | mean Spearman | negative top1 rate | mean top1 regret | verdict |
|---|---|---:|---:|---:|---:|---|
| `A_base` | `hard_reduction_total_pred` | 48 | -0.036404 | 0.5625 | 0.0303087 | `INCONCLUSIVE` |
| `A_base` | `hybrid_pred` | 48 | -0.0404847 | 0.5625 | 0.0301433 | `INCONCLUSIVE` |
| `A_only_delta_scoap` | `hard_reduction_total_pred` | 48 | -0.0975985 | 0.541667 | 0.0289992 | `INCONCLUSIVE` |
| `A_only_delta_scoap` | `hybrid_pred` | 48 | -0.104651 | 0.541667 | 0.0289992 | `INCONCLUSIVE` |
| `A_only_scoap` | `hard_reduction_total_pred` | 48 | 0.102406 | 0.270833 | 0.0216556 | `INCONCLUSIVE` |
| `A_only_scoap` | `hybrid_pred` | 48 | 0.0783979 | 0.270833 | 0.0215867 | `INCONCLUSIVE` |
