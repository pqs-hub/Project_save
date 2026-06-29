# Oracle Action Group Audit

generated_at: `2026-06-29T19:32:38`

## Split Summary

| split | rows | groups | negative rate | all-positive groups | mean delta | min delta | max delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| topup | 20736 | 576 | 0.0312 | 78 | -0.000002 | -0.051520 | 0.288890 |

## Action Type Summary

| split | type | rows | negative rate | mean delta |
|---|---|---:|---:|---:|
| topup | `control0` | 6912 | 0.0437 | -0.000409 |
| topup | `control1` | 6912 | 0.0460 | -0.000436 |
| topup | `observe` | 6912 | 0.0038 | 0.000839 |

## Group Negative Count Histogram

| split | negative actions in group | groups |
|---|---:|---:|
| topup | 0 | 498 |
| topup | 1 | 22 |
| topup | 2 | 17 |
| topup | 3 | 1 |
| topup | 4 | 6 |
| topup | 5 | 1 |
| topup | 6 | 2 |
| topup | 8 | 4 |
| topup | 9 | 1 |
| topup | 10 | 1 |
| topup | 11 | 1 |
| topup | 12 | 1 |
| topup | 16 | 1 |
| topup | 18 | 1 |
| topup | 19 | 1 |
| topup | 20 | 1 |
| topup | 21 | 1 |
| topup | 22 | 3 |
| topup | 23 | 3 |
| topup | 24 | 7 |
| topup | 25 | 3 |

## Recommendation

- Balance by group, not only by rows.
- Require both positive and negative actions inside a group for rank training.
- Prefer negative `control0` / `control1` examples because transfer failures concentrate there.
- Keep transfer evaluation-only; do not train on `b15_C` or `i2c_aig` transfer rows.
