# Balanced Oracle Action Groups

generated_at: `2026-06-29T18:49:14`

## Summary

| split | kept groups | kept rows | negative rate | group target | neg-rate target |
|---|---:|---:|---:|---:|---:|
| train | 71 | 1278 | 0.5008 | 0 | 1 |
| expanded_val | 34 | 612 | 0.5801 | 1 | 1 |
| transfer_eval_only | 0 | 0 | 0.3403 | eval_only | eval_only |

## Decision

- train minimum target: `80` groups
- val minimum target: `24` groups
- needs_more_oracle_collection: `true`

Existing labels are not enough for the balanced-group target; collect negative-rich oracle groups next.
