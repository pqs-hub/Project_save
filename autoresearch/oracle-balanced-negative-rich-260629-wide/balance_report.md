# Balanced Oracle Action Groups

generated_at: `2026-06-29T20:38:58`

## Summary

| split | kept groups | kept rows | negative rate | group target | neg-rate target |
|---|---:|---:|---:|---:|---:|
| train | 180 | 4320 | 0.2880 | 1 | 1 |
| expanded_val | 37 | 666 | 0.5390 | 1 | 1 |
| transfer_eval_only | 0 | 0 | 0.3403 | eval_only | eval_only |

## Decision

- train minimum target: `120` groups
- val minimum target: `24` groups
- needs_more_oracle_collection: `false`

Existing labels meet the group-count target; rescore these balanced subsets before rerunning the ranker.
