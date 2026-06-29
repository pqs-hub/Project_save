# Version B SCOAP / Delta-SCOAP Ablation

## Setup

All variants use Version B hard setup: no direct `hard_count`, no direct `hard_reduction`, no FC/return/oracle loss. Only SCOAP weights change.

| variant | lambda_scoap | lambda_delta_scoap | SCOAP MAE | delta-SCOAP MAE | hard F1 | derived score | expanded Spearman | expanded neg top1 | transfer Spearman | transfer neg top1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `B_base` | 0.5 | 0.3 | 0.049371 | 0.005997 | 0.802648 | 0.591637 | -0.143447 | 0.583333 | -0.007391 | 0.500000 |
| `B_only_scoap` | 0.5 | 0.0 | 0.059970 | 0.131037 | 0.814177 | 0.624106 | 0.094394 | 0.354167 | -0.176001 | 0.166667 |
| `B_only_delta_scoap` | 0.0 | 0.3 | 0.431092 | 0.004640 | 0.828005 | 0.645423 | 0.016842 | 0.458333 | 0.134591 | 0.333333 |

## Findings

- `B_only_scoap` is best on expanded validation: Spearman `0.094394`, negative top1 `0.354167`. It beats B_base on the small held-out subckt oracle gate.
- `B_only_delta_scoap` is best on transfer: Spearman `0.134591`, negative top1 `0.333333`. It transfers better to the larger smoke circuits.
- `B_base` is worst among the three on both expanded and transfer derived ranking.
- Removing one of SCOAP/delta-SCOAP helps Version B. Keeping both appears redundant or conflicting for derived action value.
- However neither ablation is ready to promote globally: `B_only_scoap` fails transfer, and `B_only_delta_scoap` is only modest on expanded val.

## Practical Conclusion

For Version B, SCOAP and delta-SCOAP together are not best. If optimizing small-subckt validation, keep only SCOAP. If optimizing transfer, keep only delta-SCOAP. This supports the idea that the two losses are redundant/conflicting.

Next reasonable experiment: repeat the same SCOAP/delta-SCOAP ablation on Version A, because Version A keeps the direct hard_reduction head and is closer to the planner we actually want.
