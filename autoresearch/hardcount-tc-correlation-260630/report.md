# Hard Fault Count vs Test Coverage Correlation

## Data

- main labels: `/data4/pengqingsong/DFT/Dataset/atalanta_bist_lowtc_subckt_100k_labels/labels.csv`
- oracle train: `autoresearch/oracle-balanced-negative-rich-260629-wide/balanced_train_oracle_actions.tsv`
- oracle val: `autoresearch/oracle-balanced-negative-rich-260629-wide/balanced_val_oracle_actions.tsv`
- oracle transfer: `autoresearch/oracle-action-probe-260629-smoke/oracle_actions.tsv`
- oracle expanded val: `autoresearch/oracle-action-probe-260629-expanded-subckt-val/oracle_actions.tsv`

## Key Result

Absolute hard/undetected fault count is strongly negatively correlated with absolute test coverage.

| dataset | x | y | n | Pearson | Spearman |
|---|---|---|---:|---:|---:|
| main_labels_100k | undetected_fault_count | test_coverage | 100131 | -0.876233 | -0.853907 |
| oracle_balanced_train_wide | oracle_hard_fault_count | oracle_test_coverage | 4320 | -0.852980 | -0.917237 |
| oracle_balanced_val_wide | oracle_hard_fault_count | oracle_test_coverage | 666 | -0.909059 | -0.794907 |
| oracle_transfer_smoke | oracle_hard_fault_count | oracle_test_coverage | 288 | -0.978114 | -0.999492 |
| oracle_expanded_val | oracle_hard_fault_count | oracle_test_coverage | 864 | -0.918978 | -0.859626 |

This is mostly a definition-level relationship:

`test_coverage = detected_faults / total_faults`

and hard/undetected count is approximately:

`undetected_faults = total_faults - detected_faults`

So higher hard/undetected count should imply lower absolute test coverage.

## More Important For Action Ranking

Absolute hard count is not a stable proxy for action value.

| dataset | x | y | n | Pearson | Spearman |
|---|---|---|---:|---:|---:|
| main_labels_100k | undetected_fault_count | delta_test_coverage | 100131 | -0.003590 | -0.016215 |
| oracle_balanced_train_wide | oracle_hard_fault_count | oracle_delta_tc | 4320 | 0.056217 | 0.083744 |
| oracle_balanced_val_wide | oracle_hard_fault_count | oracle_delta_tc | 666 | -0.675845 | -0.403255 |
| oracle_transfer_smoke | oracle_hard_fault_count | oracle_delta_tc | 288 | 0.752561 | 0.388787 |
| oracle_expanded_val | oracle_hard_fault_count | oracle_delta_tc | 864 | -0.615617 | -0.307119 |

The direction changes across splits, so absolute hard count alone should not be used as the action-value target.

## Correct Signal

The useful signal is hard-fault reduction, not final hard-fault count.

For oracle rows, baseline hard count was inferred from:

`baseline_tc = oracle_test_coverage - oracle_delta_tc`

and:

`delta_hard_count = baseline_hard_count - oracle_hard_fault_count`

| dataset | x | y | n | Pearson | Spearman |
|---|---|---|---:|---:|---:|
| oracle_balanced_train_wide | inferred_delta_hard_count | oracle_delta_tc | 4320 | 0.700235 | 0.980337 |
| oracle_balanced_val_wide | inferred_delta_hard_count | oracle_delta_tc | 666 | 0.977518 | 0.967401 |
| oracle_transfer_smoke | inferred_delta_hard_count | oracle_delta_tc | 288 | 0.929606 | 0.992546 |
| oracle_expanded_val | inferred_delta_hard_count | oracle_delta_tc | 864 | 0.978720 | 0.976472 |

## Conclusion

- Absolute hard/undetected fault count and absolute TC are strongly negatively correlated.
- That result is expected and mostly tautological.
- Absolute hard count is not a reliable action-value ranking signal.
- Hard-fault reduction is strongly aligned with delta TC and is the better target/proxy.

