# Scheme A SCOAP / delta-SCOAP Ablation

## Setup

This repeats the same SCOAP / delta-SCOAP ablation previously run on Scheme B, but on Scheme A.

Scheme A keeps the direct `hard_reduction` head and disables `hard_count`, `FC`, and `return`.

Variants:

| Variant | SCOAP loss weight | delta-SCOAP loss weight |
|---|---:|---:|
| A_base | 0.5 | 0.3 |
| A_only_scoap | 0.5 | 0.0 |
| A_only_delta_scoap | 0.0 | 0.3 |

Checkpoints:

| Variant | Checkpoint |
|---|---|
| A_base | `autoresearch/autoresearch-260629-1550/runs/version_A_no_hard_count/best.pt` |
| A_only_scoap | `autoresearch/ablate-scoap-version-a-260629-1806/runs/A_only_scoap/best.pt` |
| A_only_delta_scoap | `autoresearch/ablate-scoap-version-a-260629-1806/runs/A_only_delta_scoap/best.pt` |

## Hard / SCOAP Gate

| Variant | hard F1 | SCOAP MAE | SCOAP acc@0.05 | delta-SCOAP MAE | delta-SCOAP acc@0.001 | hard reduction score |
|---|---:|---:|---:|---:|---:|---:|
| A_base | 0.810817 | 0.059089 | 0.555089 | 0.005066 | 0.873118 | 0.830606 |
| A_only_scoap | 0.811870 | 0.056825 | 0.543368 | 0.171434 | 0.029255 | 0.817459 |
| A_only_delta_scoap | 0.820835 | 0.428721 | 0.051389 | 0.003647 | 0.988229 | 0.827909 |

Interpretation:

- `A_only_scoap` predicts SCOAP slightly better by MAE, but delta-SCOAP collapses.
- `A_only_delta_scoap` predicts delta-SCOAP best, but SCOAP collapses.
- hard F1 is not enough to choose a planner model here, because both ablations keep hard F1 near or above base while their action ranking behavior changes differently.

## Expanded Oracle Validation Gate

Metric shown here uses `hybrid_pred`.

| Variant | Spearman | negative top1 | top1 real delta TC | top1 regret |
|---|---:|---:|---:|---:|
| A_base | -0.040485 | 0.562500 | -0.003676 | 0.030143 |
| A_only_scoap | 0.078398 | 0.270833 | 0.004881 | 0.021587 |
| A_only_delta_scoap | -0.104651 | 0.541667 | -0.002532 | 0.028999 |

Interpretation:

- On expanded validation, `A_only_scoap` is clearly better than `A_base`.
- It changes the top selected action from negative average true gain to positive average true gain.
- It also cuts the bad-top1 rate from 56.25% to 27.08%.
- `A_only_delta_scoap` does not help this gate.

## Transfer Oracle Gate

Metric shown here uses `hybrid_pred`.

| Variant | Spearman | negative top1 | top1 real delta TC | top1 regret |
|---|---:|---:|---:|---:|
| A_base | -0.084822 | 0.500000 | 0.001800 | 0.021390 |
| A_only_scoap | -0.012033 | 0.500000 | -0.004248 | 0.027438 |
| A_only_delta_scoap | -0.002604 | 0.500000 | 0.001368 | 0.021822 |

Interpretation:

- `A_only_scoap` improves transfer Spearman, but its chosen top1 action has worse real delta TC and higher regret than `A_base`.
- `A_only_delta_scoap` has the least negative transfer Spearman, but does not improve top1 quality.
- All variants still have 50% negative top1 on transfer, so none is safe enough to promote.

## Decision

Do not promote either A ablation.

Reason:

- `A_only_scoap` is useful evidence: removing delta-SCOAP helps the expanded validation oracle ranking, but this does not transfer cleanly.
- `A_only_delta_scoap` is not useful as a promotion candidate: it improves delta-SCOAP prediction and hard F1, but damages SCOAP and expanded oracle ranking.
- Scheme A still needs a stronger objective directly tied to action value ordering; changing only SCOAP versus delta-SCOAP losses is not enough.

## Files

- Hard gate CSVs: `autoresearch/ablate-scoap-version-a-260629-1806/gates/hard/`
- Expanded oracle gate: `autoresearch/ablate-scoap-version-a-260629-1806/gates/expanded_val/oracle_action_value_summary.tsv`
- Transfer oracle gate: `autoresearch/ablate-scoap-version-a-260629-1806/gates/transfer/oracle_action_value_summary.tsv`
- Summary TSV: `autoresearch/ablate-scoap-version-a-260629-1806/ablation_summary.tsv`
