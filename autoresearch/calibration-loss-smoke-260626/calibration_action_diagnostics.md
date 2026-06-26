# Calibration / Action-Ranking Diagnostics

## Evaluated Checkpoints

- checkpoint_count: `6`
- best_epoch: `2`
- best_hard_macro_f1_tuned: `0.26655997154796834`
- best_predictive_score: `0.44444943833137907`

## Calibration Summary

- mean_ece_sa0: `0.182724`
- mean_ece_sa1: `0.457790`
- thresholds_by_class: `thresholds_by_class.tsv`
- threshold_sweep: `threshold_sweep.tsv`
- calibration_bins: `calibration_bins.tsv`

## Worst Benchmarks

| benchmark | epoch | hard F1 | SA0 F1 | SA1 F1 | nodes |
|---|---:|---:|---:|---:|---:|
| `subckt_0093` | `1` | `0.1958` | `0.0241` | `0.3675` | `254` |
| `subckt_0093` | `4` | `0.2031` | `0.1622` | `0.2440` | `254` |
| `subckt_0093` | `-1` | `0.2031` | `0.1622` | `0.2440` | `254` |
| `subckt_0093` | `3` | `0.2526` | `0.0550` | `0.4502` | `254` |
| `subckt_0093` | `2` | `0.2666` | `0.1159` | `0.4172` | `254` |

## Worst Buckets

| node bucket | positive-rate bucket | action type | samples | hard F1 |
|---|---|---|---:|---:|
| `<500` | `[0.001,0.01)` | `control1` | `3` | `0.0536` |
| `<500` | `[0.001,0.01)` | `control1` | `3` | `0.0769` |
| `<500` | `[0.001,0.01)` | `control1` | `3` | `0.0769` |
| `<500` | `[0.001,0.01)` | `control1` | `3` | `0.1034` |
| `<500` | `[0.001,0.01)` | `control0` | `2` | `0.1071` |

## Action-Ranking Summary

- comparable_action_groups: `0`
- mean_action_pairwise_acc: `0.000000`
- mean_action_ndcg_at_10: `0.000000`
- mean_action_top1_hit: `0.000000`
- metrics_file: `action_ranking_metrics.tsv`
- examples_file: `action_group_examples.tsv`

## Recommendation

- No comparable action groups were found; action-ranking diagnostics need richer candidate grouping/data.
- Calibration error is high; prioritize threshold/calibration work before claiming new hard-F1 gains.
