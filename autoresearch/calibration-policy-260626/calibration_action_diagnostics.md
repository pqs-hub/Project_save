# Calibration / Action-Ranking Diagnostics

## Evaluated Checkpoints

- checkpoint_count: `6`
- best_epoch: `2`
- best_hard_macro_f1_tuned: `0.26655997154796834`
- best_predictive_score: `0.44444943833137907`

## Calibration Summary

- mean_ece_sa0: `0.000000`
- mean_ece_sa1: `0.000000`
- thresholds_by_class: `thresholds_by_class.tsv`
- threshold_sweep: `threshold_sweep.tsv`
- calibration_bins: `calibration_bins.tsv`

## Worst Benchmarks

| benchmark | epoch | hard F1 | SA0 F1 | SA1 F1 | nodes |
|---|---:|---:|---:|---:|---:|

## Worst Buckets

| node bucket | positive-rate bucket | action type | samples | hard F1 |
|---|---|---|---:|---:|

## Action-Ranking Summary

- comparable_action_groups: `0`
- mean_action_pairwise_acc: `0.000000`
- mean_action_ndcg_at_10: `0.000000`
- mean_action_top1_hit: `0.000000`
- metrics_file: `action_ranking_metrics.tsv`
- examples_file: `action_group_examples.tsv`

## Recommendation

- No comparable action groups were found; action-ranking diagnostics need richer candidate grouping/data.
- Calibration error is not the dominant issue by ECE; inspect benchmark buckets and labels.
