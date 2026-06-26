# Calibration Policy Comparison

## Scope

- No model training was run.
- This report compares post-hoc hard-node threshold policies on the same evaluated validation samples.
- ECE/Brier are probability calibration diagnostics; threshold policies change F1/FP/FN tradeoffs, not raw probability calibration.

## Headline

- best_policy: `class_tuned`
- best_shrinkage: ``
- best_hard_macro_f1: `0.266560`
- class_tuned_hard_macro_f1: `0.266560`
- global_0p5_hard_macro_f1: `0.050538`
- delta_vs_class_tuned: `0.000000`
- best_worst_benchmark_f1: `0.266560`

## Policy Table

| policy | shrinkage | hard F1 | delta vs class | worst benchmark F1 | decision |
|---|---:|---:|---:|---:|---|
| `class_tuned` | `` | `0.2666` | `0.0000` | `0.2666` | `neutral` |
| `benchmark_tuned` | `` | `0.2666` | `0.0000` | `0.2666` | `neutral` |
| `benchmark_shrinkage_0.25` | `0.25` | `0.2666` | `0.0000` | `0.2666` | `neutral` |
| `benchmark_shrinkage_0.5` | `0.5` | `0.2666` | `0.0000` | `0.2666` | `neutral` |
| `benchmark_shrinkage_0.75` | `0.75` | `0.2666` | `0.0000` | `0.2666` | `neutral` |
| `class_tuned` | `` | `0.2666` | `0.0000` | `0.2666` | `neutral` |
| `benchmark_tuned` | `` | `0.2666` | `0.0000` | `0.2666` | `neutral` |
| `benchmark_shrinkage_0.25` | `0.25` | `0.2666` | `0.0000` | `0.2666` | `neutral` |
| `benchmark_shrinkage_0.5` | `0.5` | `0.2666` | `0.0000` | `0.2666` | `neutral` |
| `benchmark_shrinkage_0.75` | `0.75` | `0.2666` | `0.0000` | `0.2666` | `neutral` |
| `class_tuned` | `` | `0.2526` | `0.0000` | `0.2526` | `neutral` |
| `benchmark_tuned` | `` | `0.2526` | `0.0000` | `0.2526` | `neutral` |
| `benchmark_shrinkage_0.25` | `0.25` | `0.2526` | `0.0000` | `0.2526` | `neutral` |
| `benchmark_shrinkage_0.5` | `0.5` | `0.2526` | `0.0000` | `0.2526` | `neutral` |
| `benchmark_shrinkage_0.75` | `0.75` | `0.2526` | `0.0000` | `0.2526` | `neutral` |
| `class_tuned` | `` | `0.2031` | `0.0000` | `0.2031` | `neutral` |
| `benchmark_tuned` | `` | `0.2031` | `0.0000` | `0.2031` | `neutral` |
| `benchmark_shrinkage_0.25` | `0.25` | `0.2031` | `0.0000` | `0.2031` | `neutral` |
| `benchmark_shrinkage_0.5` | `0.5` | `0.2031` | `0.0000` | `0.2031` | `neutral` |
| `benchmark_shrinkage_0.75` | `0.75` | `0.2031` | `0.0000` | `0.2031` | `neutral` |
| `class_tuned` | `` | `0.2031` | `0.0000` | `0.2031` | `neutral` |
| `benchmark_tuned` | `` | `0.2031` | `0.0000` | `0.2031` | `neutral` |
| `benchmark_shrinkage_0.25` | `0.25` | `0.2031` | `0.0000` | `0.2031` | `neutral` |
| `benchmark_shrinkage_0.5` | `0.5` | `0.2031` | `0.0000` | `0.2031` | `neutral` |
| `benchmark_shrinkage_0.75` | `0.75` | `0.2031` | `0.0000` | `0.2031` | `neutral` |
| `class_tuned` | `` | `0.1958` | `0.0000` | `0.1958` | `neutral` |
| `benchmark_tuned` | `` | `0.1958` | `0.0000` | `0.1958` | `neutral` |
| `benchmark_shrinkage_0.25` | `0.25` | `0.1958` | `0.0000` | `0.1958` | `neutral` |
| `benchmark_shrinkage_0.5` | `0.5` | `0.1958` | `0.0000` | `0.1958` | `neutral` |
| `benchmark_shrinkage_0.75` | `0.75` | `0.1958` | `0.0000` | `0.1958` | `neutral` |
| `global_0p5` | `` | `0.0505` | `-0.1525` | `0.0505` | `reject` |
| `global_0p5` | `` | `0.0505` | `-0.1525` | `0.0505` | `reject` |
| `global_0p5` | `` | `0.0420` | `-0.2106` | `0.0420` | `reject` |
| `global_0p5` | `` | `0.0344` | `-0.2321` | `0.0344` | `reject` |
| `global_0p5` | `` | `0.0344` | `-0.2321` | `0.0344` | `reject` |
| `global_0p5` | `` | `0.0168` | `-0.1790` | `0.0168` | `reject` |

## Worst Per-Benchmark Cases

| policy | benchmark | hard F1 | SA0 F1 | SA1 F1 | thresholds |
|---|---|---:|---:|---:|---|
| `global_0p5` | `subckt_0093` | `0.0168` | `0.0012` | `0.0325` | `0.500/0.500` |
| `global_0p5` | `subckt_0093` | `0.0344` | `0.0299` | `0.0390` | `0.500/0.500` |
| `global_0p5` | `subckt_0093` | `0.0344` | `0.0299` | `0.0390` | `0.500/0.500` |
| `global_0p5` | `subckt_0093` | `0.0420` | `0.0290` | `0.0550` | `0.500/0.500` |
| `global_0p5` | `subckt_0093` | `0.0505` | `0.0257` | `0.0754` | `0.500/0.500` |
| `global_0p5` | `subckt_0093` | `0.0505` | `0.0257` | `0.0754` | `0.500/0.500` |
| `class_tuned` | `subckt_0093` | `0.1958` | `0.0241` | `0.3675` | `0.639/0.973` |
| `benchmark_tuned` | `subckt_0093` | `0.1958` | `0.0241` | `0.3675` | `0.639/0.973` |

## Recommendation

- Do not change training/evaluation policy yet; calibration policy did not produce a reliable F1 gain.
- Promotion rule used here: hard F1 improves by at least `0.03`, or worst benchmark F1 improves by at least `0.10` versus class-tuned.
- Output files: `calibration_mode_metrics.tsv`, `per_benchmark_calibrated_metrics.tsv`, `threshold_policy_comparison.tsv`.
