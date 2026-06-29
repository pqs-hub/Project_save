# Fixed-Checkpoint Action-Value Ranker

generated_at: `2026-06-29T18:36:02`

verdict: `INCONCLUSIVE`
best_variant: `linear`
baseline_score_field: `hybrid_pred`

## Gate Reasons

- transfer negative_top1_rate safety failed

## Metrics

| split | variant | score | kind | Spearman | negative top1 | top1 real delta | top1 regret | pairwise acc | ndcg@10 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| expanded_val | `baseline` | `hybrid_pred` | `baseline` | 0.031476 | 0.375000 | -0.009015 | 0.035483 | 0.510698 | 0.539340 |
| expanded_val | `baseline` | `hard_reduction_total_pred` | `baseline` | 0.044034 | 0.375000 | -0.009015 | 0.035483 | 0.516522 | 0.540581 |
| expanded_val | `baseline` | `reward_pred` | `baseline` | -0.119646 | 0.541667 | -0.000172 | 0.026640 | 0.458634 | 0.528720 |
| expanded_val | `baseline` | `bounded_residual_hybrid_pred` | `baseline` | 0.042002 | 0.375000 | -0.009015 | 0.035483 | 0.515060 | 0.539787 |
| expanded_val | `baseline` | `derived_hard_reduction_hybrid_pred` | `baseline` | 0.086327 | 0.479167 | 0.006753 | 0.019714 | 0.535270 | 0.598905 |
| transfer | `baseline` | `hybrid_pred` | `baseline` | 0.327398 | 0.166667 | 0.010638 | 0.012552 | 0.666266 | 0.690582 |
| transfer | `baseline` | `hard_reduction_total_pred` | `baseline` | 0.324443 | 0.166667 | 0.010638 | 0.012552 | 0.665144 | 0.687172 |
| transfer | `baseline` | `reward_pred` | `baseline` | 0.294742 | 0.500000 | 0.002967 | 0.020223 | 0.641057 | 0.571091 |
| transfer | `baseline` | `bounded_residual_hybrid_pred` | `baseline` | 0.324835 | 0.166667 | 0.010638 | 0.012552 | 0.665406 | 0.688284 |
| transfer | `baseline` | `derived_hard_reduction_hybrid_pred` | `baseline` | 0.046113 | 0.500000 | 0.001227 | 0.021963 | 0.519918 | 0.565558 |
| expanded_val | `linear` | `ranker_score` | `ranker` | 0.102189 | 0.354167 | -0.005004 | 0.031472 | 0.553642 | 0.555702 |
| transfer | `linear` | `ranker_score` | `ranker` | 0.055356 | 0.500000 | 0.001328 | 0.021862 | 0.545587 | 0.566443 |

## Linear Feature Weights

### linear

| feature | weight |
|---|---:|
| `benchmark_id=subckt_0168` | -0.077036 |
| `benchmark_id=subckt_0068` | -0.075158 |
| `benchmark_id=subckt_0086` | -0.073405 |
| `hard_reduction_total_pred` | 0.073389 |
| `benchmark_id=subckt_0143` | 0.073044 |
| `benchmark_id=subckt_0193` | 0.072492 |
| `benchmark_id=subckt_0016` | -0.071905 |
| `benchmark_id=subckt_0391` | 0.070941 |
| `benchmark_id=subckt_0172` | 0.070926 |
| `benchmark_id=subckt_0148` | -0.070428 |
| `bounded_residual_hybrid_pred__z_group` | -0.069854 |
| `hard_reduction_sa1_pred__z_group` | 0.069484 |
| `candidate_rank` | -0.068140 |
| `type=__UNK__` | -0.068122 |
| `fc_pred__rank_pct_group` | -0.066786 |
| `benchmark_id=subckt_0239` | -0.066563 |
| `benchmark_id=subckt_0109` | -0.066451 |
| `candidate_strategy=cached_stride` | -0.066404 |
| `benchmark_id=subckt_0322` | -0.066404 |
| `derived_hard_count_pre_total_pred__z_group` | -0.066079 |

## Top Improved Groups

| split | benchmark | strategy | baseline top1 | ranker top1 | gain |
|---|---|---|---:|---:|---:|
| expanded_val | `subckt_0150` | `cached_stride` | -0.254520 | -0.087850 | 0.166670 |
| train | `subckt_0297` | `cached_stride` | -0.055540 | 0.009290 | 0.064830 |
| train | `subckt_0384` | `cached_random` | -0.058190 | 0.001810 | 0.060000 |
| train | `subckt_0261` | `cached_hard_cone` | -0.052370 | 0.000520 | 0.052890 |
| train | `subckt_0297` | `cached_hard_cone` | 0.009290 | 0.058910 | 0.049620 |
| expanded_val | `subckt_0249` | `cached_hard_cone` | -0.038400 | 0.002530 | 0.040930 |
| train | `subckt_0327` | `cached_stride` | 0.000700 | 0.038810 | 0.038110 |
| train | `subckt_0107` | `cached_stride` | 0.001340 | 0.038840 | 0.037500 |
| train | `subckt_0231` | `cached_random` | -0.015710 | 0.019620 | 0.035330 |
| train | `subckt_0072` | `cached_hard_cone` | 0.007350 | 0.041040 | 0.033690 |

## Top Worsened Groups

| split | benchmark | strategy | baseline top1 | ranker top1 | gain |
|---|---|---|---:|---:|---:|
| transfer | `i2c_aig` | `cached_hard_cone` | 0.000060 | -0.027800 | -0.027860 |
| transfer | `i2c_aig` | `cached_stride` | 0.000060 | -0.027800 | -0.027860 |
| expanded_val | `subckt_0217` | `cached_hard_cone` | 0.029840 | 0.001410 | -0.028430 |
| train | `subckt_0063` | `cached_hard_cone` | 0.032080 | 0.002490 | -0.029590 |
| train | `subckt_0063` | `cached_random` | 0.032080 | 0.002490 | -0.029590 |
| expanded_val | `subckt_0045` | `cached_random` | 0.032080 | 0.002490 | -0.029590 |
| train | `subckt_0327` | `cached_random` | 0.038920 | 0.006760 | -0.032160 |
| train | `subckt_0072` | `cached_stride` | 0.041040 | 0.007350 | -0.033690 |
| train | `subckt_0297` | `cached_random` | 0.058910 | 0.015240 | -0.043670 |
| train | `subckt_0081` | `cached_random` | 0.048030 | 0.001460 | -0.046570 |
