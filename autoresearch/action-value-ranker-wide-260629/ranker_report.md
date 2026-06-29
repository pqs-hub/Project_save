# Fixed-Checkpoint Action-Value Ranker

generated_at: `2026-06-29T20:46:47`

verdict: `REJECT`
best_variant: `linear_l2`
baseline_score_field: `hybrid_pred`

## Gate Reasons

- transfer mean_top1_real_delta_tc became negative while baseline is positive

## Metrics

| split | variant | score | kind | Spearman | negative top1 | top1 real delta | top1 regret | pairwise acc | ndcg@10 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| expanded_val | `baseline` | `hybrid_pred` | `baseline` | 0.077941 | 0.486486 | -0.013761 | 0.040852 | 0.536992 | 0.585828 |
| expanded_val | `baseline` | `hard_reduction_total_pred` | `baseline` | 0.097497 | 0.486486 | -0.013761 | 0.040852 | 0.545815 | 0.588775 |
| expanded_val | `baseline` | `reward_pred` | `baseline` | -0.197482 | 0.702703 | -0.003186 | 0.030277 | 0.414243 | 0.497228 |
| transfer | `baseline` | `hybrid_pred` | `baseline` | 0.327398 | 0.166667 | 0.010638 | 0.012552 | 0.666266 | 0.690582 |
| transfer | `baseline` | `hard_reduction_total_pred` | `baseline` | 0.324443 | 0.166667 | 0.010638 | 0.012552 | 0.665144 | 0.687172 |
| transfer | `baseline` | `reward_pred` | `baseline` | 0.294742 | 0.500000 | 0.002967 | 0.020223 | 0.641057 | 0.571091 |
| expanded_val | `linear` | `ranker_score` | `ranker` | 0.584548 | 0.000000 | 0.006692 | 0.020399 | 0.768076 | 0.788452 |
| transfer | `linear` | `ranker_score` | `ranker` | -0.082316 | 0.166667 | -0.003533 | 0.026723 | 0.463268 | 0.397525 |
| expanded_val | `linear_l2` | `ranker_score` | `ranker` | 0.584742 | 0.000000 | 0.006692 | 0.020399 | 0.767770 | 0.788496 |
| transfer | `linear_l2` | `ranker_score` | `ranker` | -0.083407 | 0.166667 | -0.003533 | 0.026723 | 0.463436 | 0.397525 |
| expanded_val | `mlp_small` | `ranker_score` | `ranker` | 0.503289 | 0.000000 | 0.007385 | 0.019706 | 0.734526 | 0.767048 |
| transfer | `mlp_small` | `ranker_score` | `ranker` | 0.234959 | 0.000000 | 0.008225 | 0.014965 | 0.622885 | 0.770797 |
| expanded_val | `action_type_linear` | `ranker_score` | `ranker` | 0.563868 | 0.000000 | 0.006625 | 0.020466 | 0.756051 | 0.786032 |
| transfer | `action_type_linear` | `ranker_score` | `ranker` | -0.046073 | 0.000000 | 0.001242 | 0.021948 | 0.474215 | 0.430798 |

## Linear Feature Weights

### action_type_linear

| feature | weight |
|---|---:|
| `type=control0` | -0.315294 |
| `type=observe` | 0.304652 |
| `candidate_rank__minus_group_mean` | 0.262723 |
| `derived_hard_reduction_sa0_pred__minus_group_mean` | -0.253683 |
| `derived_hard_reduction_sa0_pred` | -0.219898 |
| `bounded_residual_hybrid_pred__rank_pct_group` | 0.204381 |
| `candidate_rank__rank_pct_group` | -0.195108 |
| `type=control1` | 0.193178 |
| `hybrid_pred__minus_group_mean` | -0.167065 |
| `derived_hard_count_pre_total_pred` | -0.165821 |
| `derived_hard_count_post_total_pred` | -0.162156 |
| `hard_reduction_sa1_pred` | 0.154875 |
| `derived_hard_count_pre_total_pred__rank_pct_group` | 0.145364 |
| `derived_hard_count_post_total_pred__minus_group_mean` | -0.145063 |
| `derived_hard_count_pre_total_pred__z_group` | -0.143945 |
| `candidate_rank` | 0.131036 |
| `derived_hard_reduction_total_pred__minus_group_mean` | -0.130375 |
| `hard_reduction_total_pred__rank_pct_group` | 0.127075 |
| `hybrid_pred` | -0.124607 |
| `hard_reduction_sa0_pred` | -0.120852 |

### linear

| feature | weight |
|---|---:|
| `candidate_rank__minus_group_mean` | 0.213194 |
| `type=control0` | -0.207039 |
| `type=observe` | 0.176321 |
| `type=control1` | -0.165084 |
| `hybrid_pred__minus_group_mean` | -0.161629 |
| `hard_reduction_sa1_pred` | 0.161573 |
| `candidate_rank__rank_pct_group` | -0.156531 |
| `bounded_residual_hybrid_pred__rank_pct_group` | 0.151291 |
| `derived_hard_reduction_sa0_pred__minus_group_mean` | -0.130403 |
| `derived_hard_reduction_sa0_pred__rank_pct_group` | 0.128275 |
| `hybrid_pred` | -0.119171 |
| `hard_reduction_sa0_pred` | -0.118360 |
| `hard_reduction_total_pred__rank_pct_group` | 0.110641 |
| `derived_hard_count_pre_total_pred__z_group` | -0.108566 |
| `derived_hard_count_pre_total_pred` | -0.104835 |
| `hard_reduction_sa0_pred__z_group` | 0.104114 |
| `derived_hard_reduction_sa1_pred` | -0.101122 |
| `derived_hard_reduction_sa0_pred` | -0.096618 |
| `derived_hard_reduction_total_pred__minus_group_mean` | -0.093100 |
| `guarded_reward` | 0.088818 |

### linear_l2

| feature | weight |
|---|---:|
| `candidate_rank__minus_group_mean` | 0.213069 |
| `type=control0` | -0.207021 |
| `type=observe` | 0.176318 |
| `type=control1` | -0.165080 |
| `hybrid_pred__minus_group_mean` | -0.161473 |
| `hard_reduction_sa1_pred` | 0.160893 |
| `candidate_rank__rank_pct_group` | -0.156438 |
| `bounded_residual_hybrid_pred__rank_pct_group` | 0.151234 |
| `derived_hard_reduction_sa0_pred__minus_group_mean` | -0.130368 |
| `derived_hard_reduction_sa0_pred__rank_pct_group` | 0.128206 |
| `hybrid_pred` | -0.118712 |
| `hard_reduction_sa0_pred` | -0.118163 |
| `hard_reduction_total_pred__rank_pct_group` | 0.110520 |
| `derived_hard_count_pre_total_pred__z_group` | -0.108454 |
| `hard_reduction_sa0_pred__z_group` | 0.104069 |
| `derived_hard_reduction_sa1_pred` | -0.100980 |
| `derived_hard_reduction_sa0_pred` | -0.096517 |
| `derived_hard_reduction_total_pred__minus_group_mean` | -0.093087 |
| `guarded_reward` | 0.088439 |
| `hard_reduction_sa0_pred__rank_pct_group` | 0.086044 |

## Top Improved Groups

| split | benchmark | strategy | baseline top1 | ranker top1 | gain |
|---|---|---|---:|---:|---:|
| train | `subckt_0373` | `cached_hard_cone` | -0.026330 | 0.246340 | 0.272670 |
| expanded_val | `subckt_0150` | `cached_stride` | -0.254520 | 0.004270 | 0.258790 |
| expanded_val | `subckt_0008` | `cached_random` | -0.013020 | 0.096760 | 0.109780 |
| expanded_val | `subckt_0299` | `cached_stride` | -0.010630 | 0.093070 | 0.103700 |
| expanded_val | `subckt_0150` | `cached_hard_cone` | -0.081180 | 0.004270 | 0.085450 |
| expanded_val | `subckt_0150` | `cached_random` | -0.081180 | 0.004270 | 0.085450 |
| train | `subckt_0189` | `cached_stride` | 0.002140 | 0.077480 | 0.075340 |
| train | `subckt_0297` | `cached_stride` | -0.055540 | 0.009290 | 0.064830 |
| train | `subckt_0384` | `cached_hard_cone` | -0.058190 | 0.003560 | 0.061750 |
| train | `subckt_0384` | `cached_random` | -0.058190 | 0.003560 | 0.061750 |

## Top Worsened Groups

| split | benchmark | strategy | baseline top1 | ranker top1 | gain |
|---|---|---|---:|---:|---:|
| train | `subckt_0188` | `cached_random` | 0.011220 | 0.000210 | -0.011010 |
| train | `subckt_0261` | `cached_stride` | 0.020000 | 0.000520 | -0.019480 |
| train | `subckt_0373` | `cached_stride` | 0.225060 | 0.203780 | -0.021280 |
| transfer | `b15_C` | `hard_fault_recall_union` | 0.030640 | 0.007100 | -0.023540 |
| train | `subckt_0260` | `cached_random` | 0.028880 | 0.003660 | -0.025220 |
| expanded_val | `subckt_0217` | `cached_hard_cone` | 0.029840 | 0.001410 | -0.028430 |
| expanded_val | `subckt_0217` | `cached_random` | 0.029840 | 0.001410 | -0.028430 |
| transfer | `b15_C` | `cached_stride` | 0.030590 | 0.000030 | -0.030560 |
| transfer | `b15_C` | `cached_hard_cone` | 0.030590 | 0.000030 | -0.030560 |
| train | `subckt_0081` | `cached_random` | 0.048030 | 0.001460 | -0.046570 |
