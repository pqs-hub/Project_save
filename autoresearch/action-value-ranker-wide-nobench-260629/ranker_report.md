# Fixed-Checkpoint Action-Value Ranker

generated_at: `2026-06-29T20:47:28`

verdict: `PROMOTE`
best_variant: `linear`
baseline_score_field: `hybrid_pred`

## Metrics

| split | variant | score | kind | Spearman | negative top1 | top1 real delta | top1 regret | pairwise acc | ndcg@10 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| expanded_val | `baseline` | `hybrid_pred` | `baseline` | 0.077941 | 0.486486 | -0.013761 | 0.040852 | 0.536992 | 0.585828 |
| expanded_val | `baseline` | `hard_reduction_total_pred` | `baseline` | 0.097497 | 0.486486 | -0.013761 | 0.040852 | 0.545815 | 0.588775 |
| expanded_val | `baseline` | `reward_pred` | `baseline` | -0.197482 | 0.702703 | -0.003186 | 0.030277 | 0.414243 | 0.497228 |
| transfer | `baseline` | `hybrid_pred` | `baseline` | 0.327398 | 0.166667 | 0.010638 | 0.012552 | 0.666266 | 0.690582 |
| transfer | `baseline` | `hard_reduction_total_pred` | `baseline` | 0.324443 | 0.166667 | 0.010638 | 0.012552 | 0.665144 | 0.687172 |
| transfer | `baseline` | `reward_pred` | `baseline` | 0.294742 | 0.500000 | 0.002967 | 0.020223 | 0.641057 | 0.571091 |
| expanded_val | `linear` | `ranker_score` | `ranker` | 0.571581 | 0.000000 | 0.006692 | 0.020399 | 0.760324 | 0.787756 |
| transfer | `linear` | `ranker_score` | `ranker` | -0.077317 | 0.000000 | 0.001218 | 0.021972 | 0.463849 | 0.430350 |
| expanded_val | `linear_l2` | `ranker_score` | `ranker` | 0.571581 | 0.000000 | 0.006692 | 0.020399 | 0.760324 | 0.787756 |
| transfer | `linear_l2` | `ranker_score` | `ranker` | -0.077317 | 0.000000 | 0.001218 | 0.021972 | 0.463849 | 0.430350 |
| expanded_val | `mlp_small` | `ranker_score` | `ranker` | 0.551690 | 0.000000 | 0.001768 | 0.025324 | 0.748294 | 0.771049 |
| transfer | `mlp_small` | `ranker_score` | `ranker` | -0.056032 | 0.000000 | 0.001105 | 0.022085 | 0.492705 | 0.484455 |
| expanded_val | `action_type_linear` | `ranker_score` | `ranker` | 0.569304 | 0.000000 | 0.004225 | 0.022866 | 0.755034 | 0.780144 |
| transfer | `action_type_linear` | `ranker_score` | `ranker` | -0.056142 | 0.166667 | -0.003500 | 0.026690 | 0.479879 | 0.450267 |

## Linear Feature Weights

### action_type_linear

| feature | weight |
|---|---:|
| `type=observe` | 0.469115 |
| `type=control1` | -0.415983 |
| `derived_hard_reduction_sa0_pred__minus_group_mean` | -0.398060 |
| `derived_hard_reduction_sa0_pred` | -0.347853 |
| `candidate_rank__minus_group_mean` | 0.345976 |
| `candidate_rank__rank_pct_group` | -0.251392 |
| `derived_hard_count_post_total_pred` | -0.249843 |
| `derived_hard_count_post_total_pred__minus_group_mean` | -0.224445 |
| `hard_reduction_sa1_pred` | 0.219152 |
| `bounded_residual_hybrid_pred__rank_pct_group` | 0.214559 |
| `derived_hard_count_pre_total_pred` | -0.209138 |
| `hard_reduction_sa0_pred` | -0.205817 |
| `hybrid_pred__minus_group_mean` | -0.195795 |
| `hard_reduction_total_pred__rank_pct_group` | 0.184583 |
| `derived_hard_reduction_sa1_pred__rank_pct_group` | -0.177201 |
| `derived_hard_count_pre_total_pred__rank_pct_group` | 0.176827 |
| `candidate_rank` | 0.150283 |
| `derived_hard_reduction_sa0_pred__rank_pct_group` | 0.148860 |
| `derived_hard_count_pre_total_pred__z_group` | -0.139535 |
| `candidate_rank__z_group` | -0.136002 |

### linear

| feature | weight |
|---|---:|
| `candidate_rank__minus_group_mean` | 0.238461 |
| `type=control0` | -0.223541 |
| `hybrid_pred__minus_group_mean` | -0.199215 |
| `type=observe` | 0.185514 |
| `derived_hard_reduction_sa0_pred__minus_group_mean` | -0.182666 |
| `hard_reduction_sa1_pred` | 0.181988 |
| `type=control1` | -0.168663 |
| `bounded_residual_hybrid_pred__rank_pct_group` | 0.163297 |
| `hard_reduction_sa0_pred` | -0.162966 |
| `derived_hard_reduction_sa0_pred__rank_pct_group` | 0.161720 |
| `candidate_rank__rank_pct_group` | -0.156506 |
| `derived_hard_count_pre_total_pred__z_group` | -0.150444 |
| `hybrid_pred` | -0.136121 |
| `derived_hard_count_pre_total_pred` | -0.134761 |
| `derived_hard_reduction_sa0_pred` | -0.132461 |
| `derived_hard_reduction_sa1_pred` | -0.127831 |
| `guarded_reward` | 0.123504 |
| `derived_hard_reduction_total_pred__minus_group_mean` | -0.121945 |
| `hard_reduction_total_pred__rank_pct_group` | 0.117284 |
| `hard_reduction_sa0_pred__z_group` | 0.112691 |

### linear_l2

| feature | weight |
|---|---:|
| `candidate_rank__minus_group_mean` | 0.238064 |
| `type=control0` | -0.223460 |
| `hybrid_pred__minus_group_mean` | -0.198781 |
| `type=observe` | 0.185506 |
| `derived_hard_reduction_sa0_pred__minus_group_mean` | -0.182519 |
| `hard_reduction_sa1_pred` | 0.180425 |
| `type=control1` | -0.168650 |
| `bounded_residual_hybrid_pred__rank_pct_group` | 0.163179 |
| `hard_reduction_sa0_pred` | -0.162206 |
| `derived_hard_reduction_sa0_pred__rank_pct_group` | 0.161455 |
| `candidate_rank__rank_pct_group` | -0.156250 |
| `derived_hard_count_pre_total_pred__z_group` | -0.149604 |
| `hybrid_pred` | -0.135129 |
| `derived_hard_reduction_sa0_pred` | -0.132073 |
| `derived_hard_reduction_sa1_pred` | -0.127288 |
| `guarded_reward` | 0.122628 |
| `derived_hard_reduction_total_pred__minus_group_mean` | -0.121875 |
| `hard_reduction_total_pred__rank_pct_group` | 0.117053 |
| `hard_reduction_sa0_pred__z_group` | 0.112516 |
| `derived_hard_reduction_sa0_pred__z_group` | 0.096545 |

## Top Improved Groups

| split | benchmark | strategy | baseline top1 | ranker top1 | gain |
|---|---|---|---:|---:|---:|
| expanded_val | `subckt_0150` | `cached_stride` | -0.254520 | 0.004270 | 0.258790 |
| train | `subckt_0373` | `cached_hard_cone` | -0.026330 | 0.161230 | 0.187560 |
| expanded_val | `subckt_0008` | `cached_random` | -0.013020 | 0.096760 | 0.109780 |
| expanded_val | `subckt_0299` | `cached_stride` | -0.010630 | 0.093070 | 0.103700 |
| train | `subckt_0384` | `cached_random` | -0.058190 | 0.036460 | 0.094650 |
| expanded_val | `subckt_0150` | `cached_hard_cone` | -0.081180 | 0.004270 | 0.085450 |
| expanded_val | `subckt_0150` | `cached_random` | -0.081180 | 0.004270 | 0.085450 |
| train | `subckt_0189` | `cached_stride` | 0.002140 | 0.077480 | 0.075340 |
| train | `subckt_0297` | `cached_stride` | -0.055540 | 0.009290 | 0.064830 |
| train | `subckt_0384` | `cached_hard_cone` | -0.058190 | 0.003560 | 0.061750 |

## Top Worsened Groups

| split | benchmark | strategy | baseline top1 | ranker top1 | gain |
|---|---|---|---:|---:|---:|
| train | `subckt_0261` | `cached_stride` | 0.020000 | 0.000520 | -0.019480 |
| train | `subckt_0373` | `cached_stride` | 0.225060 | 0.203780 | -0.021280 |
| transfer | `b15_C` | `hard_fault_recall_union` | 0.030640 | 0.007100 | -0.023540 |
| train | `subckt_0260` | `cached_random` | 0.028880 | 0.003660 | -0.025220 |
| expanded_val | `subckt_0217` | `cached_hard_cone` | 0.029840 | 0.001410 | -0.028430 |
| expanded_val | `subckt_0217` | `cached_random` | 0.029840 | 0.001410 | -0.028430 |
| transfer | `b15_C` | `cached_stride` | 0.030590 | 0.000030 | -0.030560 |
| transfer | `b15_C` | `cached_hard_cone` | 0.030590 | 0.000030 | -0.030560 |
| train | `subckt_0327` | `cached_random` | 0.038920 | 0.000700 | -0.038220 |
| train | `subckt_0081` | `cached_random` | 0.048030 | 0.001460 | -0.046570 |
