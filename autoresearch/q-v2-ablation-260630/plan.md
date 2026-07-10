# Q-v2 Ablation Plan

Goal: test whether decision-quality improves when the main loss is candidate-group ranking instead of flat value regression or auxiliary proxy prediction.

## Variants

| variant | decision loss | utility loss | purpose |
|---|---|---|---|
| `q_v2_value_only` | `1.0 * q_value` | off | regression baseline: learns delta-TC value scale without ranking/listwise pressure |
| `q_v2_pairwise_only` | `1.0 * q_rank` | off | tests whether pairwise best-vs-negative order alone improves candidate ranking |
| `q_v2_pairwise_listwise` | `1.0 * q_rank + 0.5 * candidate` | off | tests whether listwise candidate distribution improves top-k/top-1 selection |
| `q_v2_pairwise_listwise_hardred` | `1.0 * q_rank + 0.5 * candidate + 0.1 * q_value` | `0.8 * hard_reduction + 0.25 * reward + 0.2 * return` | main candidate: ranking objective plus hard-fault utility regularization |

## Shared Representation Losses

All variants use the same weak representation regularizers:

```text
0.05 * jepa
0.03 * delta_scoap
0.01 * scoap
0.02 * hard_bce
0.05 * hard_rank
0.02 * hard_soft_f1
0.00 * hard_count
```

## Oracle Ranking Setup

```text
oracle_pairwise_mode = best_vs_hard_topk
oracle_positive_topk = 1
oracle_hard_negative_topk = 16
oracle_max_pairs_per_group = 128
oracle_pairwise_temperature = 0.5
candidate_target_temperature = 0.5
candidate_pred_temperature = 1.0
```

`best_vs_hard_topk` now fills extra eligible negatives up to `oracle_max_pairs_per_group`, so the hard negative pool starts from model top-k mistakes but does not cap the group at only 16 pairs.

## Run

```bash
cd /data4/pengqingsong/DFT/TPI-my.3
bash scripts/run_q_v2_parallel.sh
```

The script runs four jobs on GPUs 4, 5, 6, 7, streams all logs live to the terminal, writes per-job logs to `autoresearch/q-v2-ablation-260630/logs/`, runs expanded/transfer gates, then writes:

```text
autoresearch/q-v2-ablation-260630/q_v2_ablation_summary.tsv
autoresearch/q-v2-ablation-260630/q_v2_ablation_report.md
```

## Promotion Gate

A candidate promotes only if it passes the same fixed oracle gate used for Q-v1:

```text
expanded_spearman >= 0.50
expanded_negative_top1 <= 0.162
transfer_negative_top1 <= 0.167
transfer_top1_regret <= 0.012552
transfer_spearman >= 0.20
```
