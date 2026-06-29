# Prediction Accuracy Comparison

## Key Table

| metric | incumbent | control | version_A | version_B | readout |
|---|---:|---:|---:|---:|---|
| `latent_smooth_l1` | 0.065472 | 0.045460 | 0.052211 | 0.044572 | 状态预测误差 |
| `latent_cosine` | 0.968105 | 0.977944 | 0.975160 | 0.978399 | 状态预测相似度 |
| `scoap_mae` | 0.068131 | 0.056134 | 0.059089 | 0.049371 | SCOAP误差 |
| `scoap_acc_at_005` | 0.499832 | 0.549735 | 0.555089 | 0.603871 | SCOAP准确率 |
| `delta_scoap_mae` | 0.005661 | 0.003198 | 0.005066 | 0.005997 | SCOAP变化误差 |
| `delta_scoap_acc_at_001` | 0.869946 | 0.977686 | 0.873118 | 0.927258 | SCOAP变化准确率 |
| `hard_bce` | 0.113479 | 0.106482 | 0.104464 | 0.088259 | 节点hard训练误差 |
| `hard_macro_f1_tuned` | 0.794805 | 0.782191 | 0.810817 | 0.802648 | 节点hard F1 |
| `hard_sa0_f1` | 0.283465 | 0.637275 | 0.596685 | 0.632812 | SA0 hard F1 |
| `hard_sa1_f1` | 0.823636 | 0.851756 | 0.863093 | 0.857764 | SA1 hard F1 |
| `hard_recall_at_top_10pct` | 0.996644 | 0.996199 | 0.997774 | 0.999931 | top10% hard召回 |
| `hard_count_mae` | 0.488317 | 0.484386 | 0.613002 | 0.604993 | 旧hard_count头误差 |
| `hard_count_spearman` | 0.453320 | 0.203453 | -0.425166 | 0.038323 | 旧hard_count排序相关 |
| `hard_count_top10_overlap` | 0.685392 | 0.685702 | 0.114025 | 0.000000 | 旧hard_count top10重合 |
| `hard_reduction_mae` | 0.201380 | 0.166256 | 0.169394 | 0.903956 | 旧hard_reduction头误差 |
| `hard_reduction_sign_acc` | 0.602865 | 0.602865 | 0.602865 | 0.610677 | 旧hard_reduction方向准确 |
| `hard_reduction_score` | 0.798620 | 0.833744 | 0.830606 | 0.096044 | 旧hard_reduction分数 |
| `derived_hard_count_post_mae` | 0.064223 | 0.049753 | 0.048377 | 0.031909 | derived hard_count误差 |
| `derived_hard_reduction_mae` | 0.307793 | 0.363111 | 0.361729 | 0.408363 | derived hard_reduction误差 |
| `derived_hard_reduction_sign_acc` | 0.769531 | 0.769531 | 0.713542 | 0.769531 | derived hard_reduction方向准确 |
| `derived_hard_reduction_score` | 0.692207 | 0.636889 | 0.638271 | 0.591637 | derived hard_reduction分数 |
| `reward_mae` | 1.925061 | 1.933529 | 1.996263 | 1.970852 | reward/FC头误差 |
| `reward_sign_acc` | 0.792969 | 0.792969 | 0.792969 | 0.792969 | reward方向准确 |
| `predictive_score` | 0.820147 | 0.822154 | 0.778059 | 0.655775 | 综合预测分 |

## Main Findings

- Version A predicts node-level hard best: hard F1 `0.810817`, higher than incumbent `0.794805`.
- Version B also predicts node-level hard well: hard F1 `0.802648`, higher than incumbent.
- Version A old hard_count head becomes unusable because it was not trained: hard_count Spearman `-0.425166`.
- Version B old hard_reduction head collapses because it was not trained: old hard_reduction score `0.096044`.
- Version B derived hard_count itself is numerically accurate: derived hard_count MAE `0.031909`, better than incumbent `0.064223`.
- But Version B derived hard_reduction is weak: derived hard_reduction score `0.591637`, below incumbent `0.692207`.
- So the failure of B is not counting hard nodes; the failure is using before/after count difference as a stable action-value signal.

## Practical Conclusion

`hard_count` can likely be removed as a training loss. `hard_reduction` should stay as a direct action-level prediction target.
