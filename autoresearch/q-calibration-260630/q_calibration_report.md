# Q Calibration Fixed-Candidate Ablation

generated_at: `2026-06-30T01:29:09`

## Summary

promoted: `0`

| checkpoint | method | verdict | expanded Spearman | expanded neg top1 | transfer Spearman | transfer neg top1 | transfer regret | rank changes expanded/transfer |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Q_v0_rank0p5 | `raw` | REJECT | 0.548923 | 0.270270 | 0.173058 | 0.166667 | 0.017697 | 0/0 |
| Q_v0_rank0p5 | `group_center` | REJECT | 0.548923 | 0.270270 | 0.173058 | 0.166667 | 0.017697 | 0/0 |
| Q_v0_rank0p5 | `group_zscore` | REJECT | 0.548923 | 0.270270 | 0.173058 | 0.166667 | 0.017697 | 0/0 |
| Q_v0_rank0p5 | `group_rank_pct` | REJECT | 0.548390 | 0.270270 | 0.172795 | 0.166667 | 0.017697 | 0/1 |
| Q_v0_rank0p5 | `circuit_zscore` | REJECT | 0.548923 | 0.270270 | 0.173058 | 0.166667 | 0.017697 | 0/0 |
| Q_v0_rank0p5 | `global_zscore` | REJECT | 0.548923 | 0.270270 | 0.173058 | 0.166667 | 0.017697 | 0/0 |
| Q_v0_rank0p5 | `platt` | REJECT | 0.548923 | 0.270270 | 0.173058 | 0.166667 | 0.017697 | 0/0 |
| Q_v0_rank1p0 | `raw` | REJECT | 0.576224 | 0.135135 | 0.076622 | 0.166667 | 0.017712 | 0/0 |
| Q_v0_rank1p0 | `group_center` | REJECT | 0.576224 | 0.135135 | 0.076622 | 0.166667 | 0.017712 | 0/0 |
| Q_v0_rank1p0 | `group_zscore` | REJECT | 0.576224 | 0.135135 | 0.076622 | 0.166667 | 0.017712 | 0/0 |
| Q_v0_rank1p0 | `group_rank_pct` | REJECT | 0.575729 | 0.135135 | 0.076748 | 0.166667 | 0.017712 | 1/1 |
| Q_v0_rank1p0 | `circuit_zscore` | REJECT | 0.576224 | 0.135135 | 0.076622 | 0.166667 | 0.017712 | 0/0 |
| Q_v0_rank1p0 | `global_zscore` | REJECT | 0.576224 | 0.135135 | 0.076622 | 0.166667 | 0.017712 | 0/0 |
| Q_v0_rank1p0 | `platt` | REJECT | 0.576224 | 0.135135 | 0.076622 | 0.166667 | 0.017712 | 0/0 |
| Q_v0_rank2p0 | `raw` | REJECT | 0.538813 | 0.351351 | 0.188922 | 0.500000 | 0.019468 | 0/0 |
| Q_v0_rank2p0 | `group_center` | REJECT | 0.538813 | 0.351351 | 0.188922 | 0.500000 | 0.019468 | 0/0 |
| Q_v0_rank2p0 | `group_zscore` | REJECT | 0.538813 | 0.351351 | 0.188922 | 0.500000 | 0.019468 | 0/0 |
| Q_v0_rank2p0 | `group_rank_pct` | REJECT | 0.538845 | 0.351351 | 0.189044 | 0.500000 | 0.019468 | 0/0 |
| Q_v0_rank2p0 | `circuit_zscore` | REJECT | 0.538813 | 0.351351 | 0.188922 | 0.500000 | 0.019468 | 0/0 |
| Q_v0_rank2p0 | `global_zscore` | REJECT | 0.538813 | 0.351351 | 0.188922 | 0.500000 | 0.019468 | 0/0 |
| Q_v0_rank2p0 | `platt` | REJECT | 0.538813 | 0.351351 | 0.188922 | 0.500000 | 0.019468 | 0/0 |
| Q_v0_value1_rank1 | `raw` | REJECT | 0.495627 | 0.324324 | 0.300224 | 0.166667 | 0.012273 | 0/0 |
| Q_v0_value1_rank1 | `group_center` | REJECT | 0.495627 | 0.324324 | 0.300224 | 0.166667 | 0.012273 | 0/0 |
| Q_v0_value1_rank1 | `group_zscore` | REJECT | 0.495627 | 0.324324 | 0.300224 | 0.166667 | 0.012273 | 0/0 |
| Q_v0_value1_rank1 | `group_rank_pct` | REJECT | 0.495146 | 0.324324 | 0.300209 | 0.166667 | 0.012273 | 0/1 |
| Q_v0_value1_rank1 | `circuit_zscore` | REJECT | 0.495627 | 0.324324 | 0.300224 | 0.166667 | 0.012273 | 0/0 |
| Q_v0_value1_rank1 | `global_zscore` | REJECT | 0.495627 | 0.324324 | 0.300224 | 0.166667 | 0.012273 | 0/0 |
| Q_v0_value1_rank1 | `platt` | REJECT | 0.495627 | 0.324324 | 0.300224 | 0.166667 | 0.012273 | 0/0 |

## Best Expanded

- checkpoint: `Q_v0_rank1p0`
- method: `raw`
- expanded Spearman: `0.576224348635029`

## Best Transfer Regret

- checkpoint: `Q_v0_value1_rank1`
- method: `raw`
- transfer regret: `0.01227333333333334`

## Important Note

Most calibration methods here are monotonic transforms of `q_pred` within each fixed candidate group. They can improve score scale and sign calibration, but they cannot change top1 selection unless the transform changes action order. `rank_changed_groups` explicitly records whether action ordering changed.

## Outputs

- `autoresearch/q-calibration-260630/q_calibration_promotion.tsv`
- `autoresearch/q-calibration-260630/q_calibration_summary.tsv`
- `autoresearch/q-calibration-260630/q_calibrated_actions.tsv`
