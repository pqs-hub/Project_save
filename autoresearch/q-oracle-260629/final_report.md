# Q Oracle Experiment

generated_at: `2026-06-30T00:46:25`

| variant | score | verdict | expanded Spearman | expanded neg top1 | transfer Spearman | transfer neg top1 | transfer regret |
|---|---|---:|---:|---:|---:|---:|---:|
| incumbent | `hybrid_pred` | BASELINE | 0.078581 | 0.486486 | 0.326968 | 0.166667 | 0.012552 |
| B_oracle_0p05 | `derived_hard_reduction_hybrid_pred` | BASELINE | 0.424585 | 0.162162 | 0.081279 | 0.000000 | 0.017512 |
| Q_v0_rank0p5 | `q_pred` | REJECT | 0.548923 | 0.270270 | 0.173058 | 0.166667 | 0.017697 |
| Q_v0_rank1p0 | `q_pred` | REJECT | 0.576224 | 0.135135 | 0.076622 | 0.166667 | 0.017712 |
| Q_v0_rank2p0 | `q_pred` | REJECT | 0.538813 | 0.351351 | 0.188922 | 0.500000 | 0.019468 |
| Q_v0_value1_rank1 | `q_pred` | REJECT | 0.495627 | 0.324324 | 0.300224 | 0.166667 | 0.012273 |

## Notes

- `q_pred` is the Q(s,a) decision score.
- Legacy scores are included only as baselines.
