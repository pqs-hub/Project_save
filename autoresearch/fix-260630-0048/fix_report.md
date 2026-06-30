# Fix Report: Direct Q(s,a)-Centric Refactor Execution

generated_at: `2026-06-30T00:48:00+08:00`

## What Was Executed

按要求直接执行收敛重构，不只写计划：

- `model.py` 新增 `q_head`。
- `predict_from_latent(...)` 输出 `q_pred`。
- `score_pred` 改为直接等于 `q_pred`。
- `train.py` 新增 Q oracle losses：
  - `lambda_q_value`
  - `lambda_q_rank`
  - `lambda_candidate`
- oracle training 默认 `oracle_ranking_score_field = q_pred`。
- `plan.py` planner 默认 `--score-field q_pred`。
- oracle probe / checkpoint gate 支持 `q_pred`。
- 新增并行实验脚本 `scripts/run_q_oracle_experiment.py`。
- 并行训练 4 个 Q-only 变体，使用 GPU `cuda:4,cuda:5,cuda:6,cuda:7`。

## Verification

通过：

```bash
python -m py_compile tpi_jepa/model.py tpi_jepa/train.py tpi_jepa/plan.py tpi_jepa/smoke_test.py scripts/oracle_action_value_probe.py scripts/evaluate_oracle_action_values.py scripts/run_q_oracle_experiment.py
python -m tpi_jepa.smoke_test
```

## Experiment

实验目录：

`autoresearch/q-oracle-260629`

训练数据：

`autoresearch/oracle-balanced-negative-rich-260629-wide/balanced_train_oracle_actions.tsv`

验证：

- expanded val: `autoresearch/oracle-balanced-negative-rich-260629-wide/balanced_val_oracle_actions.tsv`
- transfer: `autoresearch/oracle-action-probe-260629-smoke/oracle_actions.tsv`

变体：

| variant | lambda_q_value | lambda_q_rank | lambda_candidate |
|---|---:|---:|---:|
| `Q_v0_rank0p5` | 0.5 | 0.5 | 0.0 |
| `Q_v0_rank1p0` | 0.5 | 1.0 | 0.0 |
| `Q_v0_rank2p0` | 0.5 | 2.0 | 0.0 |
| `Q_v0_value1_rank1` | 1.0 | 1.0 | 0.0 |

## Results

| variant | score | verdict | expanded Spearman | expanded neg top1 | transfer Spearman | transfer neg top1 | transfer regret |
|---|---|---:|---:|---:|---:|---:|---:|
| incumbent | `hybrid_pred` | BASELINE | 0.078581 | 0.486486 | 0.326968 | 0.166667 | 0.012552 |
| `B_oracle_0p05` | `derived_hard_reduction_hybrid_pred` | BASELINE | 0.424585 | 0.162162 | 0.081279 | 0.000000 | 0.017512 |
| `Q_v0_rank0p5` | `q_pred` | REJECT | 0.548923 | 0.270270 | 0.173058 | 0.166667 | 0.017697 |
| `Q_v0_rank1p0` | `q_pred` | REJECT | 0.576224 | 0.135135 | 0.076622 | 0.166667 | 0.017712 |
| `Q_v0_rank2p0` | `q_pred` | REJECT | 0.538813 | 0.351351 | 0.188922 | 0.500000 | 0.019468 |
| `Q_v0_value1_rank1` | `q_pred` | REJECT | 0.495627 | 0.324324 | 0.300224 | 0.166667 | 0.012273 |

## Interpretation

Q-only 不是没学到信号。

证据：

- 4 个 Q 变体 expanded Spearman 全部超过 `B_oracle_0p05` 的 `0.424585`。
- `Q_v0_rank1p0` expanded Spearman 达到 `0.576224`，并且 expanded negative top1 `0.135135` 优于 `B_oracle_0p05`。
- `Q_v0_value1_rank1` transfer Spearman `0.300224`，接近 incumbent `0.326968`。
- `Q_v0_value1_rank1` transfer regret `0.012273`，略好于 incumbent `0.012552`。

但没有 promote。

原因：

- `Q_v0_rank1p0` expanded 好，但 transfer regret `0.017712` 比门槛 `0.0175` 略差。
- `Q_v0_value1_rank1` transfer 好，但 expanded negative top1 `0.324324` 太高。
- `Q_v0_rank2p0` rank 权重过大，transfer negative top1 变差到 `0.500000`。

## Conclusion

直接 Q 收敛方向是有效的，但第一版 Q-only 还没有同时解决：

- expanded 排序强
- expanded top1 安全
- transfer regret 低
- transfer negative top1 安全

最接近可用的两个方向：

- `Q_v0_rank1p0`: 排序最强，安全性较好，但 transfer regret 略差。
- `Q_v0_value1_rank1`: transfer 最接近 incumbent，但 expanded top1 安全性差。

下一步不应该回到旧 multi-head。

更直接的下一步：

- 在 Q-only 上加 `lambda_candidate = 0.1 / 0.3`。
- 或者加入 top1 safety/listwise loss，专门压 negative top1 和 regret。
- 以 `Q_v0_rank1p0` 和 `Q_v0_value1_rank1` 为起点继续，而不是从 A/B 旧头继续。

## Outputs

- Q report: `autoresearch/q-oracle-260629/final_report.md`
- Q summary: `autoresearch/q-oracle-260629/q_oracle_summary.tsv`
- Q handoff: `autoresearch/q-oracle-260629/handoff.json`
