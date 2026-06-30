# Fix Report: Scheme A/B Oracle-Ranking Sweep

generated_at: `2026-06-29T22:04:00+08:00`

## What Changed

本轮把 `A/B + oracle pairwise ranking loss` 的剩余训练从串行改成并行跑：

- 独立训练变体使用 `--parallel-devices cuda:4,cuda:5,cuda:6,cuda:7`。
- 已完成的 checkpoint 自动跳过，避免重复训练。
- hard-fault gate 并行评估各 checkpoint。
- oracle expanded/transfer gate 仍按数据集顺序跑，因为它一次性比较同一批 checkpoint，需要固定同一张 oracle 表。

之后所有类似 sweep，如果各配置互不依赖，默认按多 GPU / 多 CPU 并行执行；只有存在共享写路径、显存不足、或前后依赖时才串行。

## Runs

实验目录：

`autoresearch/train-ab-oracle-rank-260629`

最终统一评估的 6 个变体：

- `A_oracle_0p01`
- `A_oracle_0p03`
- `A_oracle_0p05`
- `B_oracle_0p01`
- `B_oracle_0p03`
- `B_oracle_0p05`

每个变体都用同一套 gate 判断：

- hard-fault eval：检查基础 hard label 能力有没有崩。
- expanded oracle val：检查在更宽、更负样本丰富的 oracle action group 上，模型能不能把真实收益高的 action 排前面。
- transfer oracle gate：检查同一模型排序能力能不能迁移到旧的 transfer oracle action group。

## Final Verdict

唯一 promote：

`B_oracle_0p05`

判定为：

`PROMOTE_GUARDED_RERANK`

含义：

- 可以作为受保护的 rerank 候选继续试。
- 不是强替代 incumbent。
- 原因是 expanded val 明显变好，transfer 没有把负收益 action 放到第一，但 transfer regret 没有优于 incumbent。

关键指标：

| variant | verdict | score | expanded Spearman | expanded neg top1 | transfer Spearman | transfer neg top1 | transfer regret | hard F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| incumbent | BASELINE | `hybrid_pred` | 0.078579 | 0.486486 | 0.326661 | 0.166667 | 0.012552 | nan |
| A_oracle_0p03 | REJECT | `hard_reduction_total_pred` | -0.002181 | 0.486486 | 0.098460 | 0.000000 | 0.007262 | 0.761439 |
| B_oracle_0p01 | REJECT | `derived_hard_reduction_hybrid_pred` | 0.244340 | 0.216216 | 0.000919 | 0.000000 | 0.022608 | 0.804093 |
| B_oracle_0p03 | REJECT | `derived_hard_reduction_hybrid_pred` | 0.336100 | 0.297297 | 0.074052 | 0.333333 | 0.022492 | 0.805229 |
| B_oracle_0p05 | PROMOTE_GUARDED_RERANK | `derived_hard_reduction_hybrid_pred` | 0.424585 | 0.162162 | 0.081119 | 0.000000 | 0.017512 | 0.769712 |

## Interpretation

方案 B 加 oracle ranking loss 的方向比方案 A 更有希望。

具体看：

- `B_oracle_0p05` 在 expanded val 上排序最强，Spearman 从 incumbent 的 `0.078579` 提到 `0.424585`。
- `B_oracle_0p05` 在 expanded val 上负收益 top1 从 incumbent 的 `0.486486` 降到 `0.162162`。
- `B_oracle_0p05` 在 transfer 上负收益 top1 是 `0.000000`，安全性好。
- 但 `B_oracle_0p05` 的 transfer Spearman `0.081119` 低于 incumbent `0.326661`。
- `B_oracle_0p05` 的 transfer regret `0.017512` 高于 incumbent `0.012552`。

所以结论不是“B 已经全面超过 incumbent”，而是：

`B_oracle_0p05` 学到了 expanded oracle 分布里的排序信号，但这个信号迁移还不够强。它适合进入下一步受保护 planner rerank 测试，不适合直接替换主排序分数。

## Outputs

- Final report: `autoresearch/train-ab-oracle-rank-260629/final_report.md`
- Summary TSV: `autoresearch/train-ab-oracle-rank-260629/ab_oracle_rank_summary.tsv`
- Handoff: `autoresearch/train-ab-oracle-rank-260629/handoff.json`

## Next Step

下一步建议只拿 `B_oracle_0p05` 做 guarded planner rerank：

- 保留 incumbent 原本候选生成和安全过滤。
- 只在候选池内部用 `B_oracle_0p05` 的 `derived_hard_reduction_hybrid_pred` 做 rerank。
- 和 incumbent 在 8 个评估电路上比较真实 TC。
- 如果真实 TC 没提升，说明 expanded oracle val 仍然和最终电路分布不匹配，需要继续扩 transfer/大电路 oracle action group，而不是继续调 loss weight。
