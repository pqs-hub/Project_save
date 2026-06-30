# Plan: Q Calibration With Fixed Candidate Ablation

generated_at: `2026-06-30T01:24:00+08:00`

## Goal

下一步只做一件事：

`Q calibration`

不要继续：

- 不加新 loss。
- 不调 candidate。
- 不做 hybrid。
- 不新增旧 head。
- 不继续做 reward / return / hard-reduction 混合。

当前问题不是 Q 学不到排序，而是 Q 的分数尺度和 top1 安全性不稳定。

## Evidence From Last Run

上一轮 Q-only 结果：

| variant | expanded Spearman | expanded neg top1 | transfer Spearman | transfer neg top1 | transfer regret |
|---|---:|---:|---:|---:|---:|
| incumbent | 0.078581 | 0.486486 | 0.326968 | 0.166667 | 0.012552 |
| `B_oracle_0p05` | 0.424585 | 0.162162 | 0.081279 | 0.000000 | 0.017512 |
| `Q_v0_rank0p5` | 0.548923 | 0.270270 | 0.173058 | 0.166667 | 0.017697 |
| `Q_v0_rank1p0` | 0.576224 | 0.135135 | 0.076622 | 0.166667 | 0.017712 |
| `Q_v0_rank2p0` | 0.538813 | 0.351351 | 0.188922 | 0.500000 | 0.019468 |
| `Q_v0_value1_rank1` | 0.495627 | 0.324324 | 0.300224 | 0.166667 | 0.012273 |

结论：

- Q 排序信号已经存在。
- rank-heavy 版本 expanded 排序强。
- value-heavy 版本 transfer regret 好。
- 失败点不是“再多一个 loss”，而是同一个 `q_pred` 在不同 group/circuit 上 scale 不稳定，导致 top1 选择不稳。

## Core Hypothesis

`q_pred` 的绝对值不能直接跨 circuit / candidate group 比较。

应该比较的是校准后的相对分数：

```text
q_cal = (q - mu) / sigma
```

或者：

```text
q_cal = sigmoid((q - mu) / sigma)
```

其中 `mu/sigma` 可以按不同粒度估计：

- per candidate group
- per circuit
- global train split
- train-fitted calibration model

## What To Implement

### 1. Add Post-hoc Q Calibration Evaluator

新增脚本：

`scripts/evaluate_q_calibration.py`

输入：

- 已经 rescored 的 oracle action TSV，或直接 checkpoint + oracle action TSV。
- score field: `q_pred`。
- fixed candidate group。

输出：

- 每种 calibration method 的 expanded / transfer metrics。
- 同一 checkpoint 内 raw Q vs calibrated Q 对比。

### 2. Calibration Methods

第一轮只做 post-hoc，不改训练。

必须包含：

| method | formula | 用途 |
|---|---|---|
| `raw` | `q` | baseline |
| `group_zscore` | `(q - mean(group)) / std(group)` | 消除同一候选池内 scale |
| `group_center` | `q - mean(group)` | 只消除偏置，不改方差 |
| `group_rank_pct` | rank percentile | 只保留顺序，完全丢弃 scale |
| `circuit_zscore` | `(q - mean(circuit)) / std(circuit)` | 消除 circuit-level scale |
| `global_zscore` | `(q - train_mean) / train_std` | 全局校准 |
| `platt` | `sigmoid(a*q+b)` | 用 train 拟合概率式校准 |

暂不做 isotonic。

原因：

- isotonic 容易在当前 oracle group 数量下过拟合。
- 第一轮先验证 scale normalization 是否能稳住 negative top1 / regret。

### 3. Fixed Candidate Only

候选池必须固定。

本轮只允许：

```text
candidate_strategy = hard_fault_recall_union
```

或者已有 oracle 表里的固定 union/cached union。

禁止同时改：

- candidate strategy
- diversity penalty
- beam objective
- hybrid score

目的：

只测 Q variant / Q calibration，不让 candidate 变量污染结果。

### 4. Q Variant Set

只比较上一轮已有 checkpoint，不重新训练：

- `Q_v0_rank0p5`
- `Q_v0_rank1p0`
- `Q_v0_rank2p0`
- `Q_v0_value1_rank1`

重点候选：

- `Q_v0_rank1p0`: expanded 最强，先看 calibration 能否降低 transfer regret。
- `Q_v0_value1_rank1`: transfer 最强，先看 calibration 能否降低 expanded negative top1。

### 5. Metrics

每个 calibration method 输出：

| metric | meaning |
|---|---|
| `mean_spearman` | group 内排序相关性 |
| `mean_kendall_tau` | pairwise 顺序稳定性 |
| `mean_pearson` | value calibration 相关性 |
| `negative_top1_rate` | 选第一名时是否经常选到负收益 |
| `mean_top1_real_delta_tc` | 被模型选中的 top1 真实收益 |
| `mean_top1_regret` | top1 和 oracle best 的差距 |
| `mean_sign_accuracy` | 正负收益方向是否判断对 |
| `calibration_slope/intercept` | Q 分数和真实 delta_TC 的线性关系 |

## Promotion Criteria

基线：

| baseline | expanded Spearman | expanded neg top1 | transfer Spearman | transfer neg top1 | transfer regret |
|---|---:|---:|---:|---:|---:|
| incumbent | 0.078581 | 0.486486 | 0.326968 | 0.166667 | 0.012552 |
| `B_oracle_0p05` | 0.424585 | 0.162162 | 0.081279 | 0.000000 | 0.017512 |

Promote 条件：

- expanded Spearman >= `0.50`
- expanded negative top1 <= `0.162`
- transfer negative top1 <= `0.167`
- transfer regret <= `0.012552`
- transfer Spearman >= `0.20`

如果某个 calibration 只满足 expanded，不满足 transfer：

`REJECT`

如果 transfer regret 好但 expanded negative top1 高：

`REJECT`

只有同时满足安全和 transfer，才进入 8-circuit planner eval。

## Expected Outcome

最可能成功的组合：

| checkpoint | calibration | why |
|---|---|---|
| `Q_v0_rank1p0` | `group_zscore` / `group_rank_pct` | expanded 排序最强，去 scale 后可能降低 regret |
| `Q_v0_value1_rank1` | `group_zscore` / `circuit_zscore` | transfer 好，去 group 偏置后可能降低 expanded negative top1 |

## Implementation Steps

### Step 1: Build Calibration Metrics

新增：

```text
scripts/evaluate_q_calibration.py
```

它读取：

```text
autoresearch/q-oracle-260629/gates/expanded_val/rescored_oracle_actions.tsv
autoresearch/q-oracle-260629/gates/transfer/rescored_oracle_actions.tsv
```

直接基于已有 rescored actions 计算校准分数，不重新跑 checkpoint。

这样最快，也保证 candidate/action set 完全固定。

### Step 2: Run Full Ablation

输出目录：

```text
autoresearch/q-calibration-260630
```

命令：

```bash
python scripts/evaluate_q_calibration.py \
  --expanded-rescored autoresearch/q-oracle-260629/gates/expanded_val/rescored_oracle_actions.tsv \
  --transfer-rescored autoresearch/q-oracle-260629/gates/transfer/rescored_oracle_actions.tsv \
  --checkpoints Q_v0_rank0p5,Q_v0_rank1p0,Q_v0_rank2p0,Q_v0_value1_rank1 \
  --score-field q_pred \
  --methods raw,group_center,group_zscore,group_rank_pct,circuit_zscore,global_zscore,platt \
  --out-dir autoresearch/q-calibration-260630
```

### Step 3: Decide

如果有 promote：

- 用 promote 的 calibrated Q score 进入 fixed `hard_fault_recall_union` planner eval。
- 仍然不改 candidate。

如果没有 promote：

- 不加新 loss。
- 回到 calibration 粒度分析：哪些 circuit/group 造成 negative top1。
- 扩 oracle transfer group，而不是继续改模型结构。

## Non-goals

本轮明确不做：

- 不训练新 checkpoint。
- 不加 candidate-aware loss。
- 不加 top1 safety loss。
- 不调 candidate strategy。
- 不跑 hybrid score。
- 不删除旧字段。

## Next Command

```bash
$autoresearch fix autoresearch/plan-260630-0124/plan.md
```

执行时只实现 Q calibration evaluator 和 fixed candidate ablation。
