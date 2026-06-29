# Plan: 原训练任务 + Oracle Pairwise Ranking Loss

## 背景判断

当前目标不是继续做 head-only finetune，而是重新训练完整模型：保留原来的世界模型 / hard-fault 训练任务，同时把 backend 标注出来的 oracle action group 加入主训练，让 planner score 的排序更接近真实 `delta_tc` 排序。

已有实验说明：

- 只训练 `reward_head/return_head` 太弱，held-out oracle ranking 没有稳定提升。
- 联合训练 action encoder / dynamics 可以移动 ranking，但容易破坏 transfer。
- 冻结 hard head 仍然退化，说明风险来自上游 latent drift。
- 因此这次应该用小权重 oracle ranking，让原任务继续约束表征，不让 oracle 小数据集主导训练。

## 当前训练损失入口

主训练里的原始 total loss 由这些项组成：

```text
total =
  lambda_jepa * jepa_loss
  + lambda_scoap * scoap_loss
  + lambda_delta_scoap * delta_scoap_loss
  + lambda_hard * hard_bce
  + lambda_hard_rank * hard_rank
  + lambda_hard_brier * hard_brier
  + lambda_hard_soft_f1 * hard_soft_f1
  + lambda_hard_count * hard_count_loss
  + lambda_hard_reduction * hard_reduction_loss
  + lambda_fc * weighted_reward_loss
  + lambda_pattern * pattern_loss
  + lambda_return * weighted_return_loss
```

现在主训练还没有 oracle ranking loss，需要新增：

```text
total = original_total + current_lambda_oracle_rank * oracle_pairwise_rank_loss
```

其中 `current_lambda_oracle_rank` 应该支持 warmup / ramp，避免训练早期随机 latent 被 oracle 小样本强行拉偏。

## 推荐权重原则

### 保留的核心锚点

`lambda_jepa = 1.0`

保留为最大结构锚点。它约束 action 后的 latent prediction，避免模型只为了 oracle 排序改 planner score，而忘掉状态转移。

`lambda_hard_reduction = 0.5`

保持不降。历史上最稳的 planner 排序信号来自 `hard_reduction_total_pred` / hybrid，而不是裸 `reward_pred`。oracle ranking 应该校准这个信号，不应该替代它。

`lambda_hard = 0.5`

使用 incumbent 高分 run 的经验值，而不是 base config 里的 0.7。0.5 已经能提供 hard-fault 监督，同时给 oracle ranking 留出梯度空间。

`lambda_hard_count = 0.1`

保持不变。它是 hard-fault 数量尺度约束，权重小但有助于稳定。

`lambda_hard_soft_f1 = 0.02`

保持不变。它是之前高分 run 的轻量补充项，不应该在这次 sweep 中同时改动。

### 下调的辅助代理项

`lambda_scoap = 0.3`

从 0.5 降到 0.3。SCOAP 是有用的测试性代理任务，但它和最终 `delta_tc` 排序不是同一个目标。加入 oracle ranking 后，SCOAP 不应该占用太多梯度预算。

`lambda_delta_scoap = 0.2`

从 0.3 降到 0.2。delta SCOAP 仍然保留，用来约束 action effect 的方向性，但让位给 oracle action-value 排序。

### 暂时关闭的项

`lambda_fc = 0.0`

继续关闭。它用的是采样标签里的 coverage delta，不等价于 backend-labeled candidate action 的真实 `delta_tc` 排序；之前 reward 类 score 也更容易把负收益 action 排第一。

`lambda_return = 0.0`

暂时关闭。先只引入一个 oracle pairwise ranking loss，不同时加 value/return 监督，避免目标太多导致不可解释。

`lambda_pattern = 0.0`

保持关闭。当前问题是 action 排序，不是 pattern count 预测。

`lambda_hard_rank = 0.0`

保持关闭。oracle pairwise ranking 已经是排序目标；同时打开 hard-rank 可能和真实 `delta_tc` 排序竞争。

`lambda_hard_brier = 0.0`

保持关闭。先不要增加 calibration 目标。

## 权重 Sweep 设计

### Control: 原任务重训基线

```json
{
  "lambda_jepa": 1.0,
  "lambda_scoap": 0.3,
  "lambda_delta_scoap": 0.2,
  "lambda_hard": 0.5,
  "lambda_hard_count": 0.1,
  "lambda_hard_reduction": 0.5,
  "lambda_hard_soft_f1": 0.02,
  "lambda_hard_rank": 0.0,
  "lambda_hard_brier": 0.0,
  "lambda_fc": 0.0,
  "lambda_return": 0.0,
  "lambda_pattern": 0.0,
  "lambda_oracle_rank": 0.0,
  "lambda_oracle_value": 0.0
}
```

用途：区分“重新训练本身带来的变化”和“oracle ranking loss 带来的变化”。

### Oracle 0.05: 主推保守档

```json
{
  "lambda_jepa": 1.0,
  "lambda_scoap": 0.3,
  "lambda_delta_scoap": 0.2,
  "lambda_hard": 0.5,
  "lambda_hard_count": 0.1,
  "lambda_hard_reduction": 0.5,
  "lambda_hard_soft_f1": 0.02,
  "lambda_hard_rank": 0.0,
  "lambda_hard_brier": 0.0,
  "lambda_fc": 0.0,
  "lambda_return": 0.0,
  "lambda_pattern": 0.0,
  "lambda_oracle_rank": 0.05,
  "lambda_oracle_value": 0.0
}
```

用途：最可能不伤 transfer。它应该先作为 promotion 候选。

### Oracle 0.10: 平衡档

```json
{
  "lambda_jepa": 1.0,
  "lambda_scoap": 0.3,
  "lambda_delta_scoap": 0.2,
  "lambda_hard": 0.5,
  "lambda_hard_count": 0.1,
  "lambda_hard_reduction": 0.5,
  "lambda_hard_soft_f1": 0.02,
  "lambda_hard_rank": 0.0,
  "lambda_hard_brier": 0.0,
  "lambda_fc": 0.0,
  "lambda_return": 0.0,
  "lambda_pattern": 0.0,
  "lambda_oracle_rank": 0.10,
  "lambda_oracle_value": 0.0
}
```

用途：如果 0.05 信号太弱，0.10 是主要增强档。

### Oracle 0.20: 压力测试档

```json
{
  "lambda_jepa": 1.0,
  "lambda_scoap": 0.3,
  "lambda_delta_scoap": 0.2,
  "lambda_hard": 0.5,
  "lambda_hard_count": 0.1,
  "lambda_hard_reduction": 0.5,
  "lambda_hard_soft_f1": 0.02,
  "lambda_hard_rank": 0.0,
  "lambda_hard_brier": 0.0,
  "lambda_fc": 0.0,
  "lambda_return": 0.0,
  "lambda_pattern": 0.0,
  "lambda_oracle_rank": 0.20,
  "lambda_oracle_value": 0.0
}
```

用途：只作为上限测试。除非 oracle val 明显提升且 transfer 不退化，否则不 promote。

## Oracle Ranking Loss 细节

### 标签

每个 action 的标签是 backend 重新评估得到的：

```text
oracle_delta_tc = action 后 test_coverage - 初始 test_coverage
```

pairwise ranking 只关心同一个 `group_id` 内 action 之间的相对顺序：

```text
oracle_delta_tc(a_i) > oracle_delta_tc(a_j)
=> score(a_i) 应该 > score(a_j)
```

### Score

第一版建议用 `hybrid_pred` 做排序 score：

```text
hybrid_pred = hard_reduction_total_pred * coverage_scale + reward_pred
```

原因：

- `hard_reduction_total_pred` 是目前最稳定的排序来源。
- `reward_pred` 可以作为残差，但不能单独主导。
- 训练目标是最终 planner score 排序一致，不是单独让某个新 head 好看。

如果实现成本允许，可以把 score 写成可配置：

```text
oracle_ranking_score_field = hybrid_pred
```

后续再比较 `hard_reduction_total_pred` 和 `bounded_residual_hybrid_pred`。

### Pairwise Loss

沿用现有 finetune 脚本里的 pairwise ranking 形式：

```text
loss_ij = softplus(-(score_i - score_j) / temperature)
```

只对 `abs(oracle_delta_tc_i - oracle_delta_tc_j) >= pairwise_min_delta` 的 pair 计算，避免 backend 噪声和近似 tie 误导训练。

推荐：

```text
pairwise_min_delta = 0.001
pairwise_temperature = 1.0
```

因为训练里 target 会乘 `coverage_scale=100`，实际 pair threshold 是 `0.1` 个 scaled unit。

## 训练调度

### Warmup / Ramp

推荐不要从第 1 个 epoch 就满权重加 oracle：

```text
oracle_warmup_epochs = 1
oracle_ramp_epochs = 2
```

实际权重：

```text
epoch 1: 0
epoch 2: 0.5 * lambda_oracle_rank
epoch 3+: 1.0 * lambda_oracle_rank
```

原因：scratch retrain 早期 planner score 还没有语义，oracle pairwise 梯度容易把 action encoder / dynamics 拉向小样本捷径。

### 混合频率

不要每个原训练 batch 都叠一个 oracle group。建议：

```text
oracle_every_n_steps = 4
oracle_batch_groups = 4
```

含义：

- 每 4 个原训练 step 做一次 oracle ranking step。
- 每次 oracle step 采 4 个 action group。
- 这样 oracle 是校准项，不是主任务。

如果 GPU 时间足够，第二轮可以试：

```text
oracle_every_n_steps = 2
oracle_batch_groups = 4
```

但第一轮不建议。

## 实现范围

### 需要改主训练

在 `tpi_jepa/train.py` 中新增：

- `lambda_oracle_rank`
- `lambda_oracle_value`，默认 0，只为以后保留
- `oracle_actions`
- `val_oracle_actions`
- `oracle_ranking_score_field`
- `oracle_every_n_steps`
- `oracle_batch_groups`
- `oracle_warmup_epochs`
- `oracle_ramp_epochs`
- `oracle_pairwise_min_delta`
- `oracle_pairwise_temperature`

并把现有 `scripts/finetune_oracle_action_values.py` 中已经可用的 oracle group loading、action scoring、pairwise loss 逻辑抽出来复用，避免两份实现漂移。

### 需要新增配置生成

新增 4 个 config 或 run variants：

```text
scratch_oracle_rank_0p00
scratch_oracle_rank_0p05
scratch_oracle_rank_0p10
scratch_oracle_rank_0p20
```

每个 variant 只改 `lambda_oracle_rank`，不要同时改 seed、hard loss、sampler、edge mode，否则无法归因。

## 验证 Gate

### 必跑验证

1. Python 编译检查：

```bash
python -m py_compile tpi_jepa/train.py scripts/finetune_oracle_action_values.py scripts/evaluate_oracle_action_values.py
```

2. 小步 smoke train：

```bash
python -m tpi_jepa.train --config <generated_config> --max-steps <small>
```

3. 原 hard-fault checkpoint eval：

```text
对比原 incumbent、scratch control、oracle 0.05、0.10、0.20
```

4. 固定 oracle val gate：

```text
oracle_actions = autoresearch/oracle-action-probe-260629-expanded-subckt-val/oracle_actions.tsv
score_fields = hybrid_pred, hard_reduction_total_pred, bounded_residual_hybrid_pred
```

5. transfer gate：

```text
oracle_actions = autoresearch/oracle-action-probe-260629-smoke/oracle_actions.tsv
score_fields = hybrid_pred, hard_reduction_total_pred, bounded_residual_hybrid_pred
```

### Promote 条件

候选 checkpoint 必须同时满足：

```text
expanded val hybrid Spearman >= max(control, incumbent) + 0.02
expanded val negative_top1_rate <= max(control, incumbent) 不更差
expanded val top1_regret <= max(control, incumbent) 不更差
transfer hybrid Spearman >= incumbent - 0.02
transfer negative_top1_rate <= incumbent
hard-fault 主指标不明显退化
```

如果 0.20 只提升 oracle val 但 transfer 下降，判定为 overfit，不 promote。

## 预期解释

如果 0.05 有提升、0.10 更好、0.20 退化：

说明 oracle ranking 是有效校准项，但小数据权重要受控。

如果 0.05/0.10/0.20 都不提升：

说明当前 oracle groups 数量或 score 参数化不足，需要先扩大 action group 或改 score 结构，而不是继续加权重。

如果 oracle val 提升但 transfer 明显退化：

说明模型学到了 131 个子图分布里的局部排序捷径，需要增加跨电路 oracle group 或冻结部分上游结构。

如果 control 已经比 incumbent 好：

说明之前 checkpoint 的提升主要来自训练随机性 / 配置重训，而不是 oracle loss。此时应该先把 control 当新 baseline，再重新比较 oracle variants。

## 推荐执行顺序

1. 实现主训练 oracle pairwise ranking loss，但默认关闭。
2. 先跑 `lambda_oracle_rank=0.0` control，确认重训基线。
3. 跑 `0.05`，只要 transfer 不坏就继续。
4. 跑 `0.10`，看 oracle val 是否有单调提升。
5. 跑 `0.20`，作为过拟合压力测试。
6. 用固定 oracle gate 和 hard-fault eval 决定是否 promote。

## 最终建议

第一轮权重分配采用：

```text
lambda_jepa              = 1.0
lambda_scoap             = 0.3
lambda_delta_scoap       = 0.2
lambda_hard              = 0.5
lambda_hard_count        = 0.1
lambda_hard_reduction    = 0.5
lambda_hard_soft_f1      = 0.02
lambda_hard_rank         = 0.0
lambda_hard_brier        = 0.0
lambda_fc                = 0.0
lambda_return            = 0.0
lambda_pattern           = 0.0
lambda_oracle_value      = 0.0
lambda_oracle_rank       = 0.0 / 0.05 / 0.10 / 0.20
```

这不是把原任务整体缩小后硬塞 oracle，而是保留结构和 hard-fault 主监督，只让 SCOAP 辅助项让出一部分预算。这个分配最符合当前实验经验：oracle ranking 应该作为真实收益排序校准项，而不是替代原世界模型训练。
