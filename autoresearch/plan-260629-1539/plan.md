# Plan: 简化 hard-count / hard-reduction 训练头

## 目标

这次不先追求 oracle 排序提升。

先回答一个更基础的问题：

```text
哪些预测头是必要的？
哪些预测头其实可以从节点级 hard 标签算出来？
```

要比较两个版本：

```text
版本 A：保留 hard_reduction 头，关掉 hard_count / FC / return
版本 B：只保留节点级 hard 标签，用节点 hard 预测算 hard_count 和 hard_reduction
```

这里的 `hard` 可以理解成：

```text
这个节点上的故障还难不难测。
```

节点级 hard 标签就是：

```text
每个节点分别预测 SA0 / SA1 是否难测。
```

## 为什么先不加 oracle ranking

上一个 scratch sweep 结果说明：

```text
新训练模型的 hard-fault 主能力严重低于 incumbent。
```

也就是：

```text
模型基础判断能力还没稳住。
```

所以这次先只做结构简化，不加 oracle ranking。  
等简化版本能保持 hard 能力后，再加 oracle ranking。

## 术语说明

### hard_count

表示：

```text
还有多少难测故障。
```

现在代码里 `hard_count_post` 是每个节点一个数，不是简单的 0/1。  
它来自 `total_undetected_faults`，然后做了 log 归一化。

所以版本 B 如果只用节点 hard 标签来算 count，算出来的是：

```text
预计难测节点/故障数量
```

它和旧的 `hard_count_head` 目标不完全一样，但方向一致。

### hard_reduction

表示：

```text
一个动作让难测故障减少了多少。
```

旧做法是直接用 `hard_reduction_head` 预测。

版本 B 的新做法是：

```text
动作前 hard_count - 动作后 hard_count
```

### FC

FC 是 fault coverage，表示：

```text
总体测到了多少故障。
```

这次先关掉它，因为它和 hard_count 有重叠，而且之前 reward / FC 类分数容易把负收益动作排前面。

### return

return 表示：

```text
做多个动作以后，最后一共变好了多少。
```

这次也先关掉。以后如果需要，可以用：

```text
当前 hard_count - 多步之后 hard_count
```

直接算出来。

## 版本 A：保留 hard_reduction 头

### 思路

版本 A 是保守简化。

保留：

```text
节点级 hard 标签预测
hard_reduction 头
```

关闭：

```text
hard_count 头的损失
FC / reward 损失
return 损失
```

含义是：

```text
模型仍然直接学习“这个动作减少多少 hard fault”，
但不再额外学习“动作后每个节点的 hard_count 数值”。
```

### 训练权重

建议先用和 incumbent 更接近的基础配方，不使用上次 scratch sweep 里过度改动后的配方。

```text
lambda_jepa              = 1.0
lambda_scoap             = 0.5
lambda_delta_scoap       = 0.3
lambda_hard              = 0.5
lambda_hard_count        = 0.0
lambda_hard_reduction    = 0.5
lambda_hard_soft_f1      = 0.02
lambda_hard_rank         = 0.0
lambda_hard_brier        = 0.0
lambda_fc                = 0.0
lambda_return            = 0.0
lambda_pattern           = 0.0
lambda_oracle_rank       = 0.0
lambda_oracle_value      = 0.0
```

### 需要改代码吗

版本 A 基本不需要改模型结构。

只需要生成 config：

```text
lambda_hard_count = 0.0
lambda_fc = 0.0
lambda_return = 0.0
```

旧的 `hard_count_head` 可以先留在模型里，但不训练它。

原因：

```text
这样 checkpoint 格式不变，风险最小。
```

### 评估

版本 A 仍然可以用旧 evaluator：

```text
scripts/evaluate_hard_checkpoints.py
scripts/evaluate_oracle_action_values.py
```

因为 `hard_reduction_head` 还在，所以：

```text
hard_reduction_total_pred
hybrid_pred
```

仍然有意义。

## 版本 B：只保留节点级 hard 标签

### 思路

版本 B 是激进简化。

只直接训练：

```text
节点级 hard 标签预测
```

不再直接训练：

```text
hard_count
hard_reduction
FC / reward
return
```

然后从节点 hard 预测里算：

```text
hard_count
hard_reduction
planner score
```

### 怎么从节点 hard 算 hard_count

模型对每个节点输出两个概率：

```text
P(SA0 难测)
P(SA1 难测)
```

先定义：

```text
sa0_count = 所有节点的 P(SA0 难测) 加起来
sa1_count = 所有节点的 P(SA1 难测) 加起来
total_count = sa0_count + sa1_count
```

这就是模型认为：

```text
当前还有多少难测 fault。
```

### 怎么从节点 hard 算 hard_reduction

对同一个候选动作，算两次：

```text
动作前 hard_count
动作后 hard_count
```

然后：

```text
hard_reduction_total = (动作前 total_count - 动作后 total_count) / max(1, 动作前 total_count)
hard_reduction_sa0   = (动作前 sa0_count   - 动作后 sa0_count)   / max(1, 动作前 sa0_count)
hard_reduction_sa1   = (动作前 sa1_count   - 动作后 sa1_count)   / max(1, 动作前 sa1_count)
```

这样得到的 3 个数可以替代旧的：

```text
hard_reduction_head 输出
```

### 关键实现点

版本 B 需要新增一个配置开关：

```text
hard_value_mode = "derived_from_node_hard"
```

当这个开关打开时：

1. 训练时关闭 hard_count 和 hard_reduction direct loss。
2. evaluator 里新增 derived 指标。
3. planner 里用 derived hard_reduction 作为动作分数。
4. oracle gate 也能选择 derived score field。

### 模型是否真的要删除 head

第一阶段不建议物理删除：

```text
hard_count_head
hard_reduction_head
reward_head
return_head
```

建议先保留模块，但不训练、不使用。

原因：

```text
删除 head 会破坏 checkpoint 兼容性。
先不用它们，证明思路有效后再删。
```

### 训练权重

版本 B 建议：

```text
lambda_jepa              = 1.0
lambda_scoap             = 0.5
lambda_delta_scoap       = 0.3
lambda_hard              = 0.5
lambda_hard_count        = 0.0
lambda_hard_reduction    = 0.0
lambda_hard_soft_f1      = 0.02
lambda_hard_rank         = 0.0
lambda_hard_brier        = 0.0
lambda_fc                = 0.0
lambda_return            = 0.0
lambda_pattern           = 0.0
lambda_oracle_rank       = 0.0
lambda_oracle_value      = 0.0
hard_value_mode          = "derived_from_node_hard"
```

### 需要新增的代码

#### 1. 新增 hard 概率计数函数

放在 `tpi_jepa/model.py` 或 `tpi_jepa/train.py` 可复用位置：

```text
hard_logits -> hard counts
```

逻辑：

```text
prob = sigmoid(hard_logits)
sa0_count = sum(prob[:, 0])
sa1_count = sum(prob[:, 1])
total_count = sa0_count + sa1_count
```

#### 2. forward 输出 pre/post hard logits

现在模型只输出动作后的：

```text
hard_logits
```

版本 B 要算 reduction，需要同时拿到：

```text
pre_hard_logits
post_hard_logits
```

可以在 `TPIWorldModel.forward()` 里增加：

```text
pre_hard_logits
derived_hard_count_pred
derived_hard_reduction_pred
```

注意：

```text
如果 hard_head_type = residual_context，
pre 和 post 都要用同一份 relation_features。
```

这样比较的是：

```text
同一个候选动作相关区域里，动作前后 hard 概率变化。
```

#### 3. planner 使用 derived score

在 `tpi_jepa/plan.py` 的 `score_candidate_from_latent()` 增加新分数字段：

```text
derived_hard_reduction_total_pred
derived_hard_reduction_sa0_pred
derived_hard_reduction_sa1_pred
derived_hard_count_pre_pred
derived_hard_count_post_pred
```

然后新增可选 planner score：

```text
derived_hard_reduction_total_pred
```

或者：

```text
derived_hard_reduction_total_pred * coverage_scale
```

#### 4. oracle probe / gate 支持新 score field

在这些脚本里把新字段加入 score 列表：

```text
scripts/oracle_action_value_probe.py
scripts/evaluate_oracle_action_values.py
scripts/finetune_oracle_action_values.py
```

新增：

```text
derived_hard_reduction_total_pred
```

否则版本 B 的 oracle 排序评估看不到真正分数。

#### 5. hard evaluator 支持 derived metrics

在 `scripts/evaluate_hard_checkpoints.py` 里新增：

```text
derived_hard_count_mae
derived_hard_reduction_mae
derived_hard_reduction_sign_acc
derived_hard_reduction_score
```

版本 B 的 promote 判断应该用：

```text
derived_hard_reduction_score
```

而不是旧的：

```text
hard_reduction_score
```

## 实验设计

### 第一阶段：不加 oracle

先跑 3 个模型：

```text
control：incumbent-like 配方，不简化
version_A：关 hard_count / FC / return，保留 hard_reduction
version_B：只用节点 hard 标签，derived hard count/reduction
```

建议 seed：

```text
seed = 2030
```

先只用一个 seed，确认方向。  
如果版本 A 或 B 接近 incumbent，再扩到 3 个 seed。

### 第二阶段：通过后再加 oracle

只有当某个版本满足基础 gate，才加：

```text
lambda_oracle_rank = 0.01 / 0.03 / 0.05
```

不要再直接上：

```text
0.10 / 0.20
```

因为上次已经看到大权重会破坏基础能力。

## Gate 设计

### 基础 hard gate

必须比较：

```text
incumbent
control
version_A
version_B
```

版本 A 看：

```text
hard_macro_f1_tuned
hard_reduction_score
predictive_score
```

版本 B 看：

```text
hard_macro_f1_tuned
derived_hard_reduction_score
derived_hard_reduction_sign_acc
```

### oracle validation gate

用已有固定 oracle actions：

```text
autoresearch/oracle-action-probe-260629-expanded-subckt-val/oracle_actions.tsv
```

版本 A score：

```text
hard_reduction_total_pred
hybrid_pred
```

版本 B score：

```text
derived_hard_reduction_total_pred
```

### transfer gate

用：

```text
autoresearch/oracle-action-probe-260629-smoke/oracle_actions.tsv
```

目的：

```text
看小子图上学到的排序，是否还能迁移到大电路。
```

## Promote 条件

### 版本 A

版本 A 可以 promote 进入 oracle 阶段的条件：

```text
hard_macro_f1_tuned >= incumbent - 0.03
hard_reduction_score >= incumbent - 0.03
expanded oracle Spearman 不低于 incumbent - 0.02
transfer oracle Spearman 不低于 incumbent - 0.02
negative_top1 不变差
```

### 版本 B

版本 B 可以 promote 进入 oracle 阶段的条件：

```text
hard_macro_f1_tuned >= incumbent - 0.03
derived_hard_reduction_score 有效，不是接近 0
derived_hard_reduction_sign_acc >= 0.55
expanded oracle Spearman 不低于 incumbent - 0.02
transfer oracle Spearman 不低于 incumbent - 0.02
negative_top1 不变差
```

版本 B 不要求旧的 `hard_reduction_score`，因为旧 head 不再使用。

## 风险

### 版本 A 风险

风险较小。

可能的问题：

```text
hard_count_head 被关掉后，节点 hard 表征变弱。
```

但如果 `lambda_hard` 够强，这个风险应该可控。

### 版本 B 风险

风险较大。

主要问题：

```text
节点级 hard 标签只告诉模型每个节点是否 hard，
不直接告诉模型一个动作减少了多少 hard fault。
```

所以版本 B 可能 hard F1 很好，但动作排序不一定好。

这就是为什么必须新增：

```text
derived_hard_reduction oracle gate
```

否则只看 hard F1 会误判。

## 推荐执行顺序

1. 先实现版本 A config，不改代码。
2. 跑版本 A 和 control，确认关掉 hard_count / FC / return 是否安全。
3. 再实现版本 B 的 derived hard-count / reduction 代码。
4. 跑版本 B。
5. 三者都过 hard gate 后，再考虑加小权重 oracle ranking。

## 最终建议

优先做版本 A。

原因：

```text
它能直接回答 hard_count 头是否多余，
实现风险最低。
```

版本 B 是更干净的最终方向，但要先补 evaluator 和 planner 的 derived score。  
否则即使训练成功，也无法正确判断它是否真的会选动作。
