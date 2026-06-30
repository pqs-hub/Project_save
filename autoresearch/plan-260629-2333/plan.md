# Plan: Converge TPI-JEPA to Q(s,a)-Centric Planner

generated_at: `2026-06-29T23:33:00+08:00`

## Goal

把当前 `state -> latent -> 多个 head -> 多种 loss -> hybrid planner` 收敛成：

`state, action -> JEPA encoder/dynamics -> Q(s,a) -> planner ranking`

最终目标不是继续增加辅助头，而是让模型输出的唯一决策分数 `Q(s,a)` 和 backend 标注的真实 `delta_TC(s,a)` 单调一致。

## Current Code Facts

当前代码已经具备部分基础，不需要推倒重写：

- `tpi_jepa/model.py` 已有 `online_encoder`、`action_encoder`、`dynamics`、action-conditioned `_summary(...)`。
- `tpi_jepa/train.py` 已有 oracle action group loader、oracle pairwise ranking loss、warmup/ramp 机制。
- `tpi_jepa/plan.py` 已经用 `score_field` 切换 planner 分数。
- `scripts/evaluate_oracle_action_values.py` 已经能用固定 oracle action table 比较 checkpoint。
- 最近结果里 `B_oracle_0p05` 是当前最强 guarded rerank baseline。

所以正确路线是：先新增 Q 路径并证明它有效，再逐步关掉旧头；不是一次性删除所有旧 head。

## Before -> After Mapping

| Module | Current | Target | Transition Strategy |
|---|---|---|---|
| `features.py` | 输入特征 + SCOAP/hard 监督来源 | 输入特征 only | 保留特征生成；逐步停止把 SCOAP/delta-SCOAP 当训练 target |
| `model.py` | JEPA + reward/return/hard/SCOAP/multi-head | JEPA + `q_head` | 第一步新增 `q_head`，旧 head 保留兼容 checkpoint |
| `dataset.py` | rollout + proxy state update + 多 target | rollout + action group / delta_TC label | 保留 rollout；新增/规范 oracle Q dataset |
| `train.py` | 多任务 loss + oracle ranking 辅助 loss | `L_jepa + L_q_value + L_q_rank + L_candidate` | 先并行配置跑；验证后再默认关闭旧 loss |
| `plan.py` | candidate + heuristic/hybrid score | candidate + `q_pred` only | 先新增 `--score-field q_pred`；验证后再改默认 |

## Important Constraint

不要第一步直接删除旧 head。

原因：

- 旧 checkpoint 需要继续可加载，用来对比和回滚。
- oracle gate、planner CSV、已有脚本还依赖旧字段。
- 当前 `B_oracle_0p05` 虽然不是最终形态，但提供了可用 baseline。
- 一次性删除会让问题从“模型是否更好”变成“系统是否还能跑通”，排错成本过高。

工程上应该先做到：

`旧系统可跑 + Q 系统可跑 + 同一 gate 可比较`

然后再删旧结构。

## Target Model Design

### `q_head`

新增唯一决策头：

```text
z_t = encoder(x_state)
action_emb = action_encoder(z_t[action_node], action_type)
z_pred = dynamics(z_t, action_emb, relation_features)
summary = action-conditioned summary(z_pred, action_node, relation_features)
q_pred = q_head(summary)
```

输出语义：

```text
q_pred ~= coverage_scale * delta_TC(s,a)
```

其中 `delta_TC` 是 backend 对插入这个 action 后真实 test coverage 变化的标注。

### 为什么 Q 不是 DeepTPI

这个方案不是纯 DeepTPI，因为仍然保留：

- graph encoder
- action-conditioned dynamics
- rollout/state update 数据结构
- candidate generator 和电路约束

更准确叫法：

`world-model-internalized Q-function planner`

## Loss Design

最终 loss：

```text
L = lambda_jepa * L_jepa
  + lambda_q_value * L_q_value
  + lambda_q_rank * L_q_rank
  + lambda_candidate * L_candidate
```

### 1. `L_jepa`

保留世界模型表示能力：

```text
L_jepa = distance(z_pred, target_encoder(x_next))
```

建议初始权重：

```text
lambda_jepa = 0.1
```

原因：完全去掉 JEPA 会退化成普通 action scorer；但权重不能太大，否则继续让表示学习压过 Q 排序目标。

### 2. `L_q_value`

直接预测真实收益：

```text
target = coverage_scale * oracle_delta_tc
L_q_value = Huber(q_pred, target)
```

建议初始权重 sweep：

```text
lambda_q_value = 0.25 / 0.50 / 1.00
```

### 3. `L_q_rank`

同一个 candidate group 内，两两比较真实收益顺序：

```text
if delta_i > delta_j:
    q_i should be > q_j
loss = softplus(-(q_i - q_j) / temperature)
```

建议初始权重 sweep：

```text
lambda_q_rank = 0.5 / 1.0 / 2.0
```

排序是主目标，所以它至少应该和 value loss 同级，不能只是小辅助项。

### 4. `L_candidate`

让模型在一个候选池里选对最优 action：

```text
p_pred = log_softmax(q_pred over candidate group)
p_target = softmax(oracle_delta_tc / target_temperature)
L_candidate = CE(p_target, p_pred)
```

建议第一轮先关闭：

```text
lambda_candidate = 0.0
```

第二轮再加：

```text
lambda_candidate = 0.1 / 0.3
```

原因：`L_rank` 已经能表达顺序，第一轮先验证一个核心 ranking loss 是否足够；不要一开始又把 loss 搞复杂。

## Phase 1: Add Q Path Without Deleting Legacy Heads

### Code Changes

`model.py`

- 新增 `self.q_head = MLP(summary_dim -> 1)`。
- `predict_from_latent(...)` 输出 `q_pred`。
- 保留旧 head，保证旧 checkpoint 和旧评估脚本可用。

`plan.py`

- `PLAN_FIELDNAMES` 增加 `q_pred`。
- action scoring row 增加 `q_pred`。
- CLI `--score-field` 支持 `q_pred`。
- 不改变默认 planner 分数。

`train.py`

- oracle score field 支持 `q_pred`。
- 新增 `lambda_q_value`。
- 新增 `lambda_q_rank`，可复用现有 pairwise ranking 实现。
- 第一阶段不用改普通 rollout dataset，只在 oracle group step 里训练 Q。

### Verification

```bash
python -m py_compile tpi_jepa/model.py tpi_jepa/train.py tpi_jepa/plan.py
python -m tpi_jepa.smoke_test
```

## Phase 2: Q-Oracle Training Sweep

### Data

训练：

`autoresearch/oracle-balanced-negative-rich-260629-wide/balanced_train_oracle_actions.tsv`

验证：

`autoresearch/oracle-balanced-negative-rich-260629-wide/balanced_val_oracle_actions.tsv`

迁移：

`autoresearch/oracle-action-probe-260629-smoke/oracle_actions.tsv`

### First Sweep

并行跑以下配置：

| config | lambda_jepa | lambda_q_value | lambda_q_rank | lambda_candidate | score |
|---|---:|---:|---:|---:|---|
| `Q_v0_rank0p5` | 0.1 | 0.5 | 0.5 | 0.0 | `q_pred` |
| `Q_v0_rank1p0` | 0.1 | 0.5 | 1.0 | 0.0 | `q_pred` |
| `Q_v0_rank2p0` | 0.1 | 0.5 | 2.0 | 0.0 | `q_pred` |
| `Q_v0_value1_rank1` | 0.1 | 1.0 | 1.0 | 0.0 | `q_pred` |

必须并行执行：

```bash
--parallel-devices cuda:4,cuda:5,cuda:6,cuda:7
```

### Gates

每个 checkpoint 跑：

```text
expanded oracle val
transfer oracle
hard-fault eval
8-circuit planner eval
```

## Promotion Criteria

以 `B_oracle_0p05` 和 incumbent 双 baseline 判断。

当前 baseline：

| baseline | expanded Spearman | expanded neg top1 | transfer Spearman | transfer neg top1 | transfer regret |
|---|---:|---:|---:|---:|---:|
| incumbent | 0.079 | 0.486 | 0.327 | 0.167 | 0.0126 |
| `B_oracle_0p05` | 0.425 | 0.162 | 0.081 | 0.000 | 0.0175 |

Q checkpoint promote 条件：

- expanded Spearman >= `0.425`
- expanded negative top1 <= `0.162`
- transfer negative top1 <= `0.167`
- transfer regret <= `0.0175`
- hard macro F1 tuned >= `0.765`
- 8-circuit real TC 不低于 incumbent

如果只满足 oracle gate，不满足 8-circuit real TC，只能进入 guarded rerank，不能替换 planner 默认分数。

## Phase 3: Candidate-Aware Loss

只有当 Phase 2 的 Q 已经稳定后，再加 `L_candidate`。

第二轮 sweep：

| config | lambda_jepa | lambda_q_value | lambda_q_rank | lambda_candidate |
|---|---:|---:|---:|---:|
| `Q_cand0p1` | 0.1 | best | best | 0.1 |
| `Q_cand0p3` | 0.1 | best | best | 0.3 |

目标：

- 降低 negative top1。
- 降低 top1 regret。
- 不牺牲 transfer Spearman。

## Phase 4: Q Planner Integration

如果 Q gate 通过：

`plan.py`

- 默认 planner score 改成 `q_pred`。
- `hybrid_pred`、`reward_pred`、`return_pred` 只作为 legacy debug 字段保留。

评估：

- greedy + beam 都跑。
- 8 个评估电路必须并行跑。
- 比较真实 backend TC，不只看 oracle action table。

## Phase 5: Delete Legacy Heads

只有在 Q planner 通过 8-circuit gate 后，才删除旧 head。

删除顺序：

1. `reward_head`
2. `return_head`
3. `pattern_head`
4. `hard_reduction_head`
5. `scoap_head`
6. `delta_scoap_head`
7. `hard_count_head`
8. planner 中的 `hybrid_pred` / `bounded_residual_hybrid_pred`

保留：

- `hard_head` 可临时保留为 safety/debug head，直到 Q-only 结果稳定。
- `features.py` 中的 SCOAP 特征继续作为输入，不作为监督目标。

## Why This Plan Is Safer Than Direct Rewrite

直接删除所有旧 head 的风险：

- 无法加载旧 checkpoint。
- 无法复用现有 oracle gate。
- 一旦结果变差，不知道是 Q 设计失败、训练失败、planner 接入失败，还是数据分布失败。

分阶段方案可以回答三个问题：

1. `q_pred` 能不能在固定 oracle group 上排序好？
2. `q_pred` 能不能 transfer？
3. `q_pred` 放进真实 planner 后能不能提升 8 个电路的真实 TC？

只有三个问题都回答“是”，才值得删除旧系统。

## Expected Outputs

第一轮实现应产出：

- `q_pred` field in model/planner/evaluator。
- Q training configs。
- Q sweep report。
- oracle expanded/transfer gate summary。
- 8-circuit planner comparison table。
- promote/reject verdict。

## Next Command

建议下一步执行：

```bash
$autoresearch fix autoresearch/plan-260629-2333/plan.md
```

第一轮只实现 Phase 1 + Phase 2，不删除旧 head。
