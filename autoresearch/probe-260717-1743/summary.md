# TPI-my.3 当前项目地图

生成时间：2026-07-17 17:43 +08:00  
代码：`main@4f96fe9`，工作树有用户改动  
模式：`autoresearch / orchestrator / explore / probe (autonomous)`

## 一句话结论

`TPI-my.3` 是一个以真实 ATPG/BIST test coverage 为最终裁判的测试点插入研究系统。它把 BENCH/AIG 电路转为图，用 JEPA 风格世界模型近似“插入一个测试点后的状态变化”，再用 hard-fault、coverage 与 oracle action-value 信号训练排序 head，从候选池中逐步产生 `CP0/CP1/OP` 计划，最后交给 Atalanta-BIST 或 TMAX 实测。

项目当前阶段已经不再是“模型能不能跑”，而是两个更严格的问题：

1. action ranking 能否在完全 held-out 的大电路上选出真实 ΔTC 高的动作；
2. 这些动作是否落在原始门级网表真正可插点的位置，而不是 AIG 转换产生的临时节点。

## 主数据流

```text
labels.csv + BENCH + hard-fault sidecars
        │
        ├─ labels.py: 过滤动作、恢复动作历史、定位 BENCH
        ├─ bench.py / graph.py: 网表解析、DFF 边界化、张量图
        └─ features.py / scoap.py / testability.py: 节点与 action-cone 特征
        ↓
TPIDataset / TPIRolloutDataset
        ↓
online encoder ── action encoder ── latent dynamics
        │                              │
        └─ EMA target latent           └─ Q/reward/return/hard heads
        ↓
candidate recall → model/context/ensemble rerank → greedy/beam plan
        ↓
Atalanta-BIST / TMAX final TC
        ↓
oracle action groups、固定协议与下一轮训练/规划实验
```

## 核心模块

| 模块 | 责任 | 阅读要点 |
|---|---|---|
| `labels.py` | CSV → `LabelRow`/`TransitionSpec` | ΔTC 字段回退、动作历史、BENCH 查找 |
| `bench.py` | BENCH parser | gate/LUT/alias/constant 支持 |
| `graph.py` | `Circuit` → `GraphData` | DFF 默认作为组合/扫描边界 |
| `features.py` | 基础状态与 action relation | 已插点状态主要是 proxy，不重写网表 |
| `dataset.py` | 单步与多步 rollout 样本 | hard-fault sidecar、按电路 split、单图 batch |
| `model.py` | JEPA 世界模型 | online/EMA target、latent dynamics、多任务 heads |
| `train.py` | 多目标训练 | world-model loss 与 oracle ranking step 交织 |
| `plan.py` | 候选、打分、搜索 | 2922 行，是当前研究迭代中心 |
| `evaluate_plan_tmax.py` | 真实 backend 评估 | baseline/final/all-step、生成训练标签 |
| `protocol.py` | held-out 隔离 | 固定协议及 alias 排除 |

## 模型真正学习什么

基础 world-model loss 同时包含：下一 latent、SCOAP/delta-SCOAP、node hard-fault、hard count/reduction、单步 coverage reward 与 discounted return。planner-aligned 配置另从固定 oracle action groups 学习 Q/value、pairwise hard-negative ranking、candidate distribution、NDCG/conservative/context 等目标。

这意味着模型不是纯 JEPA，也不是纯强化学习：它是以 JEPA latent dynamics 为骨架、以监督 action-value ranking 为规划对齐手段的混合模型。`q_pred` 由 oracle group 监督，但规划仍在模型预测的 latent 状态上滚动，不会每步调用 backend。

## 当前规划主线

历史方案用 `heuristic_recall_pool` 召回、三个 safe Q checkpoint ensemble、`mean(q)-0.75*std(q)` 的 LCB、greedy 逐步选择。当前工作树又加入：

- 全候选策略统一的 `candidate_allowlist`；
- pool-relative `q_pred_context` 等多 head 支持分；
- context 分数相同时回退到 raw head；
- hard-fault cluster 更大的 hard seed 池。

当前 exact-legal 优化发现不同电路偏好的排序并不统一：b15 在 hard-fault cluster 候选上 `reward_pred` 很强；b17 更偏好 `q_pred_context + hard_fault_cluster`。这说明一个全局固定 planner recipe 仍未证明在所有电路上最优。

## 正确理解“当前最好结果”

旧记录声称 8/8 超过 DeepTPI，但它混有两个问题：对比列不正确，以及恢复 AIG 临时节点未受原始门合法性约束。当前精确白名单重跑只覆盖五个 ITC'99 电路：

| 电路 | 起始 exact-legal TC | 当前发现的更好 TC | DeepTPI Table IV | 状态 |
|---|---:|---:|---:|---|
| b15_C | 89.987 | 94.614（reward/cluster） | 93.20 | 已找到可胜方案 |
| b20_C | 96.538 | 96.538 | 95.02 | 通过 |
| b21_C | 95.752 | 95.752 | 94.51 | 通过 |
| b22_C | 96.615 | 96.615 | 95.59 | 通过 |
| b17_C | 90.390 | 91.545（当前迭代最好记录） | 91.67 | 尚差 0.125pp |

这里的“当前发现”来自 `autoresearch/improve-260716-1344/results.tsv` 的局部实验记录，尚未折叠回基准 `summary.json`；因此机械成功谓词目前仍输出 `FAIL`。b17 是剩余主要瓶颈，三个 EPFL 电路的精确原始网表合法性则是更外层的未闭环问题。

## 已验证事实

- `python -m py_compile tpi_jepa/*.py scripts/*.py`：通过。
- `PYTHONPATH=. pytest -q tests`：29 passed。
- 主 labels：100131 总行、100000 动作行、131 电路、BENCH 缺失 0。
- 动作分布：CP0 29784、CP1 30079、OP 40137；五个 step 各 20000。
- 固定评估协议：可解析，8 电路严格预算齐全。
- exact-legal 当前分数：`-4.399143600`；成功检查：`FAIL`。
- 未启动 GPU 训练、模型推理或 backend 复算。

## 最短阅读顺序

1. 本文件与 `docs/exploration_knowledge_base.md`，先理解目标和历史结论；
2. `labels.py → dataset.py → model.py → train.py`，理解训练闭环；
3. `plan.py:1920-2365`，理解打分/context/greedy；
4. `configs/eval_protocol_coverage_only.json` 与 `evaluate_plan_tmax.py`，理解最终裁判；
5. `docs/itc99_exact_recovery.md` 与 `autoresearch/improve-260716-1344/results.tsv`，理解当前研究前沿。

## 最重要的风险

1. 代理 loss 改善不等于最终 TC 改善；action ranking 和 backend top1/final TC 必须独立验证。
2. latent rollout 依赖近似 state update，长序列可能积累模型误差。
3. 大候选池提高 recall 也可能改变 pool-relative context 标度，候选策略与 score 不可割裂评估。
4. b15/b17 上反复调参会把 held-out 电路变成开发集；必须保留真正未触碰的外层验证。
5. 旧 `/data3` 配置路径在当前机器不存在；运行前要改为 `/data4` 或使用明确的新配置。
6. `best.pt` 按综合 val loss 选，不直接按 held-out ranking/TC 选，存在选模目标错位。
