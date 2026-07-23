# TPI-my.3 项目理解

生成时间：2026-07-12

模式：`autoresearch learn / summarize / comprehensive`

代码基线：`main` @ `4f96fe9`（工作树在侦察开始时干净）

## 一句话结论

`TPI-my.3` 是一个面向数字电路测试点插入（TPI）的研究系统：它把 BENCH/AIG 电路表示成图，用 JEPA 风格世界模型预测插入 `CP0`、`CP1` 或 `OP` 后的潜在状态、hard-fault 变化和真实测试覆盖率收益，再通过候选生成、动作排序与 greedy/beam planner 产生测试点序列，最后调用 Atalanta-BIST 或 TMAX backend 测量真实 TC。

项目已经从“证明世界模型能运行”的早期阶段，演进到“在严格 held-out 电路上学习可靠 action-value 排序”的阶段。当前最重要的工程资产不是单个神经网络，而是以下闭环：

```text
subcircuit labels + BENCH
        ↓
graph / testability features / hard-fault supervision
        ↓
JEPA world model + Q/reward/hard-reduction heads
        ↓
candidate recall pool + action reranking + planner
        ↓
Atalanta/TMAX real backend evaluation
        ↓
oracle action-value labels and fixed held-out protocol
```

## 1. 项目要解决什么问题

给定一个组合/扫描边界化电路和已经插入的测试点序列，系统需要选择下一个动作：

- `CP0`：控制点，使目标节点更容易被置 0；
- `CP1`：控制点，使目标节点更容易被置 1；
- `OP`：观测点，提高故障传播到可观测输出的机会。

最终目标不是单纯提高某个 node-level proxy，而是让 planner 选择的动作序列最大化真实 backend 测得的 test coverage。仓库中的研究记录反复强调：hard-fault F1、SCOAP、reward head 的回归误差都只是中间信号，真正需要关注的是同一状态内候选动作的真实 `delta_tc` 排序和最终电路 TC。

## 2. 数据与监督

### 2.1 主训练分布

主标签配置指向：

```text
/data4/pengqingsong/DFT/Dataset/atalanta_bist_lowtc_subckt_100k_labels/labels.csv
```

README 记录的数据规模为：131 个 low-TC AIG 子电路、20,000 条长度 5 的序列、100,000 个 step labels、300,000 patterns、seed 2026。训练对象是 sampled subcircuits，不是原始完整大电路。

`labels.py` 完成 CSV 过滤、动作类型规范化、BENCH 路径解析，以及从 sequence 字段恢复动作前后的插入历史。`dataset.py` 再把每行或每段连续序列变成 `TransitionSample` / `RolloutSample`。

### 2.2 监督信号

单步或 rollout 训练可组合使用：

- EMA target encoder 给出的下一状态 latent（JEPA target）；
- 下一状态 SCOAP proxy 与 delta-SCOAP；
- node-level stuck-at-0 / stuck-at-1 hard-fault 标签；
- graph-level hard-fault count；
- hard-fault reduction 三维目标；
- 单步 `delta_test_coverage`（旧数据可回退到 `delta_fault_coverage`）；
- discounted return；
- oracle action groups 中的真实 action value、pairwise/listwise ranking 和 candidate distribution。

### 2.3 数据隔离

`split_by_benchmark` 按 `benchmark_id` 划分，避免同一子电路同时进入 train/val。planner-aligned safe 配置还通过 `exclude_eval_protocol` 排除最终 8 个评估电路及 alias，并用 `oracle_forbidden_benchmarks` 拒绝含目标电路的 oracle TSV。

这是项目最重要的实验边界：最终 8 电路应只用于 held-out 最终评估，不能用于训练、校准或选模。

## 3. 从电路到张量

### 3.1 BENCH 解析与图构建

- `bench.py` 解析 `INPUT`、`OUTPUT`、普通 gate、常量、alias 和 LUT；部分 LUT 会解码或展开为 SOP。
- `graph.py` 将节点名映射为整数，构建有向边、fanin/fanout、gate type、PI/PO mask；DFF 默认作为组合边界处理。
- `scoap.py` 计算 controllability/observability proxy。
- `testability.py` 计算 reconvergence pressure、FFR span、transparent-chain、cone pressure 和 fault-path edge weights。

### 3.2 特征

`features.py` 将基础 gate/结构/SCOAP 特征与可选 real-fault prior、activation prior 合并。动作状态不是通过修改 BENCH 重建，而是通过 feature proxy 表示已插入动作；action relation feature 描述候选节点的 fanin/fanout/cone 邻域。

当前主线配置使用：

```text
feature_mode = testability
relation_mode = cone
relation_depth = 8
edge_weight_mode = fault_path
edge_keep_ratio = 0.6
```

## 4. 世界模型

核心类是 `model.py::TPIWorldModel`：

1. `online_encoder` 对当前图节点编码；
2. `target_encoder` 由 online encoder 的 EMA 更新，只提供下一状态 latent target；
3. `ActionEncoder` 结合动作节点 latent 和动作类型 embedding；
4. `DynamicsPredictor` 依据动作与 relation feature 预测 `z_{t+1}`；
5. graph/action summary 汇聚全局、top-norm、cone 和可选 action context；
6. 多个 head 从 summary 或 node latent 输出规划与辅助信号。

主要输出包括：

- `z_pred`：预测的下一状态 latent；
- `q_pred`：action Q/value 排序分数；
- `reward_pred` / `fc_pred`：单步 coverage reward；
- `return_pred`：多步 return；
- `hard_reduction_pred`：hard-fault reduction；
- `hard_logits` / `hard_count_pred`：node hard-fault 与数量；
- `scoap_pred` / `delta_scoap_pred`；
- planner 中进一步组合出的 guarded、hybrid、context 和 ensemble-LCB 分数。

默认 latent dimension 为 64、3 层 encoder、action type embedding 为 16。训练 batch size 实际为 1，因为每个电路图大小不同，`collate_one` 直接返回单图样本。

## 5. 训练逻辑

入口：

```bash
python -m tpi_jepa.train --config <config.json>
```

`train.py` 的主流程是：加载 config → 过滤 held-out benchmark → 按 benchmark split → 创建 transition/rollout dataset → 建模 → 常规 world-model step 与 oracle ranking step 交织训练 → EMA 更新 → val → 保存 checkpoint/history。

训练损失是多个目标的加权和。基础主线 `mainline_world_model_simplified.json` 偏重 JEPA、hard reduction 和 coverage reward；planner-aligned Q 配置则额外从固定 oracle action groups 学习：

- Q value regression；
- best-vs-hard-negative pairwise ranking；
- candidate distribution/listwise 信号；
- 可选 NDCG、conservative、context ranking。

典型 safe Q 配置使用 5-step rollout、hard-weighted sampling，并在 oracle 数据中显式禁止最终 8 个目标电路。

输出通常位于 `runs/<run_name>/`：`latest.pt`、`best.pt`、`best_final_horizon.pt`、`history.csv` 和可选 epoch checkpoints。

## 6. 规划逻辑

入口：

```bash
python -m tpi_jepa.plan \
  --checkpoint <checkpoint> \
  --benchmark-id <id> \
  --budget <N> \
  --planner greedy \
  --score-field q_pred
```

`plan.py` 是当前最大的核心文件（约 2,868 行），包含三类职责：

- 候选生成：netlist、testability、hard-fault、cone、cluster、reconvergence、FFR、mixed、recall-pool 及 cached 变体；
- 候选打分：Q、reward、return、hard reduction、guarded/hybrid/context、ensemble mean/LCB；
- 搜索：greedy、有限深度 beam rollout、full-sequence beam。

当前最佳记录使用的不是大 beam，而是每一步：

1. `heuristic_recall_pool` 召回最多 96 个候选；
2. 三个 Q checkpoint 分别打分；
3. 用 `mean(q) - 0.75 * std(q)` 形式的 LCB 偏好跨 seed/checkpoint 一致的动作；
4. greedy 选择 `q_pred_lcb` 最高的动作；
5. 重复到严格预算。

这说明当前收益主要来自“高召回候选池 + planner-aligned Q rerank + ensemble uncertainty penalty”，而不是更深的 rollout 搜索。

## 7. Backend 评估闭环

`evaluate_plan_tmax.py` 读取 plan CSV，把动作转换为 backend test points，然后：

1. 跑无测试点 baseline；
2. 按 `all` 或 `final` 模式评估插入后的序列；
3. 调用 `testability.py` 中的 backend 封装运行 TMAX 或 Atalanta-BIST；
4. 保存 `labels.csv`、`summary.json`、每步 `label.json`，可选导出新的 step training labels。

最终论文比较固定在 `configs/eval_protocol_coverage_only.json`：8 个 restored DeepTPI Table-II 电路、严格 Table-II `#TP` 预算、300k patterns、seed 2026、Atalanta-BIST、final-only。预算不可用当前 parser 的 gate count 重新计算。

## 8. 当前最佳结果与研究状态

仓库内 `autoresearch/improve-260706-0959/current_best.json` 记录：

- 方法：`q_lcb_ensemble_safe`；
- checkpoint：`q_rank_v1`、`q_rank_v2_safe`、`q_rank_v2_seed2_safe`；
- planner：greedy；
- candidate：`heuristic_recall_pool`，96 candidates；
- score：`q_pred_lcb`，LCB alpha 0.75；
- macro final TC：90.357%；
- minimum final TC：74.852%；
- 相对 DeepTPI：8/8 电路更高；
- 最小优势：b15_C 上 +0.046 percentage point。

这些数字与对应 checkpoint、summary TSV 和 final-TC comparison TSV 均存在并相互吻合。它们属于历史实验产物的可追溯记录；本次 learn 没有重新运行耗时的 GPU planning/backend evaluation，因此没有独立复算该结论。

## 9. 当前最值得记住的研究判断

1. **最终问题是 action ranking，不是 node classification。** hard-F1 高不保证真实 ΔTC top1 安全。
2. **hard-fault reduction 是强而稳定的 prior。** 早期 reward-only 头容易选到负收益动作。
3. **只做 checkpoint 后处理/小范围 finetune 接近瓶颈。** action encoder/dynamics 解冻能改变排序，但可能伤害 full-circuit transfer。
4. **oracle group 是关键数据资产。** 同状态多候选的真实 backend ΔTC 才能直接监督排序。
5. **held-out 隔离是硬约束。** 目标 8 电路的 exact-rank 数据不能用于训练、校准或选模。
6. **多 seed/ensemble 很重要。** 项目历史显示单 seed 方差明显；LCB ensemble 正是在惩罚不一致动作。
7. **最终最佳的 margin 很薄。** b15_C 仅 +0.046pp，高价值结论应补多 seed/pattern/backend 重复性验证。

## 10. 目录阅读地图

| 路径 | 作用 | 建议阅读优先级 |
|---|---|---:|
| `tpi_jepa/labels.py` | 标签与 BENCH 路径、sequence → transition | 1 |
| `tpi_jepa/dataset.py` | transition/rollout 样本和 hard-fault sidecar | 1 |
| `tpi_jepa/model.py` | encoder、dynamics、所有 heads | 1 |
| `tpi_jepa/train.py` | 多目标与 oracle ranking 训练 | 1 |
| `tpi_jepa/plan.py` | 候选、打分、搜索 | 1 |
| `configs/eval_protocol_coverage_only.json` | 最终评估合同 | 1 |
| `autoresearch/improve-260706-0959/current_best.md` | 当前最佳记录 | 1 |
| `bench.py` / `graph.py` / `features.py` | 电路到图特征 | 2 |
| `evaluate_plan_tmax.py` / `testability.py` | 真实 backend 闭环 | 2 |
| `scripts/` | 数据构建、oracle、消融、报告、并行 runner | 2 |
| `docs/exploration_knowledge_base.md` | 截至 6 月底的研究判断 | 2 |
| `autoresearch/` | 原始实验资产，约 8.9 GB | 按任务定向查找 |

## 11. 推荐的快速验证与复现顺序

不启动 GPU 训练时：

```bash
cd /data4/pengqingsong/DFT/TPI-my.3
python -m py_compile tpi_jepa/*.py scripts/*.py
PYTHONPATH=. pytest -q tests
python scripts/validate_eval_protocol.py --help
python -m tpi_jepa.plan --help
```

需要实际模型推理、规划、评估或训练时，应显式使用 CUDA/GPU。最终 8 电路复现是长任务，并且脚本要求显式确认 held-out evaluation；建议按 `current_best.md` 中命令运行，而不是在交互式检查中临时拼接参数。

## 12. 工程风险与文档缺口

- README 的部分 Quick Start 仍写 `/data3/.../TPI-my.3`，当前仓库实际在 `/data4/...`；后文有些路径已更新，存在混用。
- 裸 `pytest -q` 会扫描巨大的 `autoresearch/` 树，收集阶段超过 100 秒仍未开始测试；应限制到 `tests/`。
- 当前环境直接执行 `pytest -q tests` 没有把仓库根目录加入 import path；`PYTHONPATH=. pytest -q tests` 才稳定通过。
- `requirements.txt` 只有 `torch/numpy/pandas`，未包含 pytest，也没有版本锁定或环境文件，复现性较弱。
- 约 454k 个 tracked 文件位于 `autoresearch/`，仓库过重，影响 pytest discovery、git 操作与知识定位。
- `docs/codebase_guide.md` 对基础流准确，但没有覆盖 7 月形成的 Q-LCB ensemble/current-best 主线。
- `docs/exploration_knowledge_base.md` 更新时间为 2026-06-29，晚于此的 strict held-out safe ranking 和最终最佳结论尚未合并进去。
- 最佳结果 TSV 中保留了同一 benchmark 的早期 `plan_error` 行和后续成功重试行；比较脚本显然采用成功记录，但直接手工聚合 TSV 时需按 `status=ok` 去重/过滤。

## 13. 本次验证

- Python 静态编译：通过；
- `PYTHONPATH=. pytest -q tests`：18 passed；
- 主训练 labels 路径：存在；
- fixed eval protocol：可解析，8 电路和严格预算齐全；
- current-best 三个 checkpoint：均存在；
- current-best summary/comparison：均存在且数值吻合；
- GPU 训练/规划/backend 复算：未运行（learn/summarize 模式，避免启动长任务）。
