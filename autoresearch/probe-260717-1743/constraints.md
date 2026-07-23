# TPI-my.3 约束注册表

范围：`/data4/pengqingsong/DFT/TPI-my.3`。模式：autonomous。状态：6 轮后饱和。

## 目标与领域

- C01（高）：最终目标是提高真实 backend 测得的 test coverage，不是单独提高 JEPA loss、SCOAP、hard-fault F1 或 reward 回归精度。证据：`docs/exploration_knowledge_base.md`、`tpi_jepa/evaluate_plan_tmax.py:188`。
- C02（高）：动作空间是节点上的 `control0`、`control1`、`observe`，分别对应原始标签 `CP0`、`CP1`、`OP`。证据：`tpi_jepa/features.py:14`、`tpi_jepa/labels.py:33`。
- C03（高）：规划输出必须是一个不重复的、有预算上限的动作序列；论文级比较还必须保证节点属于原始门级合法候选集。证据：`tpi_jepa/plan.py:244`、`tpi_jepa/plan.py:124`。

## 数据与泄漏边界

- C04（高）：主训练分布是 131 个 sampled low-TC AIG 子电路、100000 个动作标签和 20000 条五步序列，而不是 8 个完整评估电路。本次实测全部 131 个 BENCH 可解析。
- C05（高）：标签优先使用 `delta_test_coverage`，缺失时兼容 `delta_fault_coverage`。证据：`tpi_jepa/labels.py:102`。
- C06（高）：train/val 按 `benchmark_id` 切分，不能按行随机切分。证据：`tpi_jepa/dataset.py:563`。
- C07（高）：最终 8 电路及 alias 必须从训练标签和 oracle action groups 中排除。证据：`tpi_jepa/protocol.py`、`tpi_jepa/train.py:1325-1430`。
- C08（高）：固定 oracle TSV 应复用 backend 标签，checkpoint 比较不能重新采样候选或重复 backend 后再直接比较。

## 表征与模型

- C09（高）：BENCH 经 parser、DFF 组合边界化、图构建、结构/SCOAP/testability 特征后进入模型。证据：`tpi_jepa/bench.py:285`、`tpi_jepa/graph.py:68`、`tpi_jepa/features.py:150`。
- C10（高）：已插入测试点通常由 feature proxy 表示，并不真正重写网表；因此 latent rollout 是近似动力学，不是逻辑等价仿真。证据：`tpi_jepa/features.py:182-257`。
- C11（高）：模型使用 online encoder 与 EMA target encoder；action encoder 和 dynamics 预测下一 latent，再由多个 head 输出 Q、reward、return、hard reduction 等。证据：`tpi_jepa/model.py:189-433`。
- C12（高）：图大小可变，当前训练单位是单图样本，batch size 实质为 1。证据：`tpi_jepa/dataset.py:600`。
- C13（中）：主线 planner-aligned 模型把普通 world-model loss 与固定 oracle group 上的 value/pairwise/candidate/context ranking loss 交织训练。证据：`tpi_jepa/train.py:987-1202`。

## 规划与评估

- C14（高）：candidate recall 和 model rerank 是两层问题；候选池没有召回好动作时，Q head 无法补救。证据：`tpi_jepa/plan.py:2220-2349`。
- C15（高）：当前实用 planner 支持 greedy、receding-horizon beam 和 full-sequence beam；历史最好方案主要依赖 greedy + 多 checkpoint Q-LCB，而非深 beam。
- C16（高）：最终固定协议是 300000 patterns、seed 2026、Atalanta-BIST、final-only、严格 DeepTPI Table-II 测试点预算。证据：`configs/eval_protocol_coverage_only.json`。
- C17（高）：论文比较基线必须用 DeepTPI Table IV 的最终 TC；早期 `current_best.json` 错把另一列当最终 TC，不能作为严格胜负结论。
- C18（高）：ITC'99 五电路已有逐门结构证明白名单；EPFL 的 i2c/max/mem_ctrl 尚无同等级精确原始版本映射，不能声称完整 8 电路物理合法性。

## 工程与运行

- C19（高）：训练、模型推理、规划和 checkpoint 评估默认必须用 GPU；本次只做静态检查与 CPU 单元测试，没有启动深度学习 job。
- C20（高）：项目含约 16 GB 大量实验产物，测试应显式运行 `PYTHONPATH=. pytest -q tests`，不能裸扫整个仓库。
- C21（高）：多个旧配置仍写不存在的 `/data3/...` 标签路径；当前可用主标签在 `/data4/...`，直接运行旧 smoke/full 配置会在加载标签前失败。
- C22（中）：checkpoint 目前按综合 `val_loss` 保存 `best.pt`/`best_final_horizon.pt`，并非直接按 held-out action ranking 或最终 TC 选模。证据：`tpi_jepa/train.py:1529-1583`。
- C23（高）：工作树已有用户改动与未跟踪恢复/评估资产；理解任务不得覆盖、回退或顺手整理它们。
- C24（高）：当前 exact-legal 优化以 b15、b17 为工作电路，b20/b21/b22 为回归电路；成功谓词必须五个都超过正确 DeepTPI TC。
