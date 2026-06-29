# TPI-JEPA 探索知识库

更新时间：`2026-06-29`

本文档集中保存当前项目探索经验。原则是：每个技术点单独成段，记录它解决什么问题、怎么做、结果是什么、后续如何使用。详细原始实验产物仍保存在 `autoresearch/` 目录下。

## 目录

- [一、项目目标与核心判断](#一项目目标与核心判断)
- [二、数据分布与实验边界](#二数据分布与实验边界)
- [三、评估指标与 Gate 设计](#三评估指标与-gate-设计)
- [四、Hard-Fault 表征探索](#四hard-fault-表征探索)
- [五、Oracle Action-Value 工具链](#五oracle-action-value-工具链)
- [六、Action-Value Finetune 实验](#六action-value-finetune-实验)
- [七、Planner Score 组合实验](#七planner-score-组合实验)
- [八、Transfer 评估经验](#八transfer-评估经验)
- [九、工程工具链与 Bug 经验](#九工程工具链与-bug-经验)
- [十、当前结论与下一步路线](#十当前结论与下一步路线)

## 一、项目目标与核心判断

**最终目标：让模型排序真实 TC 收益高的动作。** 项目的最终目标不是单纯提高 hard F1，也不是让模型 recall 历史动作，而是在同一电路状态下，让模型给候选 test-point action 的分数排序尽量接近 backend 标注的真实 `delta_tc` 排序。也就是说，模型要把真实收益高的动作排在前面，并尽量避免把负收益 action 选成 top1。

**Node-level proxy 不等于 planner-level action value。** 早期 hard-fault node prediction 和 hard-reduction proxy 提供了有用表征，但最终动作选择是 action-level ranking 问题。一个模型可能 hard node F1 高，却不能把真实 `delta_tc` 高的 action 排前面。因此后续评估必须用固定 action groups 上的 Spearman、negative top1、top1 regret，而不能只看 node-level 指标。

**当前 checkpoint 上的小范围 finetune 已接近瓶颈。** 多轮实验表明，只改 reward/return head 安全但排序提升弱；解冻 action_encoder/dynamics 能改变排序但破坏 transfer；只训练 bounded residual 安全但收益很小。这说明事后 finetune 不是根本解决方案，下一阶段更可能需要从头重新训练，并把 oracle ranking loss 混入原始训练目标。

**安全性比单纯 Spearman 更重要。** 有些实验能提升某个 score field 的 Spearman，但会提高 `negative_top1_rate` 或 `top1_regret`。这种 checkpoint 不能 promote，因为 planner 实际只选 top action，top1 选错比平均排序略好更危险。

## 二、数据分布与实验边界

**训练分布是 131 个 labeled sampled subckt，不是原始大电路。** 关键纠偏来自用户指出：模型训练集使用的是 sampled subgraph/subckt，而不是原始 full circuit。正确源文件是 `/data4/pengqingsong/DFT/Dataset/atalanta_bist_lowtc_subckt_100k_labels/labels.csv`，对应 bench root 是 `/data4/pengqingsong/DFT/Dataset/atalanta_bist_lowtc_subckt_100k_labels/subcircuits`。因此 oracle action-value 训练数据必须来自这 131 个 labeled subckt 分布，否则训练/评估分布不一致。

**初始 labeled-subckt oracle split。** 第一版正确分布实验使用 24 个 train subckt 和 8 个 val subckt。生成 train oracle `1296 actions / 72 groups`，val oracle `432 actions / 24 groups`。这个规模足以打通工具链，但后续证明 val 太小、结果噪声较大。

**Expanded labeled-subckt oracle split。** 后续扩展到 96 个 train subckt 和 16 个 val subckt。生成 train oracle `5184 actions / 288 groups`，val oracle `864 actions / 48 groups`。其中 train 复用旧 1296 actions，新评估 3888 actions；val 复用旧 432 actions，新评估 432 actions。这个 expanded split 当前更适合作为 in-distribution held-out gate。

**Full-circuit transfer 不是训练分布，只是风险检查。** transfer gate 使用 full-circuit smoke oracle actions，例如 `b15_C` 和 `i2c_aig` 的固定 action groups。它不用于训练，也不代表 sampled-subckt 训练分布；它的意义是检查 checkpoint 是否破坏大电路上已有的排序/安全性。

**Candidate cache 必须与 subckt 集合一致。** 使用 `cached_stride,cached_hard_cone,cached_random` 时，必须为对应 subckt 生成 candidate cache。原始 cache 只覆盖 32 个 selected subckt，expanded split 需要新 cache：`autoresearch/tp-candidates-labeled-subckt-expanded-260629/`。

## 三、评估指标与 Gate 设计

**Spearman 衡量组内排序一致性。** 对每个 `(benchmark_id, state_id, candidate_strategy)` group，取所有 action 的模型 score 和真实 `oracle_delta_tc`，计算 Spearman rank correlation。它回答的是“模型排序和真实收益排序是否一致”。这是核心 ranking 指标，但不能单独决定 promote。

**negative_top1_rate 衡量 top1 安全性。** 对每个 group，按模型 score 选 top1 action，如果该 action 的真实 `oracle_delta_tc < 0`，则记为 negative top1。`negative_top1_rate` 是所有 groups 中 negative top1 的比例。它直接反映 planner 是否会把有害动作排第一。

**top1_regret 衡量 top1 错过最优收益多少。** 对每个 group，计算 `oracle_best_delta_tc - model_top1_real_delta_tc`。即使 top1 不是负收益，也可能比最优 action 差很多；top1 regret 用来衡量这种机会损失。

**sign_accuracy 不是主指标。** 一些小 subckt 上 sign accuracy 很高，但排序仍然弱。原因是很多 action 都是正收益，预测正负号容易，但要把高收益 action 排前面仍然困难。因此 sign accuracy 只能作为辅助指标。

**固定 oracle gate 消除了采样/backend 随机性。** `scripts/evaluate_oracle_action_values.py` 会读取固定 `oracle_actions.tsv`，对同一批 action 用不同 checkpoint 重打分，不重新采样、不重新跑 backend。这使 checkpoint 比较只反映模型打分差异，而不是候选采样或 backend 噪声。

**Gate verdict 的局限。** 当前 gate 的总体 verdict 以 best-by-Spearman 逻辑为主，有时会被非实际 planner score field 影响。因此关键实验必须直接读目标 score field 的 summary 行，例如 `hybrid_pred` 或 `bounded_residual_hybrid_pred`，不能只看 `PROMOTE/REJECT/INCONCLUSIVE`。

**Promote 标准应同时看 in-distribution 和 transfer。** 当前推荐 gate：expanded labeled-subckt val 上 Spearman 应提升且 negative_top1/top1_regret 不恶化；full-circuit transfer 上至少不能出现明显 Spearman 崩塌或 top1 安全退化。单边通过不够。

## 四、Hard-Fault 表征探索

**Hard-fault-aware JEPA 方向是有效的。** 早期多 seed 实验表明，用 hard fault node prediction 作为辅助/self-supervised 信号可以让 JEPA world model 学到 fault-testability 相关表示。`lambda_hard_count=0.10` 与 `0.12` 的平均 hard F1 接近，说明大方向成立，后续瓶颈不在这个小权重差异。

**Seed/split 方差是早期主要风险。** 5-seed A/B 显示 hard F1 方差较大，同一路线可能从约 `0.42` 到 `0.80`。因此单 seed 结果不能作为强结论，任何结构改动都应尽量多 seed 或用固定 oracle gate 做验证。

**ASL 和 hard negative mining 是合理默认。** hard fault 标签不平衡，ASL/asymmetric loss 和 top-k hard negative mining 更适合此类多标签任务。后续除非专门做 loss ablation，否则不建议频繁改这部分。

**Gate/cone/rank 等 GNN 技术需要独立 ablation。** 早期 `gate_dir + cone + rank` 打包实验结果不好，但不能判断是哪一项拖累。经验是：技术组件不能捆绑解释，必须做 factorial ablation。否则失败结果没有可操作性。

**Hard-reduction signal 在 planner 中很重要。** 多轮 oracle action-value 评估显示，`hard_reduction_total_pred` 或以它为主的 hybrid score 往往比 `reward_pred` 更安全，特别是在 full-circuit transfer 上。这说明 hard-reduction 表征虽然不是最终目标，但它提供了重要的稳定排序 prior。

## 五、Oracle Action-Value 工具链

**Oracle action-value dataset 是关键资产。** 工具链现在可以为一组 candidate actions 调 backend，标注真实 `oracle_delta_tc`，并导出 `oracle_actions.tsv`、`oracle_groups.tsv`、`prediction_metrics.tsv`、`rank_metrics.tsv`、`state_summary.tsv`、`manifest.json` 和 `handoff.json`。这使后续任何 checkpoint 都能在同一批 oracle actions 上比较。

**Oracle label 定义。** 主要标签是 `oracle_delta_tc = backend TC after S + [a] - backend TC after S`。同一 group 内 action 共享相同状态 `S`，因此可以做 pairwise/listwise ranking。当前脚本主要支持 `state_id=initial`，后续如果 planner 用 rollout states，应扩展到非 initial state。

**Oracle probe 支持 resume。** `scripts/oracle_action_value_probe.py --resume` 可以复用已有 `oracle_actions.tsv` 中的 backend 结果，只评估新增 action。expanded oracle 正是通过预填旧 TSV 再 resume 的方式完成，避免重复跑 1296/432 条旧 action。

**初始 full-circuit oracle probe 结果。** 在 `b15_C` 和 `i2c_aig` 的 288 actions / 6 groups 上，`hybrid_pred` mean Spearman 约 `0.3274`，negative top1 `0.1667`；`reward_pred` Spearman 约 `0.2947`，negative top1 `0.5`。结论是模型有弱排序能力，但 reward/top1 安全不足。

**Small circuit probe 否定了“只是大电路泛化差”的简单解释。** 小 subckt probe 上 best Spearman 仍然很低，`hard_reduction_total_pred` 约 `0.0748`。这说明问题不只是训练子电路太小，而是 action-value ranking 本身没有被直接学好。

**Expanded oracle 让 val 更可信。** 旧 8-subckt val 太小，部分方法在小 val 上看似有收益；expanded 16-subckt val 显示 incumbent/hard-only 已经更强，head-only residual 很难稳定超过它。因此后续优先相信 expanded val。

## 六、Action-Value Finetune 实验

**Reward/return head-only value+rank finetune 不够。** 使用 72 train groups 对 `reward_head + return_head` 做 value loss + pairwise rank loss，rank loss 只轻微下降。held-out labeled-subckt 上 hybrid Spearman 基本不提升，reward_pred 反而更不安全；full-circuit transfer 也不能 promote。结论：只改 reward/return head 无法让模型学到可靠 action-value 排序。

**Pairwise-only loss 概念正确但 head-only 太弱。** 只用 pairwise ranking loss 训练 reward/return，train rank loss 从 `0.5873` 仅降到 `0.5860`，held-out 和 transfer 都没有改善。结论：pairwise ranking 是正确目标族，但作用在现有 reward/return head 上表达能力不足。

**Joint-hybrid 训练能动排序但损害 transfer。** 解冻 `action_encoder + dynamics + reward_head + return_head + hard_reduction_head`，直接对 `hybrid_pred` 做 pairwise ranking，train rank loss 从 `0.7966` 降到 `0.6386`。held-out hybrid Spearman 从 `0.0057` 到 `0.0255`，top1 regret 改善；但 full-circuit transfer hybrid Spearman 从 `0.3274` 掉到 `0.2567`。结论：joint training 有学习能力，但会破坏大电路稳定性。

**Freeze hard head 不能解决 transfer 退化。** `planner_joint_frozen_hard` 冻结 `hard_reduction_head`，只训练 action_encoder/dynamics/reward/return。结果 transfer 更差：hybrid Spearman 约 `0.1821`，hard_reduction_total_pred 也从 `0.2671` 进一步掉到 `0.1900`。结论：问题不在 hard head 权重，而在 action_encoder/dynamics 改变了 hard head 输入 latent。

**Head-only hybrid ranking 安全但不提升 held-out。** 冻结 action_encoder/dynamics/hard_reduction，只训练 reward/return，并用 `hybrid_pred` 做 ranking。full-circuit transfer 保住甚至略高，但 held-out subckt Spearman 不提升，negative top1 变差。结论：这种方法安全，但 reward/return 残差能力太弱。

**Expanded heads-hybrid + early stopping 证明过拟合明显。** 使用 expanded train/val 后，train rank loss 持续下降，但 val Spearman 从 epoch 1 后快速变负，negative top1 变差。`best.pt` 是 epoch 1，但 expanded val 仍不超过 incumbent。结论：更大数据也不能让 head-only residual 稳定改善排序。

**Bounded residual 是最安全的结构改动。** 新增 `bounded_residual_hybrid_pred = hard_reduction_total_pred * coverage_scale + alpha * (reward_pred + return_pred)`，冻结所有模型权重，只训练一个 `alpha`。best alpha 约 `-0.1121`。expanded val Spearman 从 hard-only `0.044034` 到 `0.045258`，negative top1/top1 regret 不变；transfer top1 安全也不变。结论：方向有效但收益很小。

## 七、Planner Score 组合实验

**原始 hybrid score。** `hybrid_pred = return_pred + reward_pred + hard_reduction_total_pred * coverage_scale`。它在早期 full-circuit transfer 上比 reward_pred 更安全，但其中 hard_reduction 项通常占主导。优点是 transfer 稳定，缺点是 reward/return 的修正作用不可控。

**Reward_pred 单独作为 planner score 风险大。** 多轮 gate 显示 reward_pred 有时 Spearman 看起来更高，但 negative top1 明显更差。典型现象是 reward_pred 会把真实负收益 action 排到前面。因此 reward_pred 不能直接作为 planner score。

**Hard-reduction-only 是强 baseline。** `hard_reduction_total_pred` 在 expanded val 和 transfer 上经常接近或超过 hybrid_pred，是当前最稳定的 score prior。任何新 score 都应先与 hard-only 对比，而不是只与 reward/hybrid 对比。

**Bounded residual 的核心价值是限制 reward/return 影响。** 通过 `alpha_bound` 限制残差幅度，避免 reward/return 把 hard score 搞坏。当前 `alpha_bound=0.25`，best alpha 负值，说明在当前 checkpoint 下 reward+return 与真实 TC 排序的关系可能需要反向小幅校正。

**下一步不要再 graph-by-graph 训练单 scalar。** bounded residual 只训练一个标量，完全可以在固定 oracle TSV 上 grid-search alpha，成本更低、结果更稳定。推荐直接扫 `alpha`，用 expanded val 选，再过 transfer gate。

## 八、Transfer 评估经验

**Transfer gate 的定义。** transfer 使用固定 full-circuit oracle action groups，例如 `autoresearch/oracle-action-probe-260629-smoke/oracle_actions.tsv`，包含 `b15_C` 和 `i2c_aig` 的 6 groups / 288 actions。它不重新采样、不重新跑 backend，只 rescoring 同一批 actions。

**Transfer 不应覆盖 in-distribution 判断。** 训练分布是 sampled subckt，所以 primary gate 应是 expanded labeled-subckt val。transfer 只用于检查是否破坏大电路行为。一个方法如果 in-distribution 没提升，即使 transfer 安全，也不应 promote；一个方法如果 in-distribution 提升但 transfer 崩，也不能直接 promote。

**改变 dynamics 会破坏 transfer。** joint-hybrid 和 freeze-hard 实验都表明，一旦训练 action_encoder/dynamics，full-circuit hard/hybrid Spearman 容易下降。这说明 dynamics latent 表示承载了跨电路泛化能力，不能用小 oracle 数据直接大幅更新。

**冻结 dynamics/hard score 能保护 transfer。** heads_hybrid 和 bounded residual 都保持 transfer top1 safety/regret，不出现 joint training 的崩塌。这说明短期内安全路线应保持 action_encoder/dynamics/hard_reduction 固定。

## 九、工程工具链与 Bug 经验

**planner relation cache 曾有 graph id 复用 bug。** 长批量多 subckt oracle probe 时报错：`Expected size 318 but got size 404`。原因是 planner cache 用 `id(graph)` 做 key，Python 对象 id 在多图循环中可能复用，导致 relation features 来自旧 graph。修复方法是在 `tpi_jepa.plan` 增加 `clear_planner_caches()`，并在 `oracle_action_value_probe.py` 每处理一个 benchmark 前调用。

**Validation early stopping 是必要能力。** head-only hybrid 在 expanded 数据上出现 train loss 持续下降但 val Spearman 迅速变差的情况。如果只保存最后一轮，会选到明显过拟合 checkpoint。现在 `finetune_oracle_action_values.py` 支持 `--val-oracle-actions`，保存 `best.pt`。

**Checkpoint config 可以携带 planner score 参数。** bounded residual 不新增模型权重，只把 `bounded_residual_alpha` 和 `bounded_residual_alpha_bound` 写入 checkpoint config。`plan.py` 加载 checkpoint 后读取这些 config，即可在 scoring/gate 中复现新 score。

**固定 TSV 比重复 backend 更可靠。** 一旦 `oracle_actions.tsv` 生成，后续 checkpoint 比较都应优先使用 `evaluate_oracle_action_values.py` 重打分，而不是重新 probe。这样能避免 backend、候选采样和工作目录清理造成的不可控差异。

**所有实验都应写 handoff。** 当前 `autoresearch/*/handoff.json` 记录了数据、checkpoint、指标、推荐下一步。后续自动化或人工复盘都应从 handoff 读取，而不是从长日志里找结论。

## 十、当前结论与下一步路线

**当前不要 promote oracle-finetuned checkpoint。** 所有已测试 checkpoint 都没有同时满足 expanded held-out 排序提升、top1 安全、full-circuit transfer 稳定这三个条件。bounded residual 是方向上最干净的改动，但收益仍太小，不能作为默认 planner checkpoint。

**重新训练比继续 checkpoint finetune 更有前景。** 现有 checkpoint 上的补救实验已经显示：冻结太多学不动，解冻 dynamics 会破坏 transfer。因此更合理路线是从头重新训练 world model，把 oracle pairwise ranking loss 混入原始训练目标，让 representation 从早期就学习真实 action-value 排序。

**重新训练时不能只优化 oracle ranking。** Oracle 数据仍然相对少，如果只优化 oracle ranking，容易过拟合 expanded train groups。应保留原始 supervised/world-model losses、hard_reduction loss、reward/return loss，再小权重加入 oracle pairwise ranking loss，例如 `0.05/0.1/0.2` sweep。

**推荐下一阶段主实验。** 设计一次 scratch retrain：原始训练目标 + hard-fault/hard-reduction 目标 + oracle pairwise ranking loss。使用 expanded train oracle groups 作为 ranking 数据，expanded val oracle gate 选 checkpoint，full-circuit transfer gate 做风险检查。目标是让 action_encoder/dynamics 表示本身学会真实 TC 收益排序，而不是事后用 residual 修补。

**推荐短期低成本实验。** 在不重新训练前，可以做 bounded residual alpha grid-search：固定 score components，直接在 expanded oracle TSV 上扫 `alpha`，选择 expanded val 最优，再跑 transfer gate。这能确认 residual score 的上限，避免再花时间 graph-by-graph 训练单 scalar。

**推荐中期数据扩展。** 当前 oracle group 只支持 initial state。如果最终 planner 使用 rollout states，那么 oracle action-value 数据也应覆盖非 initial states。否则模型只学 initial-state action ranking，无法验证多步 planner 的真实收益排序。

**最终验收标准。** 一个可 promote 的方案应满足：expanded labeled-subckt val 上目标 score Spearman 高于 incumbent，negative_top1 不升，top1_regret 不升；full-circuit transfer 不出现 Spearman 明显退化或 top1 安全退化；并且该 score 能被实际 planner 使用，而不是只存在离线分析脚本里。

