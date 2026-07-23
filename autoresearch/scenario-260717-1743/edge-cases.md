# 严重度排序边界清单

## Critical

- 最终电路/alias 泄漏进训练或 oracle group。
- 反复在最终五/八电路调 planner，把 held-out 变成开发集。
- 比较时使用错误论文表格列。
- DeepTPI 恢复网表与原始来源版本不一致。
- plan node/type 到 backend test point 的映射不一致。
- 严格预算大于合法唯一候选数时仍伪造完整计划。

## High

- latent rollout 远超训练 horizon；proxy 状态与真实网表状态偏离。
- backend 超时、CUDA 不可用或并行写入失败被误记为成功。
- candidate/global cache 跨电路污染。
- `/data3` 旧路径导致训练入口在加载标签时失败。
- 计划完成后才删除非法 AIG 节点，破坏预算和序列语义。

## Medium

- pool 大小改变 robust z-score，从而改变 context rerank。
- 多次重试结果未按 `status=ok` 去重。
- 隐式 config schema 使拼写错误落到默认值。
- 大型 artifact 树拖慢 git、pytest 和知识定位。
