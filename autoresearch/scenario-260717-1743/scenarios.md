# 项目边界场景

共 36 个场景，覆盖 happy path、validation、permissions、concurrency、state、scale、failure、security、integration、data、UX、recovery 12/12 维度。完整逐条记录见 `scenario-results.tsv`。

最高风险集中在五类：

1. 训练/选模泄漏最终评估电路；
2. 用错误 DeepTPI 列或错误原始网表版本得出论文胜负；
3. AIG 临时节点被误当成原始门级可插点位置；
4. latent/proxy 状态在长计划中漂移；
5. CUDA、backend 或并行任务失败后被误聚合为成功。

本轮最有价值的新场景是“预算超过合法唯一节点”“最终电路被反复调参后不再 held-out”“旧 `/data3` 标签路径直接失败”和“pool 大小改变 context score 标度”。
