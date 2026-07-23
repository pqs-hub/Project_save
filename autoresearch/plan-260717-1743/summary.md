# 从项目理解导出的验证配置

用户本轮目标“理解项目”已经完成；下面是当前代码库自身正在追求、且可机械验证的下一研究目标，不会在本轮自动启动。

```text
$autoresearch
Goal: 在精确原始门级合法候选约束下，使 b15_C、b20_C、b21_C、b22_C、b17_C 的 final TC 分别超过 DeepTPI Table IV，并保持计划长度等于严格预算。
Scope: tpi_jepa/plan.py,scripts/run_gmean_sweep.py,scripts/plan_candidate_baseline.py,scripts/score_exact_itc99_vs_deeptpi.py,scripts/summarize_exact_itc99_eval.py,tests/test_candidate_context_scores.py,tests/test_itc99_exact_mapping.py
Metric: -sum(remaining per-circuit TC deficits) + macro_final_TC / 1000
Direction: higher_is_better
Verify: python scripts/score_exact_itc99_vs_deeptpi.py autoresearch/exact-itc99-current-best-q-lcb-260712/summary.json
Guard: PYTHONPATH=. pytest -q tests/test_candidate_context_scores.py tests/test_itc99_exact_mapping.py tests/test_score_exact_itc99_vs_deeptpi.py
Success: python scripts/score_exact_itc99_vs_deeptpi.py autoresearch/exact-itc99-current-best-q-lcb-260712/summary.json --check
Iterations: 20
```

当前 dry-run：metric `-4.399143600`，Success 输出 `FAIL`。注意：7 月 16–17 日局部实验已找到 b15 过线方案和 b17 91.545%，但这些结果尚未统一折叠到基准 summary；严格状态仍是未收敛。

建议在继续优化前增加一个外层保留集或预注册策略。b15/b17 已经被大量直接试验，继续把它们同时当“开发集”和“最终 held-out”会造成选择偏差。
