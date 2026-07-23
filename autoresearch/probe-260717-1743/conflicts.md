# 未解决矛盾与不确定性

1. **历史“8/8 超过 DeepTPI”与当前严格口径冲突。** `autoresearch/improve-260706-0959/current_best.json` 使用 90.61/90.60 等参考值；当前 `scripts/summarize_exact_itc99_eval.py` 明确使用 Table IV 最终 TC 93.20/95.02/94.51/95.59/91.67。后者是当前优化采用的正确口径。
2. **恢复 AIG 与原始门级可实施性冲突。** 不加白名单的计划可选 AIG 展开临时节点。ITC'99 已解决；三个 EPFL 电路仍缺精确来源版本，完整 8 电路合法比较尚未闭环。
3. **训练代理目标与最终目标不完全对齐。** checkpoint 按 val loss 保存，但最终关心同状态 action ranking 与 backend TC；两者可能分离。
4. **状态更新近似与真实网表状态冲突。** feature/SCOAP proxy 和 latent rollout 不等价于真实插点后的逻辑网表，长 horizon 误差可能累积。
5. **旧配置与当前机器冲突。** README 和若干 JSON 仍有 `/data3` 路径，本机对应文件不存在；代码只会重映射部分 sidecar/BENCH 路径，不会自动重映射 `config["labels"]`。
6. **当前工作树归属。** `plan.py` 白名单、context tie-break、hard-cluster seed 上限等改动尚未提交，应视为用户正在进行的研究，不在本次理解任务中修改。
