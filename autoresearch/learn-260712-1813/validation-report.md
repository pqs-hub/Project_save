# Validation Report

生成时间：2026-07-12

## 验证结果

| 检查 | 状态 | 证据 |
|---|---|---|
| Python 语法 | PASS | `python -m py_compile tpi_jepa/*.py scripts/*.py` |
| 定向单元测试 | PASS | `PYTHONPATH=. pytest -q tests` → `18 passed in 1.56s` |
| 主 labels 路径 | PASS | `/data4/.../atalanta_bist_lowtc_subckt_100k_labels/labels.csv` 存在 |
| eval protocol JSON | PASS | 可由 `jq` 解析，包含 8 个 benchmark、预算、backend、patterns、seed |
| current-best checkpoints | PASS | 三个 `best_final_horizon.pt` 均存在 |
| current-best result files | PASS | `summary.tsv` 和 `comparison_final_tc_vs_deeptpi.tsv` 均存在 |
| current-best 数值一致性 | PASS | JSON、Markdown、comparison TSV 的 q-LCB 8 电路结果一致 |
| GPU/backend 独立复算 | NOT RUN | summarize 模式不启动长时间 planning/evaluation |

## 发现的问题

1. 裸 `pytest -q` 会进入巨型实验资产树，106 秒后仍显示 `no tests ran`；已中止。
2. `pytest -q tests` 在当前 pytest console entry 环境下出现 `ModuleNotFoundError: tpi_jepa/scripts`。
3. 修正命令 `PYTHONPATH=. pytest -q tests` 后 18 个测试全部通过。
4. README 中 `/data3` 和 `/data4` 路径混用。
5. 基础 codebase guide 未覆盖最新 Q-LCB ensemble 主线。
6. 探索知识库停留在 2026-06-29，未汇总 7 月 strict held-out 结果。
7. current-best 原始 summary TSV 含失败重试行；下游统计必须筛选 `status=ok`。

## 本次处理

本次任务目标是理解并总结项目，没有修改源代码或既有文档。上述问题均在 `summary.md` 中显式标注，并给出稳定测试命令。
