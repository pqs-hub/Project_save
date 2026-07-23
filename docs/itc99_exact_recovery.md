# DeepTPI ITC'99 精确结构恢复

DeepTPI 论文要求只有在原始电路中有对应门位置的 AIG 节点才能成为测试点候选。
五个 ITC'99 电路的公开 NPZ 保留了确定性的编号布局：若原网表有 `P` 个 PI、
`G` 个门，则：

```text
N0 ... N(P-1)       原始 PI，顺序与 BENCH 相同
NP ... N(P+G-1)     原始门锚点，顺序与 BENCH 门赋值相同
其余 N 节点          AIG 展开临时节点或 fanout BUFF
```

恢复工具不仅依赖编号，还逐门进行三项验证：

1. 从 DeepTPI 锚点向后穿过临时 AND/NOT/BUFF，边界必须恰好等于该原始门的扇入；
2. 对门的全部局部输入组合穷举，AIG 锥真值表必须与原始 AND/NAND/OR/NOR/NOT
   完全一致；
3. 使用 1024 位全局模拟签名交叉检查整个电路中的映射。

局部证明按原始门拓扑归纳后，建立了原始门和 DeepTPI 锚点的一一结构对应；它
不依赖“同功能网中任选一个”的启发式。

## 结果

| 电路 | 锚点范围 | 精确映射原始门 | 排除的 PO 锚点 | 论文合法候选 | 临时 AIG 节点 |
|---|---|---:|---:|---:|---:|
| b15_C | N485–N8851 | 8367 | 449 | 7918 | 18986 |
| b17_C | N1452–N32228 | 30777 | 1443 | 29334 | 67237 |
| b20_C | N522–N20203 | 19682 | 507 | 19175 | 41463 |
| b21_C | N522–N20548 | 20027 | 507 | 19520 | 42268 |
| b22_C | N767–N29928 | 29162 | 750 | 28412 | 61576 |
| 合计 | — | 108015 | 3656 | 104359 | 231530 |

“论文合法候选”是已证明的原始门锚点再排除 DeepTPI 环境会屏蔽的无扇出节点。
PI 从来不进入候选集。

## 输出

每个电路目录位于：

```text
autoresearch/original-netlist-recovery-260712/exact_itc99/<circuit>/
```

- `exact_node_mapping.tsv`：PI/原始门与 `Nxxx` 的精确对应及逐门证明状态；
- `exact_candidates.tsv`：带原始网名、门型和顺序的论文合法候选；
- `exact_candidate_nodes.txt`：供候选生成器使用的 `Nxxx` 白名单；
- `candidate_cache.json`：可供 planner cached-candidate 策略加载的候选缓存；
- `report.json`：机器可读验证报告。

单电路复现命令：

```bash
python scripts/recover_itc99_exact_mapping.py \
  --source-bench autoresearch/original-netlist-recovery-260712/sources/b15_C.bench \
  --deep-bench autoresearch/deeptpi_table2_restored_bench/b15_C.bench \
  --out-dir autoresearch/original-netlist-recovery-260712/exact_itc99/b15_C
```

对已有计划进行合法性审计：

```bash
python scripts/remap_plan_to_original.py \
  --plan path/to/plan.csv \
  --mapping autoresearch/original-netlist-recovery-260712/exact_itc99/b15_C/exact_node_mapping.tsv \
  --output path/to/exact_mapped_plan.csv \
  --require-all-safe
```

现有 b15 restored-AIG 278 点计划的示例审计结果为 81 个合法原始门锚点、197 个
AIG 临时节点。这说明后续规划必须在候选生成前应用精确白名单，不能在计划完成后
简单删除非法节点。
