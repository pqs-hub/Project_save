# DeepTPI 节点到原始门级网名恢复

> ITC'99 五个电路现已完成逐门结构精确恢复。其结果和推荐白名单见
> `docs/itc99_exact_recovery.md`。本页下述功能指纹方法主要保留为诊断和非精确来源的
> 保守回退，不应替代 ITC'99 的精确映射。

DeepTPI 发布的 `benchmarks_circuits_graphs.npz` 在保存前调用了
`rename_node`，因此只保留节点编号和边，不保留原 BENCH 网名。本项目恢复出的
`Nxxx` 只能保证在恢复后的 AIG/BENCH 中存在，不能直接证明它是原始门级网表中
可插点的信号。

本工具采用保守恢复流程：

1. 为 DeepTPI AIG 与公开原始网表的同序 PI 施加 1024 位确定性随机向量；
2. 对每个节点计算功能指纹，并查找原始网表中功能完全相同的门输出；
3. 只有唯一匹配、非 PO、非时钟/复位/扫描信号且非转换新增 BUFF 的节点，才标为
   `safe_insertable=True`；
4. 至少 90% 的 DeepTPI 输出必须与原始输出对齐，否则关闭整个电路的安全映射。

功能指纹不是形式等价证明。用于流片级结论前，应对最终少量节点再跑形式等价。

## 使用

下载并固定公开原始文件：

```bash
python scripts/fetch_original_netlists.py \
  --out-dir autoresearch/original-netlist-recovery-260712/sources
```

恢复一个电路：

```bash
python scripts/recover_original_net_mapping.py \
  --deep-bench autoresearch/deeptpi_table2_restored_bench/b15_C.bench \
  --source-netlist autoresearch/original-netlist-recovery-260712/sources/b15_C.bench \
  --out-dir autoresearch/original-netlist-recovery-260712/mappings/b15_C \
  --patterns 1024
```

输出包括：

- `node_mapping.tsv`：逐节点映射、歧义原因和安全标志；
- `safe_deep_nodes.txt`：候选生成阶段可直接使用的安全节点白名单；
- `report.json`：PI/PO 对齐、覆盖率和限制说明。

把已有计划恢复成原始网名并审计：

```bash
python scripts/remap_plan_to_original.py \
  --plan path/to/plan.csv \
  --mapping path/to/node_mapping.tsv \
  --output path/to/recovered_plan.csv \
  --require-all-safe
```

`--require-all-safe` 在计划包含合成分支、歧义节点或缺失映射时以状态码 2 退出，
避免把这些节点误交给真实门级插点流程。

## 当前八个电路的恢复状态

| 电路 | PI 对齐 | PO 对齐 | 可安全映射 Deep 节点 | 状态 |
|---|---:|---:|---:|---|
| b15_C | 485/485 | 449/449 | 6015 | 可用 |
| b17_C | 1452/1452 | 1443/1443 | 18177 | 可用 |
| b20_C | 522/522 | 507/507 | 10626 | 可用 |
| b21_C | 522/522 | 507/507 | 10783 | 可用 |
| b22_C | 767/767 | 750/750 | 15337 | 可用 |
| i2c_aig | 136/147 | 未运行 | 0 | EPFL 2015 的未公开预处理版本 |
| max_aig | 512/512 | 0/129 | 0 | PI 排列/版本未验证 |
| mem_ctrl_aig | 1028/1204 | 未运行 | 0 | EPFL 2015 的未公开预处理版本 |

因此当前只能把 ITC99 五个电路的白名单接入真实插点；三个 EPFL 电路必须先取得
生成 DeepTPI NPZ 时使用的确切 BENCH 文件或 PI 名称排列，不能猜测。`i2c` 和
`mem_ctrl` 的逐版本调查见
`autoresearch/original-netlist-recovery-260712/source_version_audit.md`。
