# Fix Report: 400-pool negative-rich oracle collection

## Status

`completed_ready_for_backend_collection`

本次 fix 完成了采样、合并、重建 balanced data 的工程入口，并按“组数越多越好，训练数据越多越好”的要求扩大了采样规模。

没有在本次 fix 里直接启动完整 backend 采集，因为当前 pilot 就是：

```text
96 subckts * 3 strategies * 12 nets * 3 actions = 10368 action evals
```

topup 是：

```text
192 subckts * 3 strategies * 12 nets * 3 actions = 20736 action evals
```

这属于长任务，应该单独启动并用 `--resume` 续跑。

## Code Changes

新增：

```text
scripts/sample_negative_rich_oracle_subckts.py
scripts/merge_oracle_action_tsv.py
```

修改：

```text
scripts/build_balanced_oracle_action_subset.py
```

修改点：

```text
--max-actions-per-group 0 表示保留每个 kept group 的全部 actions。
```

原因：

```text
用户要求训练数据越多越好，所以不能默认把每个 group 裁成 18 或 24 条。
```

## Generated Sampling Pool

输出目录：

```text
autoresearch/oracle-negative-rich-260629/sample
```

结果：

| item | count |
|---|---:|
| all subckt bench files | 400 |
| excluded expanded-val subckts | 16 |
| eligible train-pool subckts | 384 |
| existing train oracle subckts | 96 |
| fresh eligible subckts | 288 |
| pilot subckts | 96 |
| topup subckts | 192 |
| remaining after pilot/topup | 96 |

重要检查：

```text
pilot ∩ expanded_val = 0
topup ∩ expanded_val = 0
pilot ∩ existing_train = 0
topup ∩ existing_train = 0
pilot ∩ topup = 0
```

含义：

```text
pilot 和 topup 一共覆盖 288 个全新的、没有进入 expanded val 的 subckt。
```

最多新增 raw groups：

```text
288 fresh subckts * 3 candidate strategies = 864 groups
```

最多新增 raw action rows：

```text
288 fresh subckts * 3 strategies * 36 actions = 31104 rows
```

加上已有 train oracle：

```text
已有 288 groups / 5184 rows
```

合并后理论 raw 上限：

```text
1152 groups / 36288 rows
```

实际 balanced 后会少一些，因为仍然会过滤掉没有足够正负样本的 group。

## Runner Scripts

新增：

```text
autoresearch/oracle-negative-rich-260629/run_pilot.sh
autoresearch/oracle-negative-rich-260629/run_topup.sh
autoresearch/oracle-negative-rich-260629/rebuild_balanced_after_collection.sh
```

执行顺序：

```bash
./autoresearch/oracle-negative-rich-260629/run_pilot.sh
./autoresearch/oracle-negative-rich-260629/rebuild_balanced_after_collection.sh
```

如果 pilot 后 balanced train 仍然不够多：

```bash
./autoresearch/oracle-negative-rich-260629/run_topup.sh
./autoresearch/oracle-negative-rich-260629/rebuild_balanced_after_collection.sh
```

rebuild 脚本会自动检测：

```text
pilot/oracle_actions.tsv 是否存在
topup/oracle_actions.tsv 是否存在
```

存在就合并，不存在就跳过。

## Current Verification

已执行：

```bash
python -m py_compile scripts/sample_negative_rich_oracle_subckts.py scripts/merge_oracle_action_tsv.py scripts/build_balanced_oracle_action_subset.py
bash -n autoresearch/oracle-negative-rich-260629/run_pilot.sh autoresearch/oracle-negative-rich-260629/run_topup.sh autoresearch/oracle-negative-rich-260629/rebuild_balanced_after_collection.sh
python -m json.tool autoresearch/oracle-negative-rich-260629/sample/sample_manifest.json
python -m json.tool autoresearch/oracle-negative-rich-260629/merge_report.json
python -m json.tool autoresearch/oracle-balanced-negative-rich-260629/balanced_manifest.json
git diff --check -- scripts/sample_negative_rich_oracle_subckts.py scripts/merge_oracle_action_tsv.py scripts/build_balanced_oracle_action_subset.py
```

结果：

```text
all passed
```

## Existing-only Baseline

在还没跑新 backend 前，rebuild 只用了旧 train oracle。

结果：

| split | input groups | kept groups | kept rows | negative rate |
|---|---:|---:|---:|---:|
| train | 288 | 71 | 1278 | 0.5008 |
| expanded_val | 48 | 34 | 612 | 0.5801 |
| transfer_eval_only | 6 | 0 | 0 | 0.3403 |

结论：

```text
旧数据不够；必须跑 pilot/topup backend collection。
```

## Next Step

启动 pilot：

```bash
./autoresearch/oracle-negative-rich-260629/run_pilot.sh
```

pilot 完成后重建数据：

```bash
./autoresearch/oracle-negative-rich-260629/rebuild_balanced_after_collection.sh
```

如果 group 数仍然不够多，再启动 topup：

```bash
./autoresearch/oracle-negative-rich-260629/run_topup.sh
./autoresearch/oracle-negative-rich-260629/rebuild_balanced_after_collection.sh
```

