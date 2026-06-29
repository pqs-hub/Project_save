# Plan: 从 400 个子电路采集 negative-rich oracle action groups

## 目标

从 400 个 sampled subckt 里重新采样，重点采集含有较多负收益 control 动作的 oracle action groups，然后重建 balanced train data。

这里的 “负收益 action” 指：

```text
插入这个 TP 后，真实 test coverage 反而下降，也就是 oracle_delta_tc < 0。
```

这里的 “group” 指：

```text
同一个 benchmark_id + state_id + candidate_strategy 下的一批候选动作。
```

排序模型真正需要的是：

```text
同一个 group 里既有好动作，也有坏动作，这样模型才知道谁该排前面、谁该排后面。
```

## 当前证据

400 个 bench 文件存在：

```text
/data4/pengqingsong/DFT/Dataset/atalanta_bist_lowtc_subckt_100k_labels/subcircuits/subckt_0000.bench
...
/data4/pengqingsong/DFT/Dataset/atalanta_bist_lowtc_subckt_100k_labels/subcircuits/subckt_0399.bench
```

但是历史 `labels.csv` 只覆盖 131 个子电路：

```text
/data4/pengqingsong/DFT/Dataset/atalanta_bist_lowtc_subckt_100k_labels/labels.csv
```

这不是问题。新 oracle action groups 可以直接从 400 个 bench 文件采样，因为 `find_bench_path()` 会从 `subcircuits/` 找 bench。

已有 oracle 数据的问题：

| split | rows | groups | negative rate | all-positive groups |
|---|---:|---:|---:|---:|
| train | 5184 | 288 | 13.81% | 168 |
| expanded val | 864 | 48 | 41.55% | 11 |
| transfer | 288 | 6 | 34.03% | 3 |

已有 balanced train 只剩：

```text
71 groups
1278 rows
negative rate = 50.08%
```

最低目标是 80 个 train groups，所以已有数据不够。

## 关键设计

不要直接把 400 个子电路全部跑 backend。

原因：

```text
400 subckts * 3 strategies * 12 nets * 3 action types = 43200 次 backend action eval。
```

这太贵，而且很多 group 可能仍然全是正收益，不能帮助排序模型学习“避开坏动作”。

正确做法是分阶段：

```text
先从 400 个里抽一批 pilot。
跑 control-heavy action pool。
审计哪些 group 负样本多。
如果还不够，再补第二批。
```

## 数据隔离

训练采样池：

```text
400 个 subckt bench
减去 expanded val 的 16 个 subckt
减去任何 transfer benchmark
```

保留现有 expanded val 不变：

```text
autoresearch/oracle-action-probe-260629-expanded-subckt-val/oracle_actions.tsv
```

原因：

```text
如果 validation 也重新换，下一次 ranker 变好时，不知道是模型真的变好，还是考试题换简单了。
```

transfer 只做最终考试：

```text
autoresearch/oracle-action-probe-260629-smoke/oracle_actions.tsv
```

不要把 transfer action 加入训练。

## 采样池

使用全部 400 个 bench 作为候选池：

```text
/data4/pengqingsong/DFT/Dataset/atalanta_bist_lowtc_subckt_100k_labels/subcircuits
```

优先从现有 train/val 之外的 subckt 里抽新样本。

如果新 subckt 不够，再允许复用现有 train subckt，但不能复用 expanded val subckt。

## Candidate Pool

使用 control-heavy 候选池：

```text
--action-types CP0,CP1,OP
--max-nets 12
```

每个 group 大约是：

```text
12 个 node * 3 个动作 = 36 actions
其中 24 个是 control0/control1
其中 12 个是 observe
```

这比旧的 18 actions/group 更容易暴露坏 control 动作。

候选策略：

```text
cached_hard_cone,cached_random,cached_stride
```

含义：

```text
cached_hard_cone: 优先看和 hard fault 相关的 cone。
cached_random: 随机抽，增加动作多样性。
cached_stride: 均匀扫一遍候选空间，防止只看局部。
```

## Phase 1: 生成 400-pool 采样 manifest

新增脚本：

```text
scripts/sample_negative_rich_oracle_subckts.py
```

职责：

```text
1. 扫描 400 个 subckt_*.bench。
2. 读取已有 train oracle 和 val oracle，知道哪些 subckt 已经用过。
3. 排除 expanded val subckt。
4. 优先从未 oracle 标注过的 subckt 里抽 pilot。
5. 输出 pilot_subckts.txt、topup_subckts.txt、pool_report.md。
```

命令：

```bash
python scripts/sample_negative_rich_oracle_subckts.py \
  --subckt-dir /data4/pengqingsong/DFT/Dataset/atalanta_bist_lowtc_subckt_100k_labels/subcircuits \
  --exclude-oracle autoresearch/oracle-action-probe-260629-expanded-subckt-val/oracle_actions.tsv \
  --existing-train-oracle autoresearch/oracle-action-probe-260629-expanded-subckt-train/oracle_actions.tsv \
  --pilot-count 48 \
  --topup-count 64 \
  --seed 260629 \
  --out-dir autoresearch/oracle-negative-rich-260629/sample
```

Acceptance:

```bash
python -m py_compile scripts/sample_negative_rich_oracle_subckts.py
test -s autoresearch/oracle-negative-rich-260629/sample/pilot_subckts.txt
test -s autoresearch/oracle-negative-rich-260629/sample/topup_subckts.txt
test -s autoresearch/oracle-negative-rich-260629/sample/pool_report.md
```

## Phase 2: Pilot backend 标注

用 incumbent checkpoint 给候选动作打模型分，再用 backend 标真实收益。

默认 checkpoint：

```text
autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt
```

命令：

```bash
python scripts/oracle_action_value_probe.py \
  --checkpoint autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt \
  --benchmarks "$(paste -sd, autoresearch/oracle-negative-rich-260629/sample/pilot_subckts.txt)" \
  --candidate-strategies cached_hard_cone,cached_random,cached_stride \
  --max-nets 12 \
  --action-types CP0,CP1,OP \
  --top-ks 1,3,5 \
  --oracle-top-m 1 \
  --patterns 300000 \
  --backend atalanta-bist \
  --candidate-sample-seed 260629 \
  --out-dir autoresearch/oracle-negative-rich-260629/pilot \
  --resume \
  --cleanup-workdir
```

Pilot 规模：

```text
48 subckts * 3 strategies * 36 actions = 5184 action evals
```

这和已有 train oracle 规模相当，成本可控。

Acceptance:

```bash
test -s autoresearch/oracle-negative-rich-260629/pilot/oracle_actions.tsv
python scripts/audit_oracle_action_groups.py \
  --oracle-tsv pilot=autoresearch/oracle-negative-rich-260629/pilot/oracle_actions.tsv \
  --out-dir autoresearch/oracle-negative-rich-260629/pilot_audit
test -s autoresearch/oracle-negative-rich-260629/pilot_audit/oracle_group_audit_report.md
```

## Phase 3: 判断是否需要 topup

先把旧 train 和 pilot 合并，再尝试构造 balanced train。

新增脚本：

```text
scripts/merge_oracle_action_tsv.py
```

职责：

```text
合并多个 oracle_actions.tsv。
按 benchmark_id + state_id + candidate_strategy + action_key 去重。
保持字段顺序。
输出 merge_report.md。
```

命令：

```bash
python scripts/merge_oracle_action_tsv.py \
  --input autoresearch/oracle-action-probe-260629-expanded-subckt-train/oracle_actions.tsv \
  --input autoresearch/oracle-negative-rich-260629/pilot/oracle_actions.tsv \
  --out-tsv autoresearch/oracle-negative-rich-260629/merged_train_oracle_actions.tsv \
  --out-report autoresearch/oracle-negative-rich-260629/merge_report.md
```

然后重建 balanced data：

```bash
python scripts/build_balanced_oracle_action_subset.py \
  --train-oracle autoresearch/oracle-negative-rich-260629/merged_train_oracle_actions.tsv \
  --val-oracle autoresearch/oracle-action-probe-260629-expanded-subckt-val/oracle_actions.tsv \
  --transfer-oracle autoresearch/oracle-action-probe-260629-smoke/oracle_actions.tsv \
  --min-negatives-per-group 3 \
  --min-positives-per-group 3 \
  --prefer-negative-types control0,control1 \
  --max-actions-per-group 24 \
  --min-train-groups 120 \
  --min-val-groups 24 \
  --out-dir autoresearch/oracle-balanced-negative-rich-260629
```

判断：

```text
如果 balanced train >= 120 groups，停止采集，进入 ranker 训练。
如果 balanced train < 120 groups，执行 topup。
```

为什么目标设成 120 而不是 80：

```text
80 只是最低能跑。
120 才能给 validation early stopping 和 transfer 泛化留一点余量。
```

## Phase 4: Topup backend 标注

如果 Phase 3 不够，再跑 topup subckts。

命令：

```bash
python scripts/oracle_action_value_probe.py \
  --checkpoint autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt \
  --benchmarks "$(paste -sd, autoresearch/oracle-negative-rich-260629/sample/topup_subckts.txt)" \
  --candidate-strategies cached_hard_cone,cached_random,cached_stride \
  --max-nets 12 \
  --action-types CP0,CP1,OP \
  --top-ks 1,3,5 \
  --oracle-top-m 1 \
  --patterns 300000 \
  --backend atalanta-bist \
  --candidate-sample-seed 260630 \
  --out-dir autoresearch/oracle-negative-rich-260629/topup \
  --resume \
  --cleanup-workdir
```

然后重新 merge：

```bash
python scripts/merge_oracle_action_tsv.py \
  --input autoresearch/oracle-action-probe-260629-expanded-subckt-train/oracle_actions.tsv \
  --input autoresearch/oracle-negative-rich-260629/pilot/oracle_actions.tsv \
  --input autoresearch/oracle-negative-rich-260629/topup/oracle_actions.tsv \
  --out-tsv autoresearch/oracle-negative-rich-260629/merged_train_oracle_actions.tsv \
  --out-report autoresearch/oracle-negative-rich-260629/merge_report.md
```

再重新 build balanced data。

## 最终 Acceptance

必须满足：

```text
balanced train groups >= 120
balanced train negative rate in [0.25, 0.60]
balanced train control-negative groups >= 80
expanded val groups >= 24
transfer 未进入训练集
```

必须生成：

```text
autoresearch/oracle-balanced-negative-rich-260629/balanced_train_oracle_actions.tsv
autoresearch/oracle-balanced-negative-rich-260629/balanced_val_oracle_actions.tsv
autoresearch/oracle-balanced-negative-rich-260629/balance_report.md
autoresearch/oracle-negative-rich-260629/merge_report.md
```

## 下一步不是训练模型

这个 plan 只负责把数据补好。

数据补好后，下一步才是：

```text
用 balanced_train_oracle_actions.tsv 训练 fixed-checkpoint ranker。
用 balanced_val_oracle_actions.tsv 做 early stopping。
用 transfer oracle 做最终 gate。
```

## 风险

风险 1：

```text
400 个 subckt 里仍然可能很多 group 是全正收益。
```

应对：

```text
先 pilot，再审计，再 topup，不盲目全跑。
```

风险 2：

```text
control-heavy 会让训练集比真实部署分布更偏向 control。
```

应对：

```text
保留 observe positive anchors，并且 validation/transfer 不改。
```

风险 3：

```text
如果新训练集只来自很少几个 subckt，ranker 会记住电路特征而不是动作价值。
```

应对：

```text
balanced train 至少 120 groups，并尽量覆盖不少于 40 个 subckt。
```

