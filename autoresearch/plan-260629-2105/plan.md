# Plan: 同时试方案 A 和方案 B + oracle pairwise ranking loss

## 目标

同时训练并评估：

```text
方案 A + oracle pairwise ranking loss
方案 B + oracle pairwise ranking loss
```

目的不是继续训练外接 ranker，而是把 oracle 排序信号接回主模型。

## 背景结论

当前 incumbent 的主模型 loss 实际生效项：

```text
jepa_loss
scoap_loss
delta_scoap_loss
hard_bce_loss
hard_soft_f1_loss
hard_count_loss
hard_reduction_loss
```

其中 `fc/reward`、`return`、`pattern` 都是 0。

之前方案 A：

```text
保留 direct hard_reduction head
关闭 hard_count / FC / return
```

之前方案 B：

```text
关闭 direct hard_count / direct hard_reduction / FC / return
只训练节点级 hard 标签
用节点 hard probability 推导 hard_count 和 hard_reduction
```

已有结果：

| scheme | 主要优点 | 主要问题 |
|---|---|---|
| A | hard F1 和 hard_reduction score 较强 | oracle ranking / transfer 不稳 |
| B | 节点 hard F1 还能学 | derived hard_reduction action value 太弱 |

外接 wide-balanced ranker 的新证据：

```text
wide balanced train = 180 groups / 4320 rows
去掉 benchmark_id 的 linear ranker 在 expanded val 大幅改善
transfer negative_top1 从 0.1667 降到 0
但 transfer Spearman/regret 仍不如 baseline
```

结论：

```text
oracle ranking 有信号，但不能直接相信外接 ranker。
下一步应把这个信号用小权重接回 A/B 主模型。
```

## Oracle 数据

训练 oracle：

```text
autoresearch/oracle-balanced-negative-rich-260629-wide/balanced_train_oracle_actions.tsv
```

验证 oracle：

```text
autoresearch/oracle-balanced-negative-rich-260629-wide/balanced_val_oracle_actions.tsv
```

transfer oracle：

```text
autoresearch/oracle-action-probe-260629-smoke/oracle_actions.tsv
```

训练 oracle 规模：

```text
180 groups
4320 rows
negative rate = 0.2880
```

验证 oracle 规模：

```text
37 groups
666 rows
negative rate = 0.5390
```

## 方案定义

### Scheme A

配置核心：

```text
lambda_hard_count = 0.0
lambda_hard_reduction = 0.5
lambda_fc = 0.0
lambda_return = 0.0
```

保留：

```text
direct hard_reduction_head
```

oracle ranking score：

```text
oracle_ranking_score_field = hard_reduction_total_pred
```

原因：

```text
A 的核心就是 direct hard_reduction。
不要用 hybrid_pred 训练，因为 reward/return loss 是 0，hybrid 会混入未被主任务约束的 residual。
```

SCOAP 设置：

主 sweep 用：

```text
lambda_scoap = 0.5
lambda_delta_scoap = 0.0
```

原因：

```text
A_only_scoap 之前在 expanded validation oracle gate 上明显优于 A_base。
虽然 transfer 仍不够好，但它是 A 里最值得接 oracle ranking 的起点。
```

### Scheme B

配置核心：

```text
lambda_hard_count = 0.0
lambda_hard_reduction = 0.0
lambda_fc = 0.0
lambda_return = 0.0
hard_value_mode = derived_from_node_hard
```

保留：

```text
节点级 hard label loss
```

oracle ranking score：

```text
oracle_ranking_score_field = derived_hard_reduction_hybrid_pred
```

原因：

```text
B 没有 direct hard_reduction head。
ranking loss 必须压到 derived hard-reduction score 上。
```

SCOAP 设置：

主 sweep 用：

```text
lambda_scoap = 0.0
lambda_delta_scoap = 0.3
```

原因：

```text
B_only_delta_scoap 之前 transfer 最好。
B_only_scoap 只在 expanded val 更好，但 transfer 更差。
```

## Oracle Loss Sweep

不要用大权重。

先跑：

```text
lambda_oracle_rank = 0.01
lambda_oracle_rank = 0.03
lambda_oracle_rank = 0.05
```

不要先跑：

```text
0.10 / 0.20
```

原因：

```text
wide balanced oracle 仍来自 sampled subckt。
权重大了容易把主模型带偏，尤其会破坏 transfer。
```

训练参数：

```text
oracle_actions = autoresearch/oracle-balanced-negative-rich-260629-wide/balanced_train_oracle_actions.tsv
oracle_batch_groups = 4
oracle_every_n_steps = 4
oracle_warmup_epochs = 1
oracle_ramp_epochs = 2
oracle_pairwise_min_delta = 0.001
oracle_pairwise_temperature = 1.0
oracle_max_actions_per_group = 0
epochs = 4
max_train_steps_per_epoch = 500
max_train_samples = 20000
max_val_samples = 4096
```

## Phase 1: Smoke

先只跑两个配置：

```text
A_oracle_0p03
B_oracle_0p03
```

目的：

```text
确认主模型 oracle ranking 训练入口能跑通；
确认 history.csv 里出现 oracle_loss / oracle_rank_loss / oracle_pairs；
确认 checkpoint 能被 oracle gate 重打分。
```

Acceptance:

```bash
test -s autoresearch/train-ab-oracle-rank-260629/runs/A_oracle_0p03/best.pt
test -s autoresearch/train-ab-oracle-rank-260629/runs/B_oracle_0p03/best.pt
rg -n "oracle_" autoresearch/train-ab-oracle-rank-260629/runs/A_oracle_0p03/history.csv
rg -n "oracle_" autoresearch/train-ab-oracle-rank-260629/runs/B_oracle_0p03/history.csv
```

如果 smoke 失败，先 fix，不进入 sweep。

## Phase 2: Full Sweep

生成 6 个主配置：

| variant | scheme | lambda_oracle_rank | score field |
|---|---|---:|---|
| A_oracle_0p01 | A | 0.01 | hard_reduction_total_pred |
| A_oracle_0p03 | A | 0.03 | hard_reduction_total_pred |
| A_oracle_0p05 | A | 0.05 | hard_reduction_total_pred |
| B_oracle_0p01 | B | 0.01 | derived_hard_reduction_hybrid_pred |
| B_oracle_0p03 | B | 0.03 | derived_hard_reduction_hybrid_pred |
| B_oracle_0p05 | B | 0.05 | derived_hard_reduction_hybrid_pred |

保留两个 no-oracle baseline 对照：

```text
A_only_scoap
B_only_delta_scoap
```

## Phase 3: Gates

每个 checkpoint 都跑三类 gate。

### Gate 1: Hard / SCOAP / delta-SCOAP

目标：

```text
不能因为 oracle ranking 把基础预测能力打坏。
```

重点看：

```text
hard_macro_f1_tuned
hard_reduction_score
derived_hard_reduction_score
SCOAP MAE
delta-SCOAP MAE
```

最低要求：

```text
hard_macro_f1_tuned >= incumbent - 0.03
```

### Gate 2: Expanded oracle validation

输入：

```text
autoresearch/oracle-balanced-negative-rich-260629-wide/balanced_val_oracle_actions.tsv
```

对 A 看：

```text
hard_reduction_total_pred
hybrid_pred
```

对 B 看：

```text
derived_hard_reduction_hybrid_pred
derived_hard_reduction_total_pred
```

最低要求：

```text
negative_top1_rate <= baseline
top1_regret <= baseline
mean_top1_real_delta_tc > 0
```

### Gate 3: Transfer oracle

输入：

```text
autoresearch/oracle-action-probe-260629-smoke/oracle_actions.tsv
```

这是最终 promotion gate。

硬要求：

```text
negative_top1_rate <= incumbent hybrid_pred
mean_top1_real_delta_tc >= 0
top1_regret <= incumbent hybrid_pred + 0.005
```

如果 transfer Spearman 低于 incumbent，但 top1 safety 明显好，可以标为：

```text
SAFE_BUT_NOT_STRONG
```

不能直接 promote 到默认 planner，只能进入 guarded rerank。

## Promotion Rules

### PROMOTE_MAIN_SCORE

同时满足：

```text
expanded val 比对应 no-oracle scheme 更好
transfer negative_top1 不变差
transfer top1_real_delta 不变差
transfer top1_regret 不变差
hard F1 不掉超过 0.03
```

### PROMOTE_GUARDED_RERANK

满足：

```text
expanded val 明显变好
transfer negative_top1 变好或不变差
但 transfer regret 或 Spearman 仍弱于 incumbent
```

使用方式：

```text
只做 planner rerank guard，不替换默认 score。
```

### REJECT

任一情况：

```text
transfer top1_real_delta 从正变负
negative_top1_rate 变差
hard F1 掉超过 0.03
expanded val 没有超过对应 no-oracle baseline
```

## Expected Interpretation

如果 A 成功、B 失败：

```text
direct hard_reduction head 是必要的。
后续主线用 A。
```

如果 B 成功：

```text
节点 hard 标签加 oracle ranking 可以救回 derived action value。
后续可以继续简化模型，减少 direct graph heads。
```

如果 A/B 都失败：

```text
oracle ranking 现在更适合外接 guarded ranker；
不要把它强塞进主模型训练。
```

## Files To Produce In Fix

```text
autoresearch/train-ab-oracle-rank-260629/configs/*.json
autoresearch/train-ab-oracle-rank-260629/runs/*/best.pt
autoresearch/train-ab-oracle-rank-260629/gates/hard/
autoresearch/train-ab-oracle-rank-260629/gates/expanded_val/
autoresearch/train-ab-oracle-rank-260629/gates/transfer/
autoresearch/train-ab-oracle-rank-260629/ab_oracle_rank_summary.tsv
autoresearch/train-ab-oracle-rank-260629/final_report.md
autoresearch/train-ab-oracle-rank-260629/handoff.json
```

