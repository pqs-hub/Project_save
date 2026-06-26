# AutoResearch Plan: Calibration Diagnostics + Action-Level Ranking Diagnostics

generated_at: 2026-06-26 03:25 Asia/Shanghai

Subcommand:

```text
$autoresearch plan
```

## 1. Goal

基于 `predictive-tech-ablation-260626` 的 stop rule，冻结当前主线：

```text
encoder_type = mean
summary_mode = global
lambda_hard_rank = 0.0
```

下一轮不训练新模型，先补齐两个诊断能力：

```text
calibration diagnostics
action-level ranking diagnostics
```

目标是回答：

1. hard F1 的 seed/split 波动来自哪些 benchmark、阈值、类别或 hard-positive-rate 区间？
2. 现有 `hard_reduction_pred` 是否已经具备 action 排序能力？
3. 如果要实现 action-level ranking loss，训练目标应该对齐哪个诊断指标？

## 2. Scope

### In Scope

允许修改：

```text
scripts/evaluate_hard_checkpoints.py
tpi_jepa/dataset.py
tpi_jepa/model.py
tpi_jepa/train.py
docs/hard_f1_autoresearch_report.md
```

实际优先级：

| File | Plan |
|---|---|
| `scripts/evaluate_hard_checkpoints.py` | 主改动。增加 calibration/action-ranking diagnostic 输出。 |
| `tpi_jepa/dataset.py` | 增加诊断需要的 metadata，例如 action group key、action node/type、pre-state key。 |
| `tpi_jepa/model.py` | 暂不新增训练 head。最多暴露/命名 action score 来源。 |
| `tpi_jepa/train.py` | 暂不新增 loss。只预留后续 action-ranking loss 入口说明。 |
| `docs/hard_f1_autoresearch_report.md` | 记录本轮诊断接口、输出和后续判断标准。 |

### Out of Scope

本轮不做：

```text
new model training
action-ranking loss implementation
new model head implementation
multi-view graph
large transformer / sparse attention
gate_dir/cone/rank 继续搜索
```

原因：

```text
predictive-tech-ablation-260626 已触发 stop rule。
现在应冻结架构，先建立可解释诊断报告和 action-level evidence。
```

## 3. Metric

本轮 metric 不是训练分数，而是诊断报告是否完整、可解释、可复现。

Primary deliverable:

```text
calibration_action_diagnostics.md
```

Required diagnostic tables:

```text
thresholds_by_class.tsv
threshold_sweep.tsv
per_benchmark_metrics.tsv
bucket_metrics.tsv
calibration_bins.tsv
action_ranking_metrics.tsv
action_group_examples.tsv
```

Calibration metrics:

| Metric | Meaning |
|---|---|
| `hard_macro_f1_tuned` | 当前主指标，继续保留 |
| `hard_sa0_f1_tuned` / `hard_sa1_f1_tuned` | 分类别 F1 |
| `hard_threshold_sa0` / `hard_threshold_sa1` | tuned thresholds |
| `brier_sa0` / `brier_sa1` | 概率校准误差 |
| `ece_sa0` / `ece_sa1` | expected calibration error |
| `positive_rate_sa0` / `positive_rate_sa1` | 类别稀疏度 |
| `fp_rate` / `fn_rate` | false positive/negative 来源 |

Action-ranking metrics:

| Metric | Meaning |
|---|---|
| `action_pairwise_acc` | 同一 state/action group 内 action score pairwise accuracy |
| `action_spearman` | predicted score 与 target hard reduction 的 rank correlation |
| `action_ndcg_at_5` / `action_ndcg_at_10` | top-k ranking quality |
| `action_top1_hit` | 预测 top1 是否命中真实最佳 action |
| `action_group_count` | 可比较 action group 数量 |
| `mean_group_size` | 每组候选 action 数量 |

Action score 初始定义：

```text
action_score = hard_reduction_pred[0]
```

也记录辅助 score：

```text
action_score_sa0 = hard_reduction_pred[1]
action_score_sa1 = hard_reduction_pred[2]
reward_score = reward_pred
```

## 4. Verify Config

用户指定 verify：

```bash
python scripts/evaluate_hard_checkpoints.py --help
```

当前已验证该命令可运行。

本计划实施后，`--help` 需要能显示新增参数：

```text
--diagnostics-dir
--write-calibration-diagnostics
--write-action-ranking-diagnostics
--calibration-bins
--action-score-field
--min-action-group-size
```

建议目标接口：

```bash
python scripts/evaluate_hard_checkpoints.py \
  --config autoresearch/stability-hardf1-seeds-2026-2030/configs/seed2026__lh0p5__lhc0p1__lhr0p5__lhrk0p0__encmean__sumglobal__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0.json \
  --run-dir autoresearch/stability-hardf1-seeds-2026-2030/runs/seed2026__lh0p5__lhc0p1__lhr0p5__lhrk0p0__encmean__sumglobal__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0 \
  --out-csv autoresearch/diagnostics-calibration-action-260626/target_metrics.csv \
  --out-png autoresearch/diagnostics-calibration-action-260626/target_metrics.png \
  --diagnostics-dir autoresearch/diagnostics-calibration-action-260626 \
  --write-calibration-diagnostics \
  --write-action-ranking-diagnostics \
  --calibration-bins 10 \
  --action-score-field hard_reduction_total \
  --min-action-group-size 2
```

This command evaluates existing checkpoints only. It must not train a model.

## 5. Implementation Plan

### Step 1: Dataset Metadata

Modify `TransitionSample` in `tpi_jepa/dataset.py` to include:

```text
action_node_name: str
action_type: str
state_key: str
sequence_id / benchmark_id / step when available
pre_action_count
```

Recommended `state_key`:

```text
benchmark_id + sorted(pre_actions)
```

Purpose:

```text
Group candidate actions from the same state for action-ranking diagnostics.
```

### Step 2: Evaluator Calibration Tables

Modify `scripts/evaluate_hard_checkpoints.py`:

1. Keep existing aggregate metrics unchanged.
2. While evaluating samples, collect per-node records:

```text
checkpoint
epoch
benchmark_id
node
sa0_prob
sa1_prob
sa0_target
sa1_target
hard_count_pred
hard_count_target
```

3. Write:

```text
thresholds_by_class.tsv
threshold_sweep.tsv
calibration_bins.tsv
per_benchmark_metrics.tsv
bucket_metrics.tsv
```

Bucket dimensions:

```text
benchmark_id
num_nodes bucket
hard_positive_rate bucket
action_type
```

### Step 3: Action-Ranking Diagnostics

In `scripts/evaluate_hard_checkpoints.py`, collect per-action records:

```text
checkpoint
epoch
benchmark_id
state_key
action_node_name
action_type
hard_reduction_pred_total
hard_reduction_target_total
hard_reduction_pred_sa0
hard_reduction_target_sa0
hard_reduction_pred_sa1
hard_reduction_target_sa1
reward_pred
delta_fault_coverage
```

Group by:

```text
(checkpoint, epoch, benchmark_id, state_key)
```

For groups with at least `--min-action-group-size`, compute:

```text
pairwise accuracy
Spearman
NDCG@5
NDCG@10
top1 hit
group size
target gain spread
```

Write:

```text
action_ranking_metrics.tsv
action_group_examples.tsv
```

### Step 4: Diagnostic Markdown Report

Write:

```text
calibration_action_diagnostics.md
```

Required sections:

1. Evaluated checkpoint list.
2. Calibration summary.
3. Worst benchmark buckets.
4. Threshold sensitivity.
5. Action-ranking summary.
6. Example action groups where ranking succeeds/fails.
7. Recommendation:
   - whether to implement action-ranking loss
   - which score field should be optimized
   - whether threshold calibration is enough to reduce seed variance

### Step 5: Documentation Update

Update:

```text
docs/hard_f1_autoresearch_report.md
```

Add:

```text
Calibration/action-ranking diagnostic interface
Expected outputs
How to interpret action-ranking metrics
Decision rule for implementing action-ranking loss
```

## 6. Acceptance Rules

### Diagnostic Completeness

Pass if the evaluator can produce:

```text
target_metrics.csv
thresholds_by_class.tsv
threshold_sweep.tsv
calibration_bins.tsv
per_benchmark_metrics.tsv
bucket_metrics.tsv
action_ranking_metrics.tsv
calibration_action_diagnostics.md
```

### Interpretability

Pass if the markdown report answers:

```text
Which class is poorly calibrated?
Which benchmark/bucket causes low hard F1?
Do current hard_reduction predictions rank actions better than random?
Is action ranking signal strong enough to justify a training loss?
```

### No-Training Constraint

Pass only if:

```text
No training command is run.
No checkpoint is modified.
Only evaluator/report code and docs are changed.
```

## 7. Decision Rules After Diagnostics

Implement action-ranking loss only if:

```text
action_pairwise_acc >= 0.60
or action_ndcg_at_10 >= 0.65
or top1 hit is meaningfully above random baseline
```

Prioritize calibration first if:

```text
hard PR-AUC is acceptable but F1 is low
and ECE/Brier/threshold sweep show large threshold fragility
```

Prioritize label/data audit if:

```text
worst benchmarks have extreme positive-rate drift
or action groups have near-zero target gain spread
```

## 8. Handoff

Next concrete task:

```text
Implement diagnostics in scripts/evaluate_hard_checkpoints.py and dataset metadata in tpi_jepa/dataset.py.
```

Verify command:

```bash
python scripts/evaluate_hard_checkpoints.py --help
```

Expected next `$autoresearch` mode:

```text
$autoresearch
```

with objective:

```text
Produce calibration/action-ranking diagnostic report without training.
```
