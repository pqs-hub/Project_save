# AutoResearch Plan: Calibration / Threshold Deepening

generated_at: 2026-06-26 03:45 Asia/Shanghai

Subcommand:

```text
$autoresearch plan
```

## 1. Goal

基于 `diagnostics-calibration-action-260626` 的结论，下一轮优先深化 calibration / threshold / benchmark-bucket 评估。

冻结当前模型主线：

```text
encoder_type = mean
summary_mode = global
lambda_hard_rank = 0.0
```

暂不做：

```text
action-ranking loss
new model training
new model head
gate_dir / cone / rank search
```

当前证据：

```text
mean_ece_sa0 = 0.1827
mean_ece_sa1 = 0.4578
action_pairwise_acc = 0.4333
action_ndcg_at_10 = 0.0000
action_top1_hit = 0.0000
```

结论：

```text
Calibration 问题优先级高于 action-ranking loss。
```

## 2. Scope

### In Scope

修改范围：

```text
scripts/evaluate_hard_checkpoints.py
docs/hard_f1_autoresearch_report.md
```

可选小改：

```text
scripts/run_predictive_autoresearch.py
```

仅当需要把校准指标汇总到 autoresearch summary 时再改。

### Out of Scope

```text
tpi_jepa/train.py
tpi_jepa/model.py
tpi_jepa/dataset.py
new training
new loss
new head
```

理由：

```text
本轮目标是对已有 checkpoint 做校准模式比较，不改变模型。
```

## 3. Metric

本轮主产物：

```text
calibration_mode_report.md
```

必须输出：

```text
calibration_mode_metrics.tsv
per_benchmark_calibrated_metrics.tsv
threshold_policy_comparison.tsv
calibration_mode_report.md
```

核心指标：

| Metric | Meaning |
|---|---|
| `hard_macro_f1_tuned` | 原始 tuned F1 |
| `hard_macro_f1_global_threshold` | 全局阈值 F1 |
| `hard_macro_f1_class_threshold` | SA0/SA1 分类别阈值 F1 |
| `hard_macro_f1_benchmark_shrinkage` | benchmark threshold shrinkage 后 F1 |
| `ece_sa0_before/after` | SA0 calibration 改善 |
| `ece_sa1_before/after` | SA1 calibration 改善 |
| `worst_benchmark_f1_before/after` | 最差 benchmark 是否改善 |
| `threshold_variance_sa0/sa1` | 阈值是否过度波动 |

Calibration policy candidates:

```text
global_0p5
class_tuned
benchmark_tuned
benchmark_shrinkage_alpha_0p25
benchmark_shrinkage_alpha_0p50
benchmark_shrinkage_alpha_0p75
```

Shrinkage definition:

```text
threshold = alpha * benchmark_threshold + (1 - alpha) * global_class_threshold
```

## 4. Verify Config

先验证接口：

```bash
python scripts/evaluate_hard_checkpoints.py --help
```

实施后 `--help` 需要出现：

```text
--write-calibration-policy-report
--calibration-policy
--benchmark-threshold-shrinkage
```

建议运行命令：

```bash
python scripts/evaluate_hard_checkpoints.py \
  --config autoresearch/stability-hardf1-seeds-2026-2030/configs/seed2026__lh0p5__lhc0p1__lhr0p5__lhrk0p0__encmean__sumglobal__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0.json \
  --run-dir autoresearch/stability-hardf1-seeds-2026-2030/runs/seed2026__lh0p5__lhc0p1__lhr0p5__lhrk0p0__encmean__sumglobal__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0 \
  --out-csv autoresearch/calibration-policy-260626/target_metrics.csv \
  --out-png autoresearch/calibration-policy-260626/target_metrics.png \
  --diagnostics-dir autoresearch/calibration-policy-260626 \
  --write-calibration-diagnostics \
  --write-calibration-policy-report \
  --calibration-bins 10 \
  --max-val-samples 1024 \
  --max-steps 256 \
  --device cpu
```

No training is allowed.

## 5. Implementation Plan

### Step 1: Store Enough Calibration Records

`evaluate_hard_checkpoints.py` already collects per-node scores/targets internally.

Extend diagnostic output to compute policy-level metrics from these records:

```text
global_0p5
class_tuned
benchmark_tuned
benchmark_shrinkage
```

### Step 2: Add Policy Metrics

For each checkpoint:

1. Compute global class thresholds from all validation nodes.
2. Compute benchmark-specific class thresholds.
3. Apply shrinkage alphas:

```text
alpha in {0.25, 0.50, 0.75}
```

4. Report macro F1, SA0 F1, SA1 F1, FP/FN rates.
5. Report worst benchmark F1 under each policy.

### Step 3: Add Report

Write:

```text
calibration_mode_report.md
```

Required sections:

1. Why calibration is now prioritized.
2. Global vs class vs benchmark thresholds.
3. ECE/Brier before/after summary.
4. Worst benchmark / bucket change.
5. Recommendation:
   - use global/class threshold only
   - use benchmark shrinkage
   - perform label/data audit
   - proceed to action-ranking later

### Step 4: Update Main Report

Update:

```text
docs/hard_f1_autoresearch_report.md
```

Add:

```text
calibration policy interface
policy comparison outputs
decision rule for threshold calibration
```

## 6. Acceptance Rules

### Interface

Pass if:

```bash
python scripts/evaluate_hard_checkpoints.py --help
```

shows:

```text
--write-calibration-policy-report
--calibration-policy
--benchmark-threshold-shrinkage
```

### Output Completeness

Pass if the run writes:

```text
calibration_mode_metrics.tsv
per_benchmark_calibrated_metrics.tsv
threshold_policy_comparison.tsv
calibration_mode_report.md
```

### Decision Quality

Pass if report answers:

```text
Does class-wise thresholding reduce SA1 calibration/F1 failure?
Does benchmark shrinkage improve worst-benchmark F1?
Does calibration improve enough to justify reporting calibrated F1?
Should next step be full 5-seed calibrated eval or label audit?
```

## 7. Decision Rules After This Plan

Proceed to full 5-seed calibrated evaluation if:

```text
worst_benchmark_f1 improves by >= 0.10
or hard_macro_f1 improves by >= 0.03
without increasing ECE
```

Do label/data audit if:

```text
benchmark-specific thresholds are extreme
or positive-rate drift explains most low-F1 buckets
```

Return to action-ranking only after:

```text
calibrated hard F1 is stable enough
and action groups show pairwise_acc >= 0.60 or NDCG@10 >= 0.65
```

## 8. Next Command

Recommended next command:

```text
$autoresearch
```

with goal:

```text
Implement calibration policy comparison in evaluator and produce calibration_mode_report.md without training.
```
