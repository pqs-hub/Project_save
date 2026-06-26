# AutoResearch Plan: Hard-Head Calibration Loss

generated_at: `2026-06-26 03:59 Asia/Shanghai`

## Goal

```text
实现 hard-head calibration loss 与 evaluator temperature/Brier 诊断，
围绕 frozen mean/global/no-rank 主线设计下一轮可输出进度的训练命令。
```

## Context

上一轮 improve 的结论：

- 当前主线应冻结为 `mean + global + no-rank + ASL + residual_context + topk`。
- 固定 `0.5` hard threshold 明显失败。
- benchmark-tuned / shrinkage threshold 没有超过 class-tuned。
- `gate_dir`、`cone`、`hard_rank` 暂不进入主线。
- action-ranking loss 暂缓，先补 action group 数据质量。

本计划只推进 calibration / boundary 稳定性，不引入多视图，不重开 encoder 大扫。

## Scope

Primary files:

```text
tpi_jepa/train.py
scripts/evaluate_hard_checkpoints.py
scripts/run_predictive_autoresearch.py
docs/hard_f1_autoresearch_report.md
```

Optional files:

```text
configs/aig_lowtc_100k_hard_pretrain.json
```

Out of scope:

```text
tpi_jepa/model.py 结构大改
tpi_jepa/dataset.py 数据结构大改
gate_dir / cone / hard_rank 复扫
多视图 AIG/PM netlist
直接 action-ranking loss
planner-in-the-loop 训练
```

## Implementation Plan

### Step 1: Evaluator calibration diagnostics

Modify `scripts/evaluate_hard_checkpoints.py`.

Add metrics:

```text
hard_macro_f1_at_0p5
hard_sa0_f1_at_0p5
hard_sa1_f1_at_0p5
temperature_scaled_ece_sa0
temperature_scaled_ece_sa1
temperature_scaled_brier_sa0
temperature_scaled_brier_sa1
temperature_sa0
temperature_sa1
delta_ece_sa0_after_temperature
delta_ece_sa1_after_temperature
```

Add flags:

```text
--temperature-scale-hard
--temperature-grid
--write-hard-calibration-report
```

Expected behavior:

- Fit scalar temperature independently for SA0 and SA1 on validation logits.
- Use a simple grid, e.g. `0.5,0.75,1.0,1.25,1.5,2.0,3.0,4.0`.
- Optimize NLL or Brier; default to Brier because current concern is calibration.
- Report whether temperature scaling reduces ECE without changing model weights.

### Step 2: Train-time Brier / soft-F1 loss

Modify `tpi_jepa/train.py`.

Add config keys:

```text
lambda_hard_brier
lambda_hard_soft_f1
hard_soft_f1_eps
```

Loss definition:

```text
hard_brier = mean((sigmoid(hard_logits) - hard_targets)^2)

soft_f1_loss = 1 - macro_soft_f1(SA0, SA1)
soft_f1 = 2 * soft_tp / (2 * soft_tp + soft_fp + soft_fn + eps)
```

Total loss extension:

```text
loss += lambda_hard_brier * hard_brier
loss += lambda_hard_soft_f1 * soft_f1_loss
```

Logging:

```text
hard_brier_loss
hard_soft_f1_loss
```

Default values must preserve current behavior:

```text
lambda_hard_brier = 0.0
lambda_hard_soft_f1 = 0.0
```

### Step 3: AutoResearch runner grid support

Modify `scripts/run_predictive_autoresearch.py`.

Add result fields:

```text
lambda_hard_brier
lambda_hard_soft_f1
hard_asl_gamma_neg
hard_asl_clip
```

Add flags:

```text
--lambda-hard-briers
--lambda-hard-soft-f1s
--hard-asl-gamma-negs
--hard-asl-clips
--center-lambda-hard-brier
--center-lambda-hard-soft-f1
```

Variant grid should keep the frozen mainline fixed unless explicitly overridden:

```text
encoder_type=mean
summary_mode=global
lambda_hard_rank=0.0
hard_loss=asl
hard_head_type=residual_context
hard_negative_mining=topk
train_sample_strategy=hard_weighted
edge_weight_mode=fault_path
edge_keep_ratio=0.6
```

### Step 4: Documentation

Update `docs/hard_f1_autoresearch_report.md` with:

- calibration-loss plan summary
- new config keys
- exact run command
- promotion / stop rules

## Metric

Primary:

```text
hard_macro_f1_tuned
```

Guardrails:

```text
predictive_score
hard_recall_at_top_10pct
hard_macro_f1_at_0p5
ece_sa0
ece_sa1
temperature_scaled_ece_sa0
temperature_scaled_ece_sa1
hard_reduction_score
hard_count_top10_overlap
```

Promotion rule:

```text
Promote a calibration-loss route only if:
1. hard_macro_f1_tuned improves by >= 0.03 over frozen mainline, OR
2. mean ECE drops by >= 20% with hard_macro_f1_tuned loss <= 0.01.
```

Reject rule:

```text
Reject if predictive_score drops by > 0.03,
or hard_recall_at_top_10pct drops by > 0.05,
or tuned F1 improves only while hard_macro_f1_at_0p5 remains unusable.
```

## Verify

After implementation:

```bash
python -m py_compile \
  tpi_jepa/train.py \
  scripts/evaluate_hard_checkpoints.py \
  scripts/run_predictive_autoresearch.py

python scripts/evaluate_hard_checkpoints.py --help
python scripts/run_predictive_autoresearch.py --help
```

Diagnostic smoke run, no training:

```bash
python scripts/evaluate_hard_checkpoints.py \
  --config autoresearch/stability-hardf1-seeds-2026-2030/configs/seed2026__lh0p5__lhc0p1__lhr0p5__lhrk0p0__encmean__sumglobal__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0.json \
  --run-dir autoresearch/stability-hardf1-seeds-2026-2030/runs/seed2026__lh0p5__lhc0p1__lhr0p5__lhrk0p0__encmean__sumglobal__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0 \
  --out-csv autoresearch/calibration-loss-smoke-260626/target_metrics.csv \
  --out-png autoresearch/calibration-loss-smoke-260626/target_metrics.png \
  --diagnostics-dir autoresearch/calibration-loss-smoke-260626 \
  --write-calibration-diagnostics \
  --write-hard-calibration-report \
  --temperature-scale-hard \
  --temperature-grid 0.5,0.75,1.0,1.25,1.5,2.0,3.0,4.0 \
  --calibration-bins 10 \
  --max-val-samples 512 \
  --max-steps 128 \
  --device cpu
```

## Training Command For User

After implementation passes verification, run this first calibration sweep yourself:

```bash
python scripts/run_predictive_autoresearch.py \
  --base-config configs/aig_lowtc_100k_hard_pretrain.json \
  --objective hard_f1 \
  --max-variants 12 \
  --seeds 2026 \
  --lambda-hards 0.5 \
  --lambda-hard-counts 0.1 \
  --lambda-hard-reductions 0.5 \
  --lambda-hard-ranks 0.0 \
  --lambda-hard-briers 0.0,0.02,0.05,0.1 \
  --lambda-hard-soft-f1s 0.0,0.02 \
  --encoder-types mean \
  --summary-modes global \
  --hard-losses asl \
  --hard-asl-gamma-negs 2.0,3.0,4.0 \
  --hard-asl-clips 0.02,0.05 \
  --hard-head-types residual_context \
  --hard-pos-weight-maxes 20 \
  --hard-negative-sample-ratios 5 \
  --hard-negative-minings topk \
  --train-sample-strategies hard_weighted \
  --feature-modes testability \
  --edge-weight-modes fault_path \
  --edge-keep-ratios 0.6 \
  --lambda-fcs 0.0 \
  --center-lambda-hard 0.5 \
  --center-lambda-hard-count 0.1 \
  --center-lambda-hard-reduction 0.5 \
  --center-lambda-hard-rank 0.0 \
  --center-lambda-hard-brier 0.02 \
  --center-lambda-hard-soft-f1 0.0 \
  --center-edge-keep-ratio 0.6 \
  --stream-logs
```

## Expected Outputs

Implementation outputs:

```text
autoresearch/calibration-loss-smoke-260626/
```

Training outputs:

```text
autoresearch/predictive-<timestamp>/
```

Required report files after training:

```text
results.tsv
summary.md
logs/*.train.log
logs/*.eval.log
```

## Next Command

To execute this plan:

```text
$autoresearch
```
