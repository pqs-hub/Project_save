# AutoResearch Plan: Calibration Loss Full Coverage

generated_at: `2026-06-26 08:03 Asia/Shanghai`

## Goal

```text
在固定当前最优 ASL 配置的前提下，完整比较 hard calibration loss
（lambda_hard_brier / lambda_hard_soft_f1）是否能带来实质 hard F1 增益。
```

## Why this plan

上一轮 sweep 已经说明：

- `hard_asl_gamma_neg=2.0` 和 `hard_asl_clip=0.05` 是当前更稳的中心点。
- `lambda_hard_brier` 只在很小子集上被看到了。
- `lambda_hard_soft_f1` 没有真正进入有效比较范围，原因是 `--max-variants 12` 截断了笛卡尔积。

因此下一轮不再扩大搜索面，而是固定已验证的 ASL 中心，只完整扫 calibration loss。

## Scope

Only runtime configuration, no code changes:

```text
autoresearch/autoresearch-260626-0803/run_calibration_loss_full_sweep.sh
docs/hard_f1_autoresearch_report.md
autoresearch/predictive-260626-041643/
```

Out of scope:

```text
tpi_jepa/train.py 结构修改
scripts/evaluate_hard_checkpoints.py 结构修改
scripts/run_predictive_autoresearch.py 结构修改
gate_dir / cone / rank 重新扫
多视图
action-ranking loss
```

## Metric

Primary:

```text
hard_macro_f1_tuned
```

Guardrails:

```text
hard_macro_f1_at_0p5
ece_sa0
ece_sa1
temperature_scaled_ece_sa0
temperature_scaled_ece_sa1
predictive_score
hard_recall_at_top_10pct
```

## Verify

Before running the sweep:

```bash
bash -n autoresearch/autoresearch-260626-0803/run_calibration_loss_full_sweep.sh
python scripts/run_predictive_autoresearch.py --help
```

## Planned Run

Run command:

```bash
bash autoresearch/autoresearch-260626-0803/run_calibration_loss_full_sweep.sh
```

This command keeps progress visible because it uses `--stream-logs`.

### Grid

Frozen center:

```text
encoder_type=mean
summary_mode=global
lambda_hard_rank=0.0
hard_loss=asl
hard_asl_gamma_neg=2.0
hard_asl_clip=0.05
hard_head_type=residual_context
hard_negative_mining=topk
train_sample_strategy=hard_weighted
feature_mode=testability
edge_weight_mode=fault_path
edge_keep_ratio=0.6
lambda_fc=0.0
```

Swept variables:

```text
lambda_hard_brier in {0.0, 0.02, 0.05, 0.1}
lambda_hard_soft_f1 in {0.0, 0.02}
```

This yields 8 variants and fully covers the calibration-loss comparison.

## Success Criteria

Promote calibration loss only if one of these holds:

```text
hard_macro_f1_tuned improves by >= 0.03 over the frozen mainline
OR
mean ECE drops by >= 20% with hard_macro_f1_tuned loss <= 0.01
```

Reject if:

```text
predictive_score drops by > 0.03
or hard_recall_at_top_10pct drops by > 0.05
or tuned F1 improves only while hard_macro_f1_at_0p5 remains unusable
```

## Expected Output

```text
autoresearch/predictive-<timestamp>/
```

## Next Step

After the run, inspect `results.tsv` and `summary.md` for:

1. Whether `lambda_hard_brier` helps beyond the frozen baseline.
2. Whether `lambda_hard_soft_f1` helps once it is actually covered.
3. Whether calibration improves without hurting ranking / reduction guardrails.
