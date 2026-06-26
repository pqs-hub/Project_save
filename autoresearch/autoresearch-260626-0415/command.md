# Calibration-Loss AutoResearch Command

Run this from the project root:

```bash
bash autoresearch/autoresearch-260626-0415/run_calibration_loss_sweep.sh
```

This command prints progress because it uses:

```text
--stream-logs
```

## Purpose

Run the first training sweep after the calibration-loss implementation.

Frozen mainline:

```text
encoder_type=mean
summary_mode=global
lambda_hard_rank=0.0
hard_loss=asl
hard_head_type=residual_context
hard_negative_mining=topk
train_sample_strategy=hard_weighted
feature_mode=testability
edge_weight_mode=fault_path
edge_keep_ratio=0.6
```

Explored variables:

```text
lambda_hard_brier in {0.0, 0.02, 0.05, 0.1}
lambda_hard_soft_f1 in {0.0, 0.02}
hard_asl_gamma_neg in {2.0, 3.0, 4.0}
hard_asl_clip in {0.02, 0.05}
```

Primary metric:

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

Expected output:

```text
autoresearch/predictive-<timestamp>/
```
