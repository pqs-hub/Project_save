# Calibration-Loss Full Sweep

Run this from the project root:

```bash
bash autoresearch/plan-260626-0803/run_calibration_loss_full_sweep.sh
```

The command streams logs to the terminal, so you can watch progress live.

## Fixed center

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

## Swept terms

```text
lambda_hard_brier in {0.0, 0.02, 0.05, 0.1}
lambda_hard_soft_f1 in {0.0, 0.02}
```

## What to compare

Use the best row in `results.tsv` and compare it against the frozen baseline by:

- `hard_macro_f1_tuned`
- `hard_macro_f1_at_0p5`
- `ece_sa0`
- `ece_sa1`
- `temperature_scaled_ece_sa0`
- `temperature_scaled_ece_sa1`
- `predictive_score`

