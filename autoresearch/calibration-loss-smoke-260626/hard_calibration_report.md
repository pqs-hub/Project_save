# Hard Calibration Temperature Report

## Scope

- No model weights were changed.
- Temperatures are fit independently for SA0 and SA1 on validation logits.
- This report measures calibration headroom before training-time Brier/soft-F1 changes.

## Headline

- checkpoint_count: `6`
- best_epoch: `2`
- best_hard_macro_f1_tuned: `0.26655997154796834`
- mean_raw_ece_sa0: `0.182724`
- mean_scaled_ece_sa0: `0.085107`
- mean_ece_drop_sa0: `0.534234`
- mean_raw_ece_sa1: `0.457790`
- mean_scaled_ece_sa1: `0.442401`
- mean_ece_drop_sa1: `0.033616`
- mean_raw_f1_at_0p5: `0.038129`
- mean_scaled_f1_at_0p5: `0.038129`

## Per-Checkpoint Temperature

| epoch | class | temperature | raw ECE | scaled ECE | raw F1@0.5 | scaled F1@0.5 |
|---:|---|---:|---:|---:|---:|---:|
| `1` | `sa0` | `0.500` | `0.3705` | `0.2949` | `0.0012` | `0.0012` |
| `1` | `sa1` | `4.000` | `0.5790` | `0.5153` | `0.0325` | `0.0325` |
| `2` | `sa0` | `0.500` | `0.1855` | `0.0639` | `0.0299` | `0.0299` |
| `2` | `sa1` | `4.000` | `0.5250` | `0.4999` | `0.0390` | `0.0390` |
| `3` | `sa0` | `0.500` | `0.1058` | `0.0232` | `0.0290` | `0.0290` |
| `3` | `sa1` | `1.500` | `0.4211` | `0.4426` | `0.0550` | `0.0550` |
| `4` | `sa0` | `0.500` | `0.1245` | `0.0324` | `0.0257` | `0.0257` |
| `4` | `sa1` | `1.000` | `0.3483` | `0.3483` | `0.0754` | `0.0754` |
| `-2` | `sa0` | `0.500` | `0.1855` | `0.0639` | `0.0299` | `0.0299` |
| `-2` | `sa1` | `4.000` | `0.5250` | `0.4999` | `0.0390` | `0.0390` |
| `-1` | `sa0` | `0.500` | `0.1245` | `0.0324` | `0.0257` | `0.0257` |
| `-1` | `sa1` | `1.000` | `0.3483` | `0.3483` | `0.0754` | `0.0754` |

## Recommendation

- Temperature scaling shows meaningful calibration headroom; train-time calibration loss is justified.
