# Parallel Accuracy Variants

These variants keep the model architecture and planner-facing configuration
stable, then sweep training target weights and hard-label calibration.

| Variant | GPU | Config | Intent |
|---|---:|---|---|
| `v1_balanced` | 4 | `configs/mainline_accuracy_improve_v1.json` | Balanced first pass: reward, return, hard reduction, hard calibration |
| `v2_reward_return` | 5 | `configs/mainline_accuracy_improve_v2_reward_return.json` | Emphasize reward/return sign accuracy |
| `v3_hard_precision` | 6 | `configs/mainline_accuracy_improve_v3_hard_precision.json` | Emphasize hard-node precision/F1 while keeping recall |
| `v4_reduction_sign` | 7 | `configs/mainline_accuracy_improve_v4_reduction_sign.json` | Emphasize hard-reduction sign accuracy |

Run:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 bash autoresearch/improve-260704-1035/run_accuracy_variants_parallel.sh
```

Outputs:

- per-job train/eval live output and logs under
  `autoresearch/improve-260704-1035/parallel_accuracy_variants/logs/`
- per-variant metrics under
  `autoresearch/improve-260704-1035/parallel_accuracy_variants/accuracy/<variant>/`
- combined summary:
  `autoresearch/improve-260704-1035/parallel_accuracy_variants/accuracy_summary.tsv`
