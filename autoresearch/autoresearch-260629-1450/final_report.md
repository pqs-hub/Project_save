# Oracle Ranking Scratch Sweep Report

## Verdict

No checkpoint is promoted.

Main reason: oracle ranking scratch checkpoints did not improve expanded oracle validation and all scratch checkpoints severely regressed hard-fault predictive quality versus the incumbent.

## Hybrid Score Summary

| checkpoint | expanded Spearman | expanded neg top1 | expanded regret | transfer Spearman | transfer neg top1 | transfer regret | hard F1 tuned | predictive score | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `incumbent` | 0.031476 | 0.375000 | 0.035483 | 0.327398 | 0.166667 | 0.012552 | 0.794805 | 0.820147 | `BASELINE` |
| `scratch_0p00` | 0.008166 | 0.500000 | 0.033701 | -0.059028 | 0.500000 | 0.020397 | 0.165301 | 0.486414 | `REJECT` |
| `scratch_0p05` | -0.072460 | 0.520833 | 0.029891 | -0.095552 | 0.500000 | 0.021918 | 0.213846 | 0.449052 | `REJECT` |
| `scratch_0p10` | -0.080071 | 0.562500 | 0.031281 | 0.099302 | 0.333333 | 0.010468 | 0.160915 | 0.497586 | `REJECT` |
| `scratch_0p20` | 0.013980 | 0.416667 | 0.024583 | 0.402667 | 0.166667 | 0.012670 | 0.187658 | 0.499720 | `REJECT` |

## Decision Details

- `scratch_0p00`: `REJECT` - expanded hybrid Spearman did not beat incumbent by +0.02; expanded negative_top1 worse than incumbent; transfer hybrid Spearman dropped more than 0.02; transfer negative_top1 worse than incumbent; hard_macro_f1_tuned regressed by more than 0.03
- `scratch_0p05`: `REJECT` - expanded hybrid Spearman did not beat incumbent by +0.02; expanded negative_top1 worse than incumbent; transfer hybrid Spearman dropped more than 0.02; transfer negative_top1 worse than incumbent; hard_macro_f1_tuned regressed by more than 0.03
- `scratch_0p10`: `REJECT` - expanded hybrid Spearman did not beat incumbent by +0.02; expanded negative_top1 worse than incumbent; transfer hybrid Spearman dropped more than 0.02; transfer negative_top1 worse than incumbent; hard_macro_f1_tuned regressed by more than 0.03
- `scratch_0p20`: `REJECT` - expanded hybrid Spearman did not beat incumbent by +0.02; expanded negative_top1 worse than incumbent; hard_macro_f1_tuned regressed by more than 0.03

## Notable Observations

- `scratch_0p20` has the best transfer Spearman among the sweep (`hybrid_pred` 0.402667), but it fails expanded validation and hard-fault gates.
- `scratch_0p20` also improves expanded top1 regret versus incumbent (0.024583 vs 0.035483), but its expanded Spearman is lower and negative_top1 is worse.
- `scratch_0p00` control already regresses transfer and hard metrics, so this scratch recipe is not a drop-in replacement for incumbent highseed training.
- The oracle ranking loss is active and measurable in training history, but the current scratch setup damages the original hard-fault objective too much.

## Artifacts

- `autoresearch/autoresearch-260629-1450/manifest.json`
- `autoresearch/autoresearch-260629-1450/promotion_summary.tsv`
- `autoresearch/autoresearch-260629-1450/gates/expanded_val/oracle_action_value_summary.tsv`
- `autoresearch/autoresearch-260629-1450/gates/transfer/oracle_action_value_summary.tsv`
- `autoresearch/autoresearch-260629-1450/gates/hard`
- `autoresearch/autoresearch-260629-1450/run_sweep.sh`
- `autoresearch/autoresearch-260629-1450/run_hard_gate.sh`
