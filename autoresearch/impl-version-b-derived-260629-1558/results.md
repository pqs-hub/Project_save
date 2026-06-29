# Version B Derived Hard-Value Implementation

## Implemented

Implemented the first part of Version B: derived hard-count and hard-reduction scores from node-level hard logits.

The model now computes:

```text
derived_hard_count_pre_pred
derived_hard_count_post_pred
derived_hard_reduction_pred
```

The count vector order is:

```text
[total, sa0, sa1]
```

The reduction is:

```text
(pre_count - post_count) / max(1, pre_count)
```

## Files Changed

- `tpi_jepa/model.py`
- `tpi_jepa/plan.py`
- `tpi_jepa/train.py`
- `scripts/oracle_action_value_probe.py`
- `scripts/finetune_oracle_action_values.py`
- `scripts/evaluate_hard_checkpoints.py`

## New Planner / Oracle Fields

Added planner output fields:

```text
derived_hard_count_pre_total_pred
derived_hard_count_pre_sa0_pred
derived_hard_count_pre_sa1_pred
derived_hard_count_post_total_pred
derived_hard_count_post_sa0_pred
derived_hard_count_post_sa1_pred
derived_hard_reduction_total_pred
derived_hard_reduction_sa0_pred
derived_hard_reduction_sa1_pred
derived_hard_reduction_hybrid_pred
```

Added oracle score fields:

```text
derived_hard_reduction_total_pred
derived_hard_reduction_hybrid_pred
```

## New Hard Evaluator Metrics

Added:

```text
derived_hard_count_post_mae
derived_hard_reduction_mae
derived_hard_reduction_acc_at_005
derived_hard_reduction_sign_acc
derived_hard_reduction_score
```

Version B should be judged by these derived metrics, not by the old `hard_reduction_head` metrics.

## Verification

Compile check passed:

```bash
python -m py_compile tpi_jepa/model.py tpi_jepa/plan.py tpi_jepa/train.py scripts/oracle_action_value_probe.py scripts/evaluate_oracle_action_values.py scripts/finetune_oracle_action_values.py scripts/evaluate_hard_checkpoints.py
```

Oracle gate smoke passed:

```bash
python scripts/evaluate_oracle_action_values.py \
  --oracle-actions autoresearch/oracle-action-probe-260629-expanded-subckt-val/oracle_actions.tsv \
  --checkpoint incumbent=.../epoch_002.pt \
  --score-fields derived_hard_reduction_total_pred,derived_hard_reduction_hybrid_pred \
  --out-dir autoresearch/impl-version-b-derived-260629-1558/oracle_smoke
```

Smoke result on incumbent:

```text
derived_hard_reduction_total_pred mean_spearman = 0.086327
derived_hard_reduction_hybrid_pred mean_spearman = 0.086327
negative_top1_rate = 0.479167
mean_top1_regret = 0.019714
```

Hard evaluator smoke passed:

```bash
python scripts/evaluate_hard_checkpoints.py \
  --config ...seed2030...json \
  --run-dir autoresearch/impl-version-b-derived-260629-1558/hard_smoke_run \
  --out-csv autoresearch/impl-version-b-derived-260629-1558/hard_smoke.csv \
  --max-val-samples 8 \
  --max-steps 4 \
  --device cpu
```

Smoke derived metrics:

```text
derived_hard_count_post_mae = 0.058843
derived_hard_reduction_mae = 0.429698
derived_hard_reduction_acc_at_005 = 0.166667
derived_hard_reduction_sign_acc = 0.500000
derived_hard_reduction_score = 0.570302
```

## Notes

This implementation does not delete old heads.

Old heads remain for checkpoint compatibility:

```text
hard_count_head
hard_reduction_head
reward_head
return_head
```

Version B training should disable their direct losses:

```text
lambda_hard_count = 0.0
lambda_hard_reduction = 0.0
lambda_fc = 0.0
lambda_return = 0.0
```

Then planner/oracle should use:

```text
derived_hard_reduction_total_pred
derived_hard_reduction_hybrid_pred
```

## Next Step

Generate and train Version B config:

```text
hard_value_mode = derived_from_node_hard
lambda_hard_count = 0.0
lambda_hard_reduction = 0.0
lambda_fc = 0.0
lambda_return = 0.0
lambda_oracle_rank = 0.0
```

Then evaluate with:

```text
hard gate: derived_hard_reduction_score
expanded oracle gate: derived_hard_reduction_total_pred / derived_hard_reduction_hybrid_pred
transfer oracle gate: derived_hard_reduction_total_pred / derived_hard_reduction_hybrid_pred
```
