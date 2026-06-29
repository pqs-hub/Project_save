# Autoresearch Results: Main Training Oracle Ranking Loss

## Implemented

Added default-off oracle pairwise ranking supervision to `tpi_jepa/train.py`.

The main training loop now supports:

- `oracle_actions`
- `lambda_oracle_rank`
- `lambda_oracle_value` reserved, currently not used by main loss
- `oracle_ranking_score_field`
- `oracle_every_n_steps`
- `oracle_batch_groups`
- `oracle_warmup_epochs`
- `oracle_ramp_epochs`
- `oracle_pairwise_min_delta`
- `oracle_pairwise_temperature`
- `oracle_max_actions_per_group`

When `oracle_actions` is present and `lambda_oracle_rank > 0`, training loads backend-labeled oracle action groups and runs an auxiliary optimizer step every `oracle_every_n_steps`.

## Loss

The original training loss is unchanged.

Oracle ranking is added as a separate auxiliary step:

```text
oracle_loss = current_lambda_oracle_rank * pairwise_rank_loss(score, oracle_delta_tc)
```

The first implementation uses `hybrid_pred` by default:

```text
hybrid_pred = reward_pred + return_pred + hard_reduction_total_pred * coverage_scale
```

`current_lambda_oracle_rank` supports warmup/ramp:

```text
epoch <= oracle_warmup_epochs: 0
then linearly ramp over oracle_ramp_epochs
```

## Verification

Python compile check passed:

```bash
python -m py_compile tpi_jepa/train.py scripts/finetune_oracle_action_values.py scripts/evaluate_oracle_action_values.py
```

Default training smoke passed:

```bash
python -m tpi_jepa.train --config configs/aig_lowtc_100k_hard_pretrain.json --max-steps 1
```

Observed:

```text
[train] warning: CUDA requested but unavailable; using CPU
[train] disabling pattern loss
[train] rows=100000 train_rows=85510 val_rows=14490 train_samples=20000 val_samples=4096 rollout_training=False feature_dim=38 relation_dim=12 cache_samples=False sample_cache_max_entries=unlimited
[train] epoch=1 horizon=1 train_loss=2.034580 val_loss=1.847104 train_steps=1
```

Oracle branch smoke passed:

```bash
python -m tpi_jepa.train --config autoresearch/autoresearch-260629-1446/configs/oracle_rank_smoke.json --max-steps 1
```

Observed:

```text
[train] disabling pattern loss
[train] oracle_ranking enabled groups=288 score_field=hybrid_pred lambda_oracle_rank=0.05
[train] rows=100000 train_rows=85510 val_rows=14490 train_samples=8 val_samples=8 rollout_training=False feature_dim=38 relation_dim=12 cache_samples=False sample_cache_max_entries=unlimited
[train] epoch=1 horizon=1 train_loss=1.150908 val_loss=1.392162 train_steps=1
```

The oracle smoke `history.csv` includes:

```text
train_oracle_loss=0.034709
train_oracle_rank_loss=0.694182
train_oracle_pairs=83
train_oracle_groups=1
train_oracle_weight=0.05
train_oracle_steps=1
```

This confirms the oracle ranking step actually ran and contributed gradients.

## Artifacts

- `configs/oracle_rank_smoke.json`
- `sweep_weights.json`
- `results.md`
- `handoff.json`

## Next Step

Generate full run configs for:

```text
scratch_oracle_rank_0p00
scratch_oracle_rank_0p05
scratch_oracle_rank_0p10
scratch_oracle_rank_0p20
```

Then launch training and compare with fixed oracle gates:

- expanded labeled-subckt validation oracle gate
- transfer oracle gate
- original hard-fault evaluation
