# AutoResearch Plan: Core Component Ablation

generated_at: `2026-06-26 Asia/Shanghai`

## Goal

验证当前 hard-fault 主线中 5 个核心组件的独立贡献：

```text
ASL loss
residual_context hard head
top-k negative mining
hard_weighted sampling
fault_path edge weighting
```

当前高分结果只能说明这些组件组成的组合有效，不能证明每个组件单独有效。因此本轮采用 one-factor ablation：固定中心配置，每次只替换一个组件。

## Frozen Center

```text
seed=2026
lambda_hard=0.5
lambda_hard_count=0.1
lambda_hard_reduction=0.5
lambda_hard_rank=0.0
lambda_hard_brier=0.0
lambda_hard_soft_f1=0.02
encoder_type=mean
summary_mode=global
hard_loss=asl
hard_asl_gamma_neg=2.0
hard_asl_clip=0.05
hard_head_type=residual_context
hard_pos_weight_max=20
hard_negative_sample_ratio=5
hard_negative_mining=topk
train_sample_strategy=hard_weighted
feature_mode=testability
edge_weight_mode=fault_path
edge_keep_ratio=0.6
lambda_fc=0.0
```

Center result to beat/reference:

```text
autoresearch/predictive-260626-092536
hard_macro_f1_tuned = 0.551942
predictive_score    = 0.670407
```

## Ablation Matrix

| Variant | Changed Factor | Value | Purpose |
|---|---|---|---|
| `full_center` | none | center | Reference run under the exact current center |
| `loss_focal` | hard loss | `focal` | Test whether ASL is better than focal |
| `loss_bce` | hard loss | `bce` | Test whether ASL is better than plain BCE |
| `head_mlp` | hard head | `mlp` | Test whether residual/context head helps |
| `mining_mixed` | negative mining | `mixed` | Test whether top-k is better than mixed mining |
| `mining_random` | negative mining | `random` | Test whether top-k is better than random negatives |
| `sampling_shuffle` | train sampling | `shuffle` | Test whether hard_weighted sampling helps |
| `edge_mean` | edge weighting | `mean`, `edge_keep_ratio=1.0` | Test whether fault_path weighting helps |
| `edge_fault_path_full` | edge sparsity | `fault_path`, `edge_keep_ratio=1.0` | Separate fault_path weighting from top-edge pruning |

Note: `edge_weight_mode=mean` only supports `edge_keep_ratio=1.0` in the current runner, so `edge_mean` changes both edge weighting and pruning. `edge_fault_path_full` is included to isolate whether the effect is from fault-path weighting or from `edge_keep_ratio=0.6`.

## Metric

Primary:

```text
hard_macro_f1_tuned
```

Guardrails:

```text
predictive_score
hard_macro_f1_at_0p5
hard_recall_at_top_10pct
hard_reduction_score
hard_count_top10_overlap
ece_sa0
ece_sa1
temperature_scaled_ece_sa0
temperature_scaled_ece_sa1
```

## Promote / Reject Rules

For each component, compare the ablated variant with `full_center`.

Promote the component as independently useful if:

```text
full_center hard_macro_f1_tuned - ablated hard_macro_f1_tuned >= 0.03
AND predictive_score does not improve enough to contradict the hard-F1 conclusion
```

Mark as weak / inconclusive if:

```text
absolute F1 delta < 0.03
```

Reject or downgrade the component if:

```text
ablated variant improves hard_macro_f1_tuned by >= 0.03
OR full_center only wins tuned F1 while F1@0.5 / ECE collapse badly
```

## Scope

No code changes. Runtime config only.

Files:

```text
autoresearch/plan-260626-core-component-ablation/plan.md
autoresearch/plan-260626-core-component-ablation/run_core_component_ablation.sh
autoresearch/core-component-ablation-260626-run-*/
docs/hard_f1_autoresearch_report.md
```

## Verify

Before running:

```bash
bash -n autoresearch/plan-260626-core-component-ablation/run_core_component_ablation.sh
python scripts/run_predictive_autoresearch.py --help
```

After running:

```bash
find autoresearch -maxdepth 2 -path 'autoresearch/core-component-ablation-260626-run-*/*results.tsv' -print
```

Then compare each ablation result against `full_center`.

## Run Command

```bash
bash autoresearch/plan-260626-core-component-ablation/run_core_component_ablation.sh
```

The script uses `--stream-logs` for every run, so train/eval progress is visible.

## Expected Decision

This run should replace the current over-strong wording:

```text
ASL / residual_context / top-k / hard_weighted / fault_path are proven effective
```

with one of:

```text
independently supported
weak but retained as center
rejected / downgraded
```

for each component.
