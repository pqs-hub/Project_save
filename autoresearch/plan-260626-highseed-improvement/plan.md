# AutoResearch Plan: High-Seed Improvement

generated_at: `2026-06-26 Asia/Shanghai`

## Goal

在 `autoresearch/predictive-260626-102849` 的高分方法基础上继续提高 hard-fault predictive pretraining 的主指标：

```text
hard_macro_f1_tuned
```

已知强结果：

```text
autoresearch/predictive-260626-102849
best_variant = seed2027 center
hard_macro_f1_tuned = 0.789471903661045
predictive_score    = 0.7817596175233479
```

同一中心配置在 seed 上方差较大：

| Seed | hard_macro_f1_tuned | predictive_score | Note |
|---|---:|---:|---|
| 2026 | 0.475027 | 0.634467 | weak |
| 2027 | 0.789472 | 0.781760 | best hard F1 |
| 2028 | 0.757122 | 0.804906 | strong predictive |
| 2029 | 0.469855 | 0.620107 | weak |
| 2030 | 0.788691 | 0.819110 | best predictive |

因此下一轮不采用单 seed 决策，而是在高分 seeds 上做 family 级比较。

## Prior Evidence From Ablation

`autoresearch/core-component-ablation-260626-run-*` 给出的单 seed 消融结论：

| Factor | Evidence |
|---|---|
| `residual_context` | 必要，换 `mlp` 下降约 0.093 |
| `hard_weighted` sampling | 必要，换 `shuffle` 下降约 0.147 |
| top-k negative mining | 明显优于 random；相对 mixed 证据弱 |
| `edge_keep_ratio=0.6` sparse edge | 必要，full fault-path edge 下降约 0.110 |
| ASL vs focal | focal 在单 seed 上 hard F1 更高，但 predictive/top10 明显下降 |

本轮保留强支持组件：

```text
residual_context
hard_weighted
fault_path
edge_keep_ratio=0.6
```

重点搜索可能提升高分中心的窄变量：

```text
hard_loss / ASL focal tradeoff
ASL gamma/clip
lambda_hard_soft_f1
topk vs mixed
negative ratio / pos weight
```

## Frozen Base

以高分中心为基准：

```text
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

Primary seeds:

```text
2027,2028,2030
```

These are the seeds where the incumbent method already shows high hard F1 and strong predictive score. If a candidate cannot improve on these seeds, it is unlikely to improve the current best method.

## Candidate Families

Each family runs the same value over seeds `2027,2028,2030`.

| Family | Changed Factor | Value | Why |
|---|---|---|---|
| `baseline-highseeds` | none | incumbent center | Sanity rerun under current server/script state |
| `focal-guardrail` | hard loss | `focal` | Test if focal's F1 lift survives high seeds and guardrails |
| `mixed-mining` | negative mining | `mixed` | Ablation showed mixed nearly matches top-k with good predictive |
| `asl-gamma-1p5` | ASL gamma neg | `1.5` | Slightly less negative focusing may improve calibrated F1 |
| `asl-gamma-2p5` | ASL gamma neg | `2.5` | Check whether center gamma 2.0 is under-focused |
| `asl-clip-0p03` | ASL clip | `0.03` | Less clipping may sharpen tuned threshold behavior |
| `softf1-0p04` | lambda_hard_soft_f1 | `0.04` | Directly push macro F1 with modest extra soft-F1 pressure |
| `softf1-0p06` | lambda_hard_soft_f1 | `0.06` | Stronger F1 pressure; reject if predictive/top10 drops |
| `neg-ratio-3` | negative sample ratio | `3` | Reduce negative dominance while keeping top-k mining |
| `posweight-30` | positive weight max | `30` | Test if sparse hard positives need stronger weighting |

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

## Decision Rules

Use family-level comparison against `baseline-highseeds`.

Promote a candidate if:

```text
mean(hard_macro_f1_tuned) improves by >= 0.02
AND best-seed hard_macro_f1_tuned exceeds 0.789471903661045
AND predictive_score mean does not drop by more than 0.03
AND hard_recall_at_top_10pct mean does not drop by more than 0.10
```

Mark as high-F1 risky if:

```text
hard_macro_f1_tuned improves
BUT predictive_score or top10 guardrails collapse
```

Reject if:

```text
mean(hard_macro_f1_tuned) drops by >= 0.02
OR no seed beats 0.789471903661045
```

If `focal-guardrail` improves hard F1 but collapses top10 again, run a follow-up hybrid plan rather than replacing ASL directly.

## Scope

No model code changes. Runtime config only.

Files:

```text
autoresearch/plan-260626-highseed-improvement/plan.md
autoresearch/plan-260626-highseed-improvement/run_highseed_improvement.sh
autoresearch/highseed-improvement-260626-run-*/
```

## Verify

Before running:

```bash
bash -n autoresearch/plan-260626-highseed-improvement/run_highseed_improvement.sh
DRY_RUN=1 bash autoresearch/plan-260626-highseed-improvement/run_highseed_improvement.sh
```

After running:

```bash
find autoresearch -maxdepth 2 -path 'autoresearch/highseed-improvement-260626-run-*/*results.tsv' -print
```

Then compare family means and best seed against the incumbent.

## Run Command

```bash
bash autoresearch/plan-260626-highseed-improvement/run_highseed_improvement.sh
```

The script defaults to:

```text
CUDA_VISIBLE_DEVICES=4,5,6,7
MAX_PARALLEL=4
SEEDS=2027,2028,2030
CACHE_SAMPLES=true
sample_cache_max_entries=25000
```

Override examples:

```bash
SEEDS=2027,2028,2030,2031 MAX_PARALLEL=4 bash autoresearch/plan-260626-highseed-improvement/run_highseed_improvement.sh
```
