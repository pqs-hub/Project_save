# PRD: Clean Gate/Cone/Rank Ablation

## Problem

The previous new-technology route bundled multiple changes:

```text
gate_dir encoder + cone summary + hard ranking loss
```

It underperformed the baseline, but the result does not identify which component helped or hurt.

## Goal

Run a clean factorial ablation to decide whether to keep, revise, or drop each component.

## Fixed Baseline

```text
ASL
residual_context hard head
topk mining
hard_weighted sampling
lambda_hard=0.5
lambda_hard_count=0.10
lambda_hard_reduction=0.5
edge_keep_ratio=0.6
```

## Ablation Matrix

```text
encoder_type in {mean, gate_dir}
summary_mode in {global, cone}
lambda_hard_rank in {0.0, 0.03, 0.05}
```

Total:

```text
12 variants
```

## Command

```bash
python scripts/run_predictive_autoresearch.py \
  --base-config configs/aig_lowtc_100k_hard_pretrain.json \
  --objective hard_f1 \
  --max-variants 12 \
  --out-dir autoresearch/predictive-tech-ablation-260626 \
  --lambda-hards 0.5 \
  --lambda-hard-counts 0.1 \
  --lambda-hard-reductions 0.5 \
  --lambda-hard-ranks 0.0,0.03,0.05 \
  --encoder-types mean,gate_dir \
  --summary-modes global,cone \
  --hard-losses asl \
  --hard-head-types residual_context \
  --hard-pos-weight-maxes 20 \
  --hard-negative-sample-ratios 5 \
  --hard-negative-minings topk \
  --train-sample-strategies hard_weighted \
  --feature-modes testability \
  --edge-weight-modes fault_path \
  --edge-keep-ratios 0.6 \
  --lambda-fcs 0.0 \
  --center-lambda-hard 0.5 \
  --center-lambda-hard-count 0.1 \
  --center-lambda-hard-reduction 0.5 \
  --center-lambda-hard-rank 0.03 \
  --center-edge-keep-ratio 0.6 \
  --stream-logs
```

## Acceptance Criteria

Keep a component only if:

```text
matched-seed hard F1 improves by >= 0.03
or predictive score improves by >= 0.03
with no large drop in hard recall top10
```

## Decision Rules

- If `mean + global + rank` helps, ranking is useful independent of new encoder.
- If `mean + cone` helps, action cone pooling is useful and cheap.
- If `gate_dir + global` helps, gate-direction encoding is useful but cone/rank interaction was bad.
- If only `gate_dir + cone + rank` helps, keep as candidate but require multi-seed validation.
- If none help, freeze GNN architecture and move to action ranking/calibration.
