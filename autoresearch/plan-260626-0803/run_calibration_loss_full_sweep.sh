#!/usr/bin/env bash
set -euo pipefail

python scripts/run_predictive_autoresearch.py \
  --base-config configs/aig_lowtc_100k_hard_pretrain.json \
  --objective hard_f1 \
  --max-variants 8 \
  --seeds 2026 \
  --lambda-hards 0.5 \
  --lambda-hard-counts 0.1 \
  --lambda-hard-reductions 0.5 \
  --lambda-hard-ranks 0.0 \
  --lambda-hard-briers 0.0,0.02,0.05,0.1 \
  --lambda-hard-soft-f1s 0.0,0.02 \
  --encoder-types mean \
  --summary-modes global \
  --hard-losses asl \
  --hard-asl-gamma-negs 2.0 \
  --hard-asl-clips 0.05 \
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
  --center-lambda-hard-rank 0.0 \
  --center-lambda-hard-brier 0.02 \
  --center-lambda-hard-soft-f1 0.0 \
  --center-edge-keep-ratio 0.6 \
  --stream-logs
