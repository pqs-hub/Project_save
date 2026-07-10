#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="autoresearch/improve-260704-1035"
CONFIG="configs/mainline_accuracy_improve_v1.json"
RUN_DIR="runs/mainline_accuracy_improve_v1"

mkdir -p "${OUT_DIR}/logs"

echo "[$(date -Is)] training ${CONFIG}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python -m tpi_jepa.train --config "${CONFIG}" 2>&1 | tee "${OUT_DIR}/logs/train_accuracy_improve_v1.log"

echo "[$(date -Is)] evaluating trained-head accuracy"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python scripts/evaluate_trained_head_accuracy.py \
  --checkpoint "${RUN_DIR}/best.pt" \
  --config "${CONFIG}" \
  --max-samples 4096 \
  --device cuda \
  --require-cuda \
  --out-dir "${OUT_DIR}/accuracy_v1" 2>&1 | tee "${OUT_DIR}/logs/eval_accuracy_improve_v1.log"
