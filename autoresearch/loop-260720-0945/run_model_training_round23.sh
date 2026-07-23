#!/usr/bin/env bash
set -euo pipefail

BASE=autoresearch/loop-260720-0945/model_training_round23
VARIANT=late_hard_fixed
GPU=${GPU:-7}
mkdir -p "$BASE/logs"
python -u scripts/prepare_late_horizon_round23.py \
  2>&1 | tee "$BASE/logs/prepare_training.log"
echo "[train-r23] start variant=$VARIANT gpu=$GPU"
CUDA_VISIBLE_DEVICES="$GPU" python -u -m tpi_jepa.train \
  --config "$BASE/configs/$VARIANT.json" \
  2>&1 | sed -u "s|^|[train-r23/$VARIANT] |" | tee "$BASE/logs/$VARIANT.log"
echo "[train-r23] complete variant=$VARIANT"
