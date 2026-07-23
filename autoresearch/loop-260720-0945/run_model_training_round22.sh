#!/usr/bin/env bash
set -euo pipefail

BASE=autoresearch/loop-260720-0945/model_training_round22
VARIANT=toptype_r21_hard
GPU=${GPU:-7}
mkdir -p "$BASE/logs"
python -u scripts/prepare_toptype_round22.py \
  2>&1 | tee "$BASE/logs/prepare_training.log"
echo "[train-r22] start variant=$VARIANT gpu=$GPU"
CUDA_VISIBLE_DEVICES="$GPU" python -u -m tpi_jepa.train \
  --config "$BASE/configs/$VARIANT.json" \
  2>&1 | sed -u "s|^|[train-r22/$VARIANT] |" | tee "$BASE/logs/$VARIANT.log"
echo "[train-r22] complete variant=$VARIANT"
