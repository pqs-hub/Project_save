#!/usr/bin/env bash
set -euo pipefail

ROOT=autoresearch/loop-260720-0945/model_training_round3
mkdir -p "$ROOT/logs"
python -u scripts/prepare_long_real_round3.py

run_one() {
  local variant=$1
  local gpu=$2
  echo "[typed-round3] start variant=$variant gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" python -u -m tpi_jepa.train \
    --config "$ROOT/configs/$variant.json" \
    2>&1 | sed -u "s/^/[typed-round3\/$variant] /" | tee "$ROOT/logs/$variant.log"
  echo "[typed-round3] done variant=$variant gpu=$gpu"
}

run_one long32_balanced 1 &
pid_balanced=$!
run_one long32_return 2 &
pid_return=$!
wait "$pid_balanced"
wait "$pid_return"
run_one long32_marginal 1
