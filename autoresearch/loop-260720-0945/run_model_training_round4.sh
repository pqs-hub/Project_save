#!/usr/bin/env bash
set -euo pipefail

ROOT=autoresearch/loop-260720-0945/model_training_round4
mkdir -p "$ROOT/logs"
python -u scripts/prepare_onpolicy_round4.py

run_one() {
  local variant=$1 gpu=$2
  echo "[typed-round4] start variant=$variant gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" python -u -m tpi_jepa.train --config "$ROOT/configs/$variant.json" \
    2>&1 | sed -u "s/^/[typed-round4\/$variant] /" | tee "$ROOT/logs/$variant.log"
  echo "[typed-round4] done variant=$variant gpu=$gpu"
}
run_one onpolicy_balanced 1 & p1=$!
run_one onpolicy_marginal 2 & p2=$!
wait "$p1"
wait "$p2"
