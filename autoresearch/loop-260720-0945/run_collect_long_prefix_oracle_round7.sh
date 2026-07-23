#!/usr/bin/env bash
set -euo pipefail

ROOT=autoresearch/loop-260720-0945/model_training_round7
OUT="$ROOT/long_prefix_oracle"
PLANS="$ROOT/long_onpolicy_plans"
TRAINING_MANIFEST=autoresearch/loop-260720-0945/model_training_round3/structural32_current/manifest.json
mkdir -p "$OUT"

extra_args=()
if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then extra_args+=(--prepare-only); fi

python -u scripts/collect_onpolicy_prefix_oracle.py \
  --plans-dir "$PLANS" \
  --training-manifest "$TRAINING_MANIFEST" \
  --eval-protocol configs/eval_protocol_coverage_only.json \
  --prefix-steps 32,48,64,96,127 \
  --candidate-strategy hard_fault_cluster \
  --candidate-pool-size 64 \
  --actions-per-prefix 9 \
  --backend atalanta-bist \
  --patterns 300000 \
  --seed 2026 \
  --parallel-jobs 12 \
  --timeout-sec 14400 \
  --out-dir "$OUT" \
  --resume \
  --cleanup-workdir \
  "${extra_args[@]}" \
  2>&1 | tee "$OUT/driver.log"
