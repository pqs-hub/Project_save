#!/usr/bin/env bash
set -euo pipefail

ROOT=autoresearch/loop-260720-0945/model_training_round9
OUT="$ROOT/late_prefix_oracle"
PLANS="$ROOT/late_onpolicy_plans"
MANIFEST="$ROOT/late_source_manifest.json"
mkdir -p "$OUT"

extra_args=()
if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then extra_args+=(--prepare-only); fi

python -u scripts/collect_onpolicy_prefix_oracle.py \
  --plans-dir "$PLANS" \
  --training-manifest "$MANIFEST" \
  --eval-protocol configs/eval_protocol_coverage_only.json \
  --prefix-steps 144,176,208,240,255 \
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
