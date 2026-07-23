#!/usr/bin/env bash
set -euo pipefail

ROUND=autoresearch/loop-260720-0945/model_training_round3
ROOT="$ROUND/long_real_labels_v2"
SOURCE="$ROUND/structural32_current/labels.csv"
mkdir -p "$ROOT"

python -u scripts/build_structural_rollout_labels.py \
  --num-benchmarks 24 \
  --trajectories-per-benchmark 1 \
  --trajectory-length 32 \
  --pool-multiplier 4 \
  --max-nodes 4000 \
  --seed 260720 \
  --out "$SOURCE"

python -u scripts/relabel_sequences_with_backend.py \
  --labels "$SOURCE" \
  --out-dir "$ROOT" \
  --backend atalanta-bist \
  --patterns 100000 \
  --seed 2026 \
  --parallel-jobs 12 \
  --max-sequences 24 \
  --max-steps 32 \
  --timeout-sec 14400 \
  --resume \
  --cleanup-workdir \
  --drop-partial-sequences \
  2>&1 | tee "$ROOT/driver.log"
