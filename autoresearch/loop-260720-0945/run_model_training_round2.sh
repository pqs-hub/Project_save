#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
OUT="autoresearch/loop-260720-0945/model_training_round2"
mkdir -p "$OUT/logs"
python scripts/prepare_conservative_typed_round2.py

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

variants=(typed_balanced typed_long typed_rank)
gpus=(1 2 1)

run_one() {
    local variant="$1" gpu="$2"
    echo "[typed-round2] start variant=$variant gpu=$gpu"
    CUDA_VISIBLE_DEVICES="$gpu" python -u -m tpi_jepa.train \
        --config "$OUT/configs/$variant.json" 2>&1 \
        | sed -u "s|^|[typed-round2/$variant] |" \
        | tee "$OUT/logs/$variant.log"
    echo "[typed-round2] done variant=$variant gpu=$gpu"
}

pids=()
for index in "${!variants[@]}"; do
    run_one "${variants[$index]}" "${gpus[$index]}" &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
done
exit "$failed"
