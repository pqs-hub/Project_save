#!/usr/bin/env bash
set -euo pipefail

BASE=autoresearch/loop-260720-0945/model_training_round10
mkdir -p "$BASE/logs"
python -u scripts/prepare_return_rank_round10.py
variants=(return_within_lr5e5 return_within_lr1e4 return_dual_lr5e5)
IFS=',' read -r -a gpus <<< "${GPUS_CSV:-0,1,2}"
MAX_PARALLEL=${MAX_PARALLEL:-${#gpus[@]}}
if (( ${#gpus[@]} == 0 || MAX_PARALLEL < 1 )); then
  echo "GPUS_CSV must contain at least one GPU and MAX_PARALLEL must be positive" >&2
  exit 2
fi

run_one() {
  local variant=$1 gpu=$2
  echo "[train-r10] start variant=$variant gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" python -u -m tpi_jepa.train --config "$BASE/configs/$variant.json" \
    2>&1 | sed -u "s|^|[train-r10/$variant] |" | tee "$BASE/logs/$variant.log"
}

pids=(); failed=0; job=0
wait_batch() {
  local pid
  for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
  pids=()
}
for variant in "${variants[@]}"; do
  gpu=${gpus[$((job % ${#gpus[@]}))]}
  run_one "$variant" "$gpu" &
  pids+=("$!")
  job=$((job+1))
  if (( ${#pids[@]} == MAX_PARALLEL )); then wait_batch; fi
done
if (( ${#pids[@]} )); then wait_batch; fi
if (( failed )); then exit 1; fi
echo "[train-r10] complete variants=${variants[*]}"
