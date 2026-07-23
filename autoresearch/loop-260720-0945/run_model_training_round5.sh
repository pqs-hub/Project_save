#!/usr/bin/env bash
set -euo pipefail

ROOT=autoresearch/loop-260720-0945/model_training_round5
python scripts/prepare_counterfactual_round5.py
mkdir -p "$ROOT/logs"

export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
variants=(prefix_rank prefix_rank_sa prefix_cql_sa prefix_toplist_sa)
IFS=',' read -r -a gpus <<< "${GPUS_CSV:-1,2,3,4}"
MAX_PARALLEL=${MAX_PARALLEL:-${#gpus[@]}}
if (( ${#gpus[@]} == 0 || MAX_PARALLEL < 1 )); then
  echo "GPUS_CSV must contain at least one GPU and MAX_PARALLEL must be positive" >&2
  exit 2
fi
pids=()
failed=0
job=0
wait_batch() {
  local pid
  for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
  pids=()
}
for index in "${!variants[@]}"; do
  variant=${variants[$index]}
  gpu=${gpus[$((job % ${#gpus[@]}))]}
  echo "[train-r5] start variant=$variant gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" python -u -m tpi_jepa.train \
    --config "$ROOT/configs/$variant.json" \
    2>&1 | sed -u "s|^|[train-r5/$variant] |" | tee "$ROOT/logs/$variant.log" &
  pids+=("$!")
  job=$((job + 1))
  if (( ${#pids[@]} == MAX_PARALLEL )); then wait_batch; fi
done
if (( ${#pids[@]} )); then wait_batch; fi
if (( failed )); then exit 1; fi
echo "[train-r5] complete variants=${variants[*]}"
