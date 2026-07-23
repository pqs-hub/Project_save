#!/usr/bin/env bash
set -euo pipefail

BASE=autoresearch/loop-260720-0945/model_training_round20
mkdir -p "$BASE/logs"
python -u scripts/prepare_listwise_return_round20.py \
  2>&1 | tee "$BASE/logs/prepare_training.log"

IFS=',' read -r -a variants <<< "${VARIANTS_CSV:-return_pairwise_expanded,return_hybrid_listwise,return_top_listwise}"
IFS=',' read -r -a gpus <<< "${GPUS_CSV:-1,2,4}"
if (( ${#gpus[@]} < ${#variants[@]} )); then
  echo "need at least ${#variants[@]} GPUs in GPUS_CSV" >&2
  exit 2
fi

pids=(); failed=0
for index in "${!variants[@]}"; do
  variant=${variants[$index]}
  gpu=${gpus[$index]}
  echo "[train-r20] start variant=$variant gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" python -u -m tpi_jepa.train \
    --config "$BASE/configs/$variant.json" \
    2>&1 | sed -u "s|^|[train-r20/$variant] |" | tee "$BASE/logs/$variant.log" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
if (( failed )); then exit 1; fi
echo "[train-r20] complete variants=${variants[*]}"
