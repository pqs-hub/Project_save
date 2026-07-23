#!/usr/bin/env bash
set -euo pipefail

BASE=autoresearch/loop-260720-0945/model_training_round22
variants=(toptype_r21_all toptype_r10_hard)
IFS=',' read -r -a gpus <<< "${GPUS_CSV:-1,4}"
if (( ${#gpus[@]} < ${#variants[@]} )); then
  echo "need at least ${#variants[@]} GPUs" >&2
  exit 2
fi
mkdir -p "$BASE/logs"
python -u scripts/prepare_toptype_round22.py \
  2>&1 | tee "$BASE/logs/prepare_training_extra.log"
pids=(); failed=0
for index in "${!variants[@]}"; do
  variant=${variants[$index]}
  gpu=${gpus[$index]}
  echo "[train-r22-extra] start variant=$variant gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" python -u -m tpi_jepa.train \
    --config "$BASE/configs/$variant.json" \
    2>&1 | sed -u "s|^|[train-r22-extra/$variant] |" | tee "$BASE/logs/$variant.log" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
if (( failed )); then exit 1; fi
echo "[train-r22-extra] complete variants=${variants[*]}"
