#!/usr/bin/env bash
set -euo pipefail

cd /data4/pengqingsong/DFT/TPI-my.3

OUT_DIR="autoresearch/ablate-scoap-version-b-260629-1705/gates/hard"
mkdir -p "${OUT_DIR}/best_runs" "${OUT_DIR}/logs"

declare -A CONFIGS=(
  [B_base]="autoresearch/train-version-b-derived-260629-1629/configs/version_B_derived_node_hard.json"
  [B_only_scoap]="autoresearch/ablate-scoap-version-b-260629-1705/configs/B_only_scoap.json"
  [B_only_delta_scoap]="autoresearch/ablate-scoap-version-b-260629-1705/configs/B_only_delta_scoap.json"
)

declare -A CHECKPOINTS=(
  [B_base]="autoresearch/train-version-b-derived-260629-1629/runs/version_B_derived_node_hard/best.pt"
  [B_only_scoap]="autoresearch/ablate-scoap-version-b-260629-1705/runs/B_only_scoap/best.pt"
  [B_only_delta_scoap]="autoresearch/ablate-scoap-version-b-260629-1705/runs/B_only_delta_scoap/best.pt"
)

declare -A DEVICES=(
  [B_base]="cuda:4"
  [B_only_scoap]="cuda:5"
  [B_only_delta_scoap]="cuda:7"
)

variants=(B_base B_only_scoap B_only_delta_scoap)
pids=()
for variant in "${variants[@]}"; do
  best_run="${OUT_DIR}/best_runs/${variant}"
  mkdir -p "${best_run}"
  ln -sfn "/data4/pengqingsong/DFT/TPI-my.3/${CHECKPOINTS[$variant]}" "${best_run}/best.pt"
  python -u scripts/evaluate_hard_checkpoints.py \
    --config "${CONFIGS[$variant]}" \
    --run-dir "${best_run}" \
    --out-csv "${OUT_DIR}/${variant}.csv" \
    --max-val-samples 512 \
    --max-steps 256 \
    --device "${DEVICES[$variant]}" \
    >"${OUT_DIR}/logs/${variant}.log" 2>&1 &
  pids+=("$!")
  echo "${variant} pid=${pids[-1]}"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

exit "${status}"
