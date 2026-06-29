#!/usr/bin/env bash
set -euo pipefail

cd /data4/pengqingsong/DFT/TPI-my.3

OUT_DIR="autoresearch/ablate-scoap-version-a-260629-1806/gates/hard"
mkdir -p "${OUT_DIR}/best_runs" "${OUT_DIR}/logs"

declare -A CONFIGS=(
  [A_base]="autoresearch/autoresearch-260629-1550/configs/version_A_no_hard_count.json"
  [A_only_scoap]="autoresearch/ablate-scoap-version-a-260629-1806/configs/A_only_scoap.json"
  [A_only_delta_scoap]="autoresearch/ablate-scoap-version-a-260629-1806/configs/A_only_delta_scoap.json"
)

declare -A CHECKPOINTS=(
  [A_base]="autoresearch/autoresearch-260629-1550/runs/version_A_no_hard_count/best.pt"
  [A_only_scoap]="autoresearch/ablate-scoap-version-a-260629-1806/runs/A_only_scoap/best.pt"
  [A_only_delta_scoap]="autoresearch/ablate-scoap-version-a-260629-1806/runs/A_only_delta_scoap/best.pt"
)

declare -A DEVICES=(
  [A_base]="cuda:4"
  [A_only_scoap]="cuda:6"
  [A_only_delta_scoap]="cuda:7"
)

variants=(A_base A_only_scoap A_only_delta_scoap)
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
