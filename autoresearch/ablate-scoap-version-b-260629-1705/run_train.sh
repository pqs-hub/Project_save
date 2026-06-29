#!/usr/bin/env bash
set -euo pipefail

cd /data4/pengqingsong/DFT/TPI-my.3

OUT_DIR="autoresearch/ablate-scoap-version-b-260629-1705"
mkdir -p "${OUT_DIR}/logs"

variants=(
  B_only_scoap
  B_only_delta_scoap
)

pids=()
for variant in "${variants[@]}"; do
  python -u -m tpi_jepa.train \
    --config "${OUT_DIR}/configs/${variant}.json" \
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
