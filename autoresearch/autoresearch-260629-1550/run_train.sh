#!/usr/bin/env bash
set -euo pipefail

cd /data4/pengqingsong/DFT/TPI-my.3

OUT_DIR="autoresearch/autoresearch-260629-1550"
mkdir -p "${OUT_DIR}/logs"

variants=(
  control_incumbent_like
  version_A_no_hard_count
)

pids=()
for variant in "${variants[@]}"; do
  config="${OUT_DIR}/configs/${variant}.json"
  log="${OUT_DIR}/logs/${variant}.log"
  python -u -m tpi_jepa.train --config "${config}" >"${log}" 2>&1 &
  pids+=("$!")
  echo "${variant} pid=${pids[-1]} log=${log}"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

exit "${status}"
