#!/usr/bin/env bash
set -euo pipefail

cd /data4/pengqingsong/DFT/TPI-my.3

OUT_DIR="autoresearch/autoresearch-260629-1450"
mkdir -p "${OUT_DIR}/logs"

variants=(
  scratch_oracle_rank_0p00
  scratch_oracle_rank_0p05
  scratch_oracle_rank_0p10
  scratch_oracle_rank_0p20
)

pids=()
for variant in "${variants[@]}"; do
  config="${OUT_DIR}/configs/${variant}.json"
  log="${OUT_DIR}/logs/${variant}.log"
  python -m tpi_jepa.train --config "${config}" >"${log}" 2>&1 &
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
