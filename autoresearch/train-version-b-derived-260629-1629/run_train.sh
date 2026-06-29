#!/usr/bin/env bash
set -euo pipefail

cd /data4/pengqingsong/DFT/TPI-my.3

OUT_DIR="autoresearch/train-version-b-derived-260629-1629"
mkdir -p "${OUT_DIR}/logs"

python -u -m tpi_jepa.train \
  --config "${OUT_DIR}/configs/version_B_derived_node_hard.json" \
  >"${OUT_DIR}/logs/version_B_derived_node_hard.log" 2>&1
