#!/usr/bin/env bash
set -euo pipefail

cd /data4/pengqingsong/DFT/TPI-my.3

OUT_DIR="autoresearch/train-version-b-derived-260629-1629/gates/hard"
mkdir -p "${OUT_DIR}/best_runs" "${OUT_DIR}/logs"

declare -A CONFIGS=(
  [incumbent]="autoresearch/highseed-improvement-260626-run-posweight-30/configs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0.json"
  [version_B]="autoresearch/train-version-b-derived-260629-1629/configs/version_B_derived_node_hard.json"
)

declare -A CHECKPOINTS=(
  [incumbent]="autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt"
  [version_B]="autoresearch/train-version-b-derived-260629-1629/runs/version_B_derived_node_hard/best.pt"
)

declare -A DEVICES=(
  [incumbent]="cuda:4"
  [version_B]="cuda:5"
)

variants=(incumbent version_B)
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
