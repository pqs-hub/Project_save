#!/usr/bin/env bash
set -euo pipefail

cd /data4/pengqingsong/DFT/TPI-my.3

OUT_DIR="autoresearch/autoresearch-260629-1450/gates/hard"
mkdir -p "${OUT_DIR}/best_runs" "${OUT_DIR}/logs"

declare -A CONFIGS=(
  [incumbent]="autoresearch/highseed-improvement-260626-run-posweight-30/configs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0.json"
  [scratch_0p00]="autoresearch/autoresearch-260629-1450/configs/scratch_oracle_rank_0p00.json"
  [scratch_0p05]="autoresearch/autoresearch-260629-1450/configs/scratch_oracle_rank_0p05.json"
  [scratch_0p10]="autoresearch/autoresearch-260629-1450/configs/scratch_oracle_rank_0p10.json"
  [scratch_0p20]="autoresearch/autoresearch-260629-1450/configs/scratch_oracle_rank_0p20.json"
)

declare -A CHECKPOINTS=(
  [incumbent]="autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt"
  [scratch_0p00]="autoresearch/autoresearch-260629-1450/runs/scratch_oracle_rank_0p00/best.pt"
  [scratch_0p05]="autoresearch/autoresearch-260629-1450/runs/scratch_oracle_rank_0p05/best.pt"
  [scratch_0p10]="autoresearch/autoresearch-260629-1450/runs/scratch_oracle_rank_0p10/best.pt"
  [scratch_0p20]="autoresearch/autoresearch-260629-1450/runs/scratch_oracle_rank_0p20/best.pt"
)

declare -A DEVICES=(
  [incumbent]="cuda:4"
  [scratch_0p00]="cuda:5"
  [scratch_0p05]="cuda:6"
  [scratch_0p10]="cuda:7"
  [scratch_0p20]="cuda:4"
)

variants=(incumbent scratch_0p00 scratch_0p05 scratch_0p10 scratch_0p20)
pids=()
for variant in "${variants[@]}"; do
  best_run="${OUT_DIR}/best_runs/${variant}"
  mkdir -p "${best_run}"
  ln -sfn "/data4/pengqingsong/DFT/TPI-my.3/${CHECKPOINTS[$variant]}" "${best_run}/best.pt"
  python scripts/evaluate_hard_checkpoints.py \
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
