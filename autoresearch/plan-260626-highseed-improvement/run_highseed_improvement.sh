#!/usr/bin/env bash
set -euo pipefail

BASE_CONFIG="configs/aig_lowtc_100k_hard_pretrain.json"
ROOT_OUT="autoresearch/highseed-improvement-260626-run"
SEEDS="${SEEDS:-2027,2028,2030}"
EXTRA_ARGS=()

export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.cache/matplotlib}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
MAX_PARALLEL="${MAX_PARALLEL:-${#GPU_IDS[@]}}"
BATCH_PIDS=()
BATCH_NAMES=()
FAMILY_INDEX=0
FAIL=0

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  ROOT_OUT="autoresearch/highseed-improvement-260626-dryrun"
  EXTRA_ARGS+=(--dry-run)
fi

wait_batch() {
  local pid
  local name
  local idx

  for idx in "${!BATCH_PIDS[@]}"; do
    pid="${BATCH_PIDS[$idx]}"
    name="${BATCH_NAMES[$idx]}"
    if wait "${pid}"; then
      echo "[highseed-improvement] finished ${name}"
    else
      echo "[highseed-improvement] failed ${name}" >&2
      FAIL=1
    fi
  done
  BATCH_PIDS=()
  BATCH_NAMES=()
  if [[ "${FAIL}" != "0" ]]; then
    exit "${FAIL}"
  fi
}

run_family() {
  local name="$1"
  shift
  local gpu="${GPU_IDS[$((FAMILY_INDEX % ${#GPU_IDS[@]}))]}"
  FAMILY_INDEX=$((FAMILY_INDEX + 1))

  echo "[highseed-improvement] launching ${name} on CUDA_VISIBLE_DEVICES=${gpu} seeds=${SEEDS}"

  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    python scripts/run_predictive_autoresearch.py \
      --base-config "${BASE_CONFIG}" \
      --objective hard_f1 \
      --max-variants 3 \
      --out-dir "${ROOT_OUT}-${name}" \
      --seeds "${SEEDS}" \
      --lambda-hards 0.5 \
      --lambda-hard-counts 0.1 \
      --lambda-hard-reductions 0.5 \
      --lambda-hard-ranks 0.0 \
      --lambda-hard-briers 0.0 \
      --lambda-hard-soft-f1s 0.02 \
      --encoder-types mean \
      --summary-modes global \
      --hard-losses asl \
      --hard-asl-gamma-negs 2.0 \
      --hard-asl-clips 0.05 \
      --hard-head-types residual_context \
      --hard-pos-weight-maxes 20 \
      --hard-negative-sample-ratios 5 \
      --hard-negative-minings topk \
      --train-sample-strategies hard_weighted \
      --feature-modes testability \
      --edge-weight-modes fault_path \
      --edge-keep-ratios 0.6 \
      --lambda-fcs 0.0 \
      --center-lambda-hard 0.5 \
      --center-lambda-hard-count 0.1 \
      --center-lambda-hard-reduction 0.5 \
      --center-lambda-hard-rank 0.0 \
      --center-lambda-hard-brier 0.0 \
      --center-lambda-hard-soft-f1 0.02 \
      --center-edge-keep-ratio 0.6 \
      --cache-samples \
      --sample-cache-max-entries 25000 \
      --stream-logs \
      "${EXTRA_ARGS[@]}" \
      "$@"
  ) &
  BATCH_PIDS+=("$!")
  BATCH_NAMES+=("${name}")

  if [[ "${#BATCH_PIDS[@]}" -ge "${MAX_PARALLEL}" ]]; then
    wait_batch
  fi
}

run_family "baseline-highseeds"

run_family "focal-guardrail" \
  --hard-losses focal

run_family "mixed-mining" \
  --hard-negative-minings mixed

run_family "asl-gamma-1p5" \
  --hard-asl-gamma-negs 1.5

run_family "asl-gamma-2p5" \
  --hard-asl-gamma-negs 2.5

run_family "asl-clip-0p03" \
  --hard-asl-clips 0.03

run_family "softf1-0p04" \
  --lambda-hard-soft-f1s 0.04 \
  --center-lambda-hard-soft-f1 0.04

run_family "softf1-0p06" \
  --lambda-hard-soft-f1s 0.06 \
  --center-lambda-hard-soft-f1 0.06

run_family "neg-ratio-3" \
  --hard-negative-sample-ratios 3

run_family "posweight-30" \
  --hard-pos-weight-maxes 30

wait_batch
