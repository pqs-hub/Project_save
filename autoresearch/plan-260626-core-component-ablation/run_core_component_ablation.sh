#!/usr/bin/env bash
set -euo pipefail

BASE_CONFIG="configs/aig_lowtc_100k_hard_pretrain.json"
ROOT_OUT="autoresearch/core-component-ablation-260626-run"
EXTRA_ARGS=()

export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.cache/matplotlib}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
MAX_PARALLEL="${MAX_PARALLEL:-${#GPU_IDS[@]}}"
BATCH_PIDS=()
BATCH_NAMES=()
VARIANT_INDEX=0
FAIL=0

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  ROOT_OUT="autoresearch/core-component-ablation-260626-dryrun"
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
      echo "[core-ablation] finished ${name}"
    else
      echo "[core-ablation] failed ${name}" >&2
      FAIL=1
    fi
  done
  BATCH_PIDS=()
  BATCH_NAMES=()
  if [[ "${FAIL}" != "0" ]]; then
    exit "${FAIL}"
  fi
}

run_variant() {
  local name="$1"
  shift
  local gpu="${GPU_IDS[$((VARIANT_INDEX % ${#GPU_IDS[@]}))]}"
  VARIANT_INDEX=$((VARIANT_INDEX + 1))

  echo "[core-ablation] launching ${name} on CUDA_VISIBLE_DEVICES=${gpu}"

  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    python scripts/run_predictive_autoresearch.py \
      --base-config "${BASE_CONFIG}" \
      --objective hard_f1 \
      --max-variants 1 \
      --out-dir "${ROOT_OUT}-${name}" \
      --seeds 2026 \
      --lambda-hards 0.5 \
      --lambda-hard-counts 0.1 \
      --lambda-hard-reductions 0.5 \
      --lambda-hard-ranks 0.0 \
      --lambda-hard-briers 0.0 \
      --lambda-hard-soft-f1s 0.02 \
      --encoder-types mean \
      --summary-modes global \
      --hard-asl-gamma-negs 2.0 \
      --hard-asl-clips 0.05 \
      --hard-pos-weight-maxes 20 \
      --hard-negative-sample-ratios 5 \
      --feature-modes testability \
      --lambda-fcs 0.0 \
      --center-lambda-hard 0.5 \
      --center-lambda-hard-count 0.1 \
      --center-lambda-hard-reduction 0.5 \
      --center-lambda-hard-rank 0.0 \
      --center-lambda-hard-brier 0.0 \
      --center-lambda-hard-soft-f1 0.02 \
      --center-edge-keep-ratio 0.6 \
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

run_variant "full-center" \
  --hard-losses asl \
  --hard-head-types residual_context \
  --hard-negative-minings topk \
  --train-sample-strategies hard_weighted \
  --edge-weight-modes fault_path \
  --edge-keep-ratios 0.6

run_variant "loss-focal" \
  --hard-losses focal \
  --hard-head-types residual_context \
  --hard-negative-minings topk \
  --train-sample-strategies hard_weighted \
  --edge-weight-modes fault_path \
  --edge-keep-ratios 0.6

run_variant "loss-bce" \
  --hard-losses bce \
  --hard-head-types residual_context \
  --hard-negative-minings topk \
  --train-sample-strategies hard_weighted \
  --edge-weight-modes fault_path \
  --edge-keep-ratios 0.6

run_variant "head-mlp" \
  --hard-losses asl \
  --hard-head-types mlp \
  --hard-negative-minings topk \
  --train-sample-strategies hard_weighted \
  --edge-weight-modes fault_path \
  --edge-keep-ratios 0.6

run_variant "mining-mixed" \
  --hard-losses asl \
  --hard-head-types residual_context \
  --hard-negative-minings mixed \
  --train-sample-strategies hard_weighted \
  --edge-weight-modes fault_path \
  --edge-keep-ratios 0.6

run_variant "mining-random" \
  --hard-losses asl \
  --hard-head-types residual_context \
  --hard-negative-minings random \
  --train-sample-strategies hard_weighted \
  --edge-weight-modes fault_path \
  --edge-keep-ratios 0.6

run_variant "sampling-shuffle" \
  --hard-losses asl \
  --hard-head-types residual_context \
  --hard-negative-minings topk \
  --train-sample-strategies shuffle \
  --edge-weight-modes fault_path \
  --edge-keep-ratios 0.6

run_variant "edge-mean" \
  --hard-losses asl \
  --hard-head-types residual_context \
  --hard-negative-minings topk \
  --train-sample-strategies hard_weighted \
  --edge-weight-modes mean \
  --edge-keep-ratios 1.0

run_variant "edge-fault-path-full" \
  --hard-losses asl \
  --hard-head-types residual_context \
  --hard-negative-minings topk \
  --train-sample-strategies hard_weighted \
  --edge-weight-modes fault_path \
  --edge-keep-ratios 1.0

wait_batch
