#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_DIR="autoresearch/rollout-loss-ablation-260630"
mkdir -p "${OUT_DIR}/logs" "${OUT_DIR}/hard_eval"

run_variant() {
  local gpu="$1"
  local variant="$2"
  local config="$3"
  local log="${OUT_DIR}/logs/${variant}.log"
  local pid_file="${OUT_DIR}/logs/${variant}.pid"

  (
    set +e
    echo "[${variant}] gpu=${gpu} config=${config}"
    env CUDA_VISIBLE_DEVICES="$gpu" python -u -m tpi_jepa.train --config "$config"
    train_status="$?"
    if [ "$train_status" -ne 0 ]; then
      echo "[${variant}] train failed status=${train_status}"
      exit "$train_status"
    fi
    env CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/evaluate_hard_checkpoints.py \
      --config "$config" \
      --run-dir "runs/${variant}" \
      --out-csv "${OUT_DIR}/hard_eval/${variant}.csv" \
      --out-png "${OUT_DIR}/hard_eval/${variant}.png" \
      --max-val-samples 2048 \
      --max-steps 512 \
      --device cuda
    eval_status="$?"
    echo "[${variant}] exited status=${eval_status}"
    exit "$eval_status"
  ) | stdbuf -oL sed "s/^/[${variant}] /" | tee "$log" &
  echo "$!" > "$pid_file"
}

run_variant 0 rollout_loss_A_reward_only configs/rollout_loss_A_reward_only.json
run_variant 1 rollout_loss_B_tiny_return configs/rollout_loss_B_tiny_return.json
run_variant 2 rollout_loss_C_delta_scoap configs/rollout_loss_C_delta_scoap.json
run_variant 3 rollout_loss_D_recommended configs/rollout_loss_D_recommended.json

echo "started rollout-loss ablation with live logs"
echo "logs: ${OUT_DIR}/logs/*.log"

status=0
for pid_file in "${OUT_DIR}"/logs/rollout_loss_*.pid; do
  pid="$(cat "$pid_file")"
  if ! wait "$pid"; then
    status=1
  fi
done

python scripts/collect_rollout_loss_results.py --out-dir "$OUT_DIR"
exit "$status"
