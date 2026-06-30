#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p autoresearch/q-v1-parallel-260630/logs

run_variant() {
  local gpu="$1"
  local variant="$2"
  local config="$3"
  local log="autoresearch/q-v1-parallel-260630/logs/${variant}.log"
  local pid_file="autoresearch/q-v1-parallel-260630/logs/${variant}.pid"

  (
    set +e
    env CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/run_q_v1_train_eval.py \
      --variant "$variant" \
      --config "$config" \
      --out-dir autoresearch/q-v1-parallel-260630 2>&1
    status="$?"
    echo "[${variant}] exited status=${status}"
    exit "$status"
  ) | stdbuf -oL sed "s/^/[${variant}] /" | tee "$log" &
  echo "$!" > "$pid_file"
}

run_variant 4 q_v1_hard_topk configs/q_v1_candidate_rank.json
run_variant 5 q_v1_all_pairwise configs/q_v1_all_pairwise.json
run_variant 6 q_v1_listwise_only configs/q_v1_listwise_only.json
run_variant 7 q_v1_factorized_hard_topk configs/q_v1_factorized_hard_topk.json

echo "started q_v1 parallel training with live logs"
echo "logs: autoresearch/q-v1-parallel-260630/logs/*.log"

status=0
for pid_file in autoresearch/q-v1-parallel-260630/logs/q_v1_*.pid; do
  pid="$(cat "$pid_file")"
  if ! wait "$pid"; then
    status=1
  fi
done

exit "$status"
