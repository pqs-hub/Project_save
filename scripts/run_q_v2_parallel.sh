#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_DIR="autoresearch/q-v2-ablation-260630"
mkdir -p "${OUT_DIR}/logs"

run_variant() {
  local gpu="$1"
  local variant="$2"
  local config="$3"
  local log="${OUT_DIR}/logs/${variant}.log"
  local pid_file="${OUT_DIR}/logs/${variant}.pid"

  (
    set +e
    env CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/run_q_v1_train_eval.py \
      --variant "$variant" \
      --config "$config" \
      --out-dir "$OUT_DIR" \
      --promote-label PROMOTE_Q_V2 \
      --report-prefix Q-v2 2>&1
    status="$?"
    echo "[${variant}] exited status=${status}"
    exit "$status"
  ) | stdbuf -oL sed "s/^/[${variant}] /" | tee "$log" &
  echo "$!" > "$pid_file"
}

run_variant 4 q_v2_value_only configs/q_v2_value_only.json
run_variant 5 q_v2_pairwise_only configs/q_v2_pairwise_only.json
run_variant 6 q_v2_pairwise_listwise configs/q_v2_pairwise_listwise.json
run_variant 7 q_v2_pairwise_listwise_hardred configs/q_v2_pairwise_listwise_hardred.json

echo "started q_v2 parallel training with live logs"
echo "logs: ${OUT_DIR}/logs/*.log"

status=0
for pid_file in "${OUT_DIR}"/logs/q_v2_*.pid; do
  pid="$(cat "$pid_file")"
  if ! wait "$pid"; then
    status=1
  fi
done

python scripts/collect_q_v2_results.py --out-dir "$OUT_DIR"
exit "$status"
