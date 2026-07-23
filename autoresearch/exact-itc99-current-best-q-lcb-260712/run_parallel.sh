#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT_ROOT="${OUT_ROOT:-autoresearch/exact-itc99-current-best-q-lcb-260712}"
GPUS_CSV="${GPUS_CSV:-4,7}"
IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
if (( ${#GPUS[@]} == 0 )); then
  echo "GPUS_CSV must contain at least one GPU id" >&2
  exit 2
fi

benchmarks=(
  iscas99__b15_1
  iscas99__b20
  iscas99__b21
  iscas99__b22
  iscas99__b17
)
circuits=(b15_C b20_C b21_C b22_C b17_C)

checkpoint="runs/planner_aligned_q_rank_v1/best_final_horizon.pt"
ensemble="runs/planner_aligned_q_rank_v1/best_final_horizon.pt,runs/planner_aligned_q_rank_v2_safe/best_final_horizon.pt,runs/planner_aligned_q_rank_v2_seed2_safe/best_final_horizon.pt"
protocol="configs/eval_protocol_coverage_only.json"
priors="autoresearch/eval8-real-priors-budget5-v1/real_fault_priors.csv"

mkdir -p "$OUT_ROOT/console_logs"

run_one() {
  local index="$1"
  local gpu="$2"
  local benchmark="${benchmarks[$index]}"
  local circuit="${circuits[$index]}"
  local allowlist="autoresearch/original-netlist-recovery-260712/exact_itc99/${circuit}/exact_candidate_nodes.txt"
  local out_dir="$OUT_ROOT/$circuit"
  local console_log="$OUT_ROOT/console_logs/${circuit}.log"

  echo "[exact-eval] start circuit=$circuit benchmark=$benchmark gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_gmean_sweep.py \
    --eval-protocol "$protocol" \
    --protocol-keep-cli-benchmarks \
    --benchmarks "$benchmark" \
    --checkpoint "$checkpoint" \
    --ensemble-checkpoints "$ensemble" \
    --ensemble-lcb-alpha 0.75 \
    --planners greedy \
    --score-fields q_pred_lcb \
    --beam-objectives cumulative \
    --beam-widths 1 \
    --lookahead-depths 1 \
    --max-candidates 96 \
    --discount-gammas 0.9 \
    --candidate-strategies heuristic_recall_pool \
    --candidate-diversity-penalties 0.0 \
    --candidate-diversity-depths 4 \
    --candidate-allowlist "$allowlist" \
    --candidate-real-fault-priors "$priors" \
    --plan-device cuda \
    --time-limit-hours 72 \
    --out-dir "$out_dir" \
    --stream-logs 2>&1 | tee "$console_log"
  echo "[exact-eval] done circuit=$circuit gpu=$gpu"
}

 pids=()
for ((index=0; index<${#benchmarks[@]}; index++)); do
  slot=$((index % ${#GPUS[@]}))
  run_one "$index" "${GPUS[$slot]}" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "[exact-eval] all five circuits completed"
