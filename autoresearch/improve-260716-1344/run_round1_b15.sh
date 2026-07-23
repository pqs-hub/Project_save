#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT_ROOT="${OUT_ROOT:-autoresearch/improve-260716-1344/round1_b15}"
mkdir -p "$OUT_ROOT/console_logs"

checkpoint="runs/planner_aligned_q_rank_v1/best_final_horizon.pt"
ensemble="runs/planner_aligned_q_rank_v1/best_final_horizon.pt,runs/planner_aligned_q_rank_v2_safe/best_final_horizon.pt,runs/planner_aligned_q_rank_v2_seed2_safe/best_final_horizon.pt"
protocol="configs/eval_protocol_coverage_only.json"
priors="autoresearch/eval8-real-priors-budget5-v1/real_fault_priors.csv"
allowlist="autoresearch/original-netlist-recovery-260712/exact_itc99/b15_C/exact_candidate_nodes.txt"

variants=(matched_hfc_d002 hfc_d000 union_d002 recall_d002)
strategies=(hard_fault_cone hard_fault_cone hard_fault_recall_union heuristic_recall_pool)
diversities=(0.02 0.0 0.02 0.02)
gpus=(4 5 6 7)

run_one() {
  local index="$1"
  local variant="${variants[$index]}"
  local strategy="${strategies[$index]}"
  local diversity="${diversities[$index]}"
  local gpu="${gpus[$index]}"
  local out_dir="$OUT_ROOT/$variant"
  local log="$OUT_ROOT/console_logs/$variant.log"

  echo "[round1] start variant=$variant strategy=$strategy diversity=$diversity gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_gmean_sweep.py \
    --eval-protocol "$protocol" \
    --protocol-keep-cli-benchmarks \
    --benchmarks iscas99__b15_1 \
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
    --candidate-strategies "$strategy" \
    --candidate-diversity-penalties "$diversity" \
    --candidate-diversity-depths 4 \
    --candidate-allowlist "$allowlist" \
    --candidate-real-fault-priors "$priors" \
    --plan-device cuda \
    --time-limit-hours 72 \
    --out-dir "$out_dir" \
    --stream-logs 2>&1 | tee "$log"
  echo "[round1] done variant=$variant gpu=$gpu"
}

pids=()
for index in "${!variants[@]}"; do
  run_one "$index" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if (( failed )); then
  echo "[round1] at least one variant failed" >&2
  exit 1
fi
echo "[round1] all b15 variants completed"
