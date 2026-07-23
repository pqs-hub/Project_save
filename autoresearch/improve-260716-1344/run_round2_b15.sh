#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT_ROOT="${OUT_ROOT:-autoresearch/improve-260716-1344/round2_b15}"
mkdir -p "$OUT_ROOT/console_logs"

protocol="configs/eval_protocol_coverage_only.json"
allowlist="autoresearch/original-netlist-recovery-260712/exact_itc99/b15_C/exact_candidate_nodes.txt"
priors="autoresearch/eval8-real-priors-budget5-v1/real_fault_priors.csv"
q_checkpoint="runs/planner_aligned_q_rank_v1/best_final_horizon.pt"
q_ensemble="runs/planner_aligned_q_rank_v1/best_final_horizon.pt,runs/planner_aligned_q_rank_v2_safe/best_final_horizon.pt,runs/planner_aligned_q_rank_v2_seed2_safe/best_final_horizon.pt"
reward_checkpoint="runs/rollout_loss_A_reward_only/epoch_009.pt"

run_sweep() {
  local tag="$1" gpu="$2" checkpoint="$3" score="$4" planner="$5" beam="$6" depth="$7" priors_arg="$8"
  local out_dir="$OUT_ROOT/$tag"
  local log="$OUT_ROOT/console_logs/$tag.log"
  local extra=()
  if [[ "$tag" == q_lcb_cluster_c48 ]]; then
    extra+=(--ensemble-checkpoints "$q_ensemble" --ensemble-lcb-alpha 0.75)
  fi
  if [[ "$priors_arg" == yes ]]; then
    extra+=(--candidate-real-fault-priors "$priors")
  fi
  echo "[round2] start tag=$tag gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_gmean_sweep.py \
    --eval-protocol "$protocol" --protocol-keep-cli-benchmarks \
    --benchmarks iscas99__b15_1 --checkpoint "$checkpoint" \
    --planners "$planner" --score-fields "$score" --beam-objectives cumulative \
    --beam-widths "$beam" --lookahead-depths "$depth" --max-candidates 48 \
    --discount-gammas 0.9 --candidate-strategies hard_fault_cluster \
    --candidate-diversity-penalties 0.0 --candidate-diversity-depths 4 \
    --candidate-allowlist "$allowlist" --plan-device cuda --time-limit-hours 72 \
    --out-dir "$out_dir" --stream-logs "${extra[@]}" 2>&1 | tee "$log"
  echo "[round2] done tag=$tag gpu=$gpu"
}

run_heuristic() {
  local out_dir="$OUT_ROOT/heuristic_cluster_c48"
  local plan="$out_dir/iscas99__b15_1.csv"
  local log="$OUT_ROOT/console_logs/heuristic_cluster_c48.log"
  mkdir -p "$out_dir"
  {
    python scripts/plan_candidate_baseline.py \
      --benchmark-id iscas99__b15_1 --budget 278 --max-candidates 48 \
      --iterative-first --candidate-strategy hard_fault_cluster \
      --candidate-allowlist "$allowlist" --out "$plan"
    python -m tpi_jepa.evaluate_plan_tmax \
      --benchmark-id iscas99__b15_1 --plan-csv "$plan" \
      --out-dir "$out_dir/eval" --patterns 300000 --seed 2026 \
      --backend atalanta-bist --timeout-sec 14400 --eval-step-mode final --force
  } 2>&1 | tee "$log"
}

run_sweep q_lcb_cluster_c48 4 "$q_checkpoint" q_pred_lcb greedy 1 1 yes & p1=$!
run_sweep reward_cluster_c48 5 "$reward_checkpoint" reward_pred greedy 1 1 no & p2=$!
run_sweep reward_beam2_cluster_c48 6 "$reward_checkpoint" reward_pred beam 2 2 no & p3=$!
run_heuristic & p4=$!

failed=0
for pid in "$p1" "$p2" "$p3" "$p4"; do
  if ! wait "$pid"; then failed=1; fi
done
if (( failed )); then exit 1; fi
echo "[round2] all b15 cluster variants completed"
