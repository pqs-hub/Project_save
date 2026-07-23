#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
OUT_ROOT="${OUT_ROOT:-autoresearch/improve-260716-1344/round3_b17}"
mkdir -p "$OUT_ROOT/console_logs"

checkpoint="runs/planner_aligned_q_rank_v5_context_safe/best_final_horizon.pt"
protocol="configs/eval_protocol_coverage_only.json"
allowlist="autoresearch/original-netlist-recovery-260712/exact_itc99/b17_C/exact_candidate_nodes.txt"
priors="autoresearch/eval8-real-priors-budget5-v1/real_fault_priors.csv"
tags=(context_recall_d002 context_cluster_c48 context_recall_c192)
strategies=(heuristic_recall_pool hard_fault_cluster heuristic_recall_pool)
diversities=(0.02 0.0 0.0)
max_candidates=(96 48 192)
gpus=(4 6 7)

run_one() {
  local i="$1" tag="${tags[$1]}" strategy="${strategies[$1]}"
  local diversity="${diversities[$1]}" candidates="${max_candidates[$1]}" gpu="${gpus[$1]}"
  echo "[round3] start tag=$tag gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_gmean_sweep.py \
    --eval-protocol "$protocol" --protocol-keep-cli-benchmarks \
    --benchmarks iscas99__b17 --checkpoint "$checkpoint" \
    --planners greedy --score-fields q_pred_context --beam-objectives cumulative \
    --beam-widths 1 --lookahead-depths 1 --max-candidates "$candidates" \
    --discount-gammas 0.9 --candidate-strategies "$strategy" \
    --candidate-diversity-penalties "$diversity" --candidate-diversity-depths 4 \
    --candidate-allowlist "$allowlist" --candidate-real-fault-priors "$priors" \
    --plan-device cuda --time-limit-hours 72 --out-dir "$OUT_ROOT/$tag" \
    --stream-logs 2>&1 | tee "$OUT_ROOT/console_logs/$tag.log"
  echo "[round3] done tag=$tag gpu=$gpu"
}

pids=()
for i in "${!tags[@]}"; do run_one "$i" & pids+=("$!"); done
failed=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
if (( failed )); then exit 1; fi
echo "[round3] all b17 variants completed"
