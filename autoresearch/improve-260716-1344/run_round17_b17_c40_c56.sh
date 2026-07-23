#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
gpu="${GPU:-7}"
out_root="autoresearch/improve-260716-1344/round17_b17"
mkdir -p "$out_root/console_logs"

run_one() {
  local candidates="$1" tag="context_cluster_c${1}_seed1024_depth4"
  local out="$out_root/$tag"
  echo "[round17] start tag=$tag gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_gmean_sweep.py \
    --eval-protocol configs/eval_protocol_coverage_only.json --protocol-keep-cli-benchmarks \
    --benchmarks iscas99__b17 \
    --checkpoint runs/planner_aligned_q_rank_v5_context_safe/best_final_horizon.pt \
    --planners greedy --score-fields q_pred_context --beam-objectives cumulative \
    --beam-widths 1 --lookahead-depths 1 --max-candidates "$candidates" \
    --discount-gammas 0.9 --candidate-strategies hard_fault_cluster \
    --candidate-diversity-penalties 0.0 --candidate-diversity-depths 4 \
    --candidate-real-fault-priors autoresearch/eval8-real-priors-budget5-v1/real_fault_priors.csv \
    --candidate-allowlist autoresearch/original-netlist-recovery-260712/exact_itc99/b17_C/exact_candidate_nodes.txt \
    --plan-device cuda --time-limit-hours 72 --out-dir "$out" \
    --stream-logs 2>&1 | tee "$out_root/console_logs/$tag.log"
  echo "[round17] done tag=$tag"
}

run_one 40 & p40=$!
run_one 56 & p56=$!
status=0
if ! wait "$p40"; then status=1; fi
if ! wait "$p56"; then status=1; fi
exit "$status"
