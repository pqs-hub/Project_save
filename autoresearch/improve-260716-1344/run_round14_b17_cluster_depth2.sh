#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
gpu="${GPU:-7}"
out="${OUT_DIR:-autoresearch/improve-260716-1344/round14_b17/context_cluster_c48_seed1024_depth2}"
mkdir -p "$(dirname "$out")/console_logs"

echo "[round14] start tag=context_cluster_c48_seed1024_depth2 gpu=$gpu"
CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_gmean_sweep.py \
  --eval-protocol configs/eval_protocol_coverage_only.json --protocol-keep-cli-benchmarks \
  --benchmarks iscas99__b17 \
  --checkpoint runs/planner_aligned_q_rank_v5_context_safe/best_final_horizon.pt \
  --planners greedy --score-fields q_pred_context --beam-objectives cumulative \
  --beam-widths 1 --lookahead-depths 1 --max-candidates 48 \
  --discount-gammas 0.9 --candidate-strategies hard_fault_cluster \
  --candidate-diversity-penalties 0.0 --candidate-diversity-depths 4 \
  --candidate-real-fault-priors autoresearch/eval8-real-priors-budget5-v1/real_fault_priors.csv \
  --candidate-allowlist autoresearch/original-netlist-recovery-260712/exact_itc99/b17_C/exact_candidate_nodes.txt \
  --plan-device cuda --time-limit-hours 72 --out-dir "$out" \
  --stream-logs 2>&1 | tee "$(dirname "$out")/console_logs/context_cluster_c48_seed1024_depth2.log"
echo "[round14] done tag=context_cluster_c48_seed1024_depth2 gpu=$gpu"
