#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
gpu="${GPU:-7}"
out="${OUT_DIR:-autoresearch/improve-260716-1344/round7_b17/reward_cluster_c48_seed512}"
mkdir -p "$(dirname "$out")/console_logs"

echo "[round7] start tag=reward_cluster_c48_seed512 gpu=$gpu"
CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_gmean_sweep.py \
  --eval-protocol configs/eval_protocol_coverage_only.json --protocol-keep-cli-benchmarks \
  --benchmarks iscas99__b17 \
  --checkpoint runs/rollout_loss_A_reward_only/epoch_009.pt \
  --planners greedy --score-fields reward_pred --beam-objectives cumulative \
  --beam-widths 1 --lookahead-depths 1 --max-candidates 48 \
  --discount-gammas 0.9 --candidate-strategies hard_fault_cluster \
  --candidate-diversity-penalties 0.0 --candidate-diversity-depths 4 \
  --candidate-allowlist autoresearch/original-netlist-recovery-260712/exact_itc99/b17_C/exact_candidate_nodes.txt \
  --plan-device cuda --time-limit-hours 72 --out-dir "$out" \
  --stream-logs 2>&1 | tee "$(dirname "$out")/console_logs/reward_cluster_c48_seed512.log"
echo "[round7] done tag=reward_cluster_c48_seed512 gpu=$gpu"
