#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
index="${VARIANT_INDEX:-0}"
gpu="${GPU:-7}"
OUT_ROOT="${OUT_ROOT:-autoresearch/improve-260716-1344/round4_b17}"
mkdir -p "$OUT_ROOT/console_logs"

tags=(context_cluster_c96_prior context_cluster_c48_noprior context_cluster_c96_noprior context_hfc_c96_prior)
strategies=(hard_fault_cluster hard_fault_cluster hard_fault_cluster hard_fault_cone)
candidates=(96 48 96 96)
use_priors=(yes no no yes)
if (( index < 0 || index >= ${#tags[@]} )); then
  echo "VARIANT_INDEX must be between 0 and $((${#tags[@]} - 1))" >&2
  exit 2
fi

tag="${tags[$index]}"
strategy="${strategies[$index]}"
max_candidates="${candidates[$index]}"
extra=()
if [[ "${use_priors[$index]}" == yes ]]; then
  extra+=(--candidate-real-fault-priors autoresearch/eval8-real-priors-budget5-v1/real_fault_priors.csv)
fi

echo "[round4] start tag=$tag gpu=$gpu"
CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_gmean_sweep.py \
  --eval-protocol configs/eval_protocol_coverage_only.json --protocol-keep-cli-benchmarks \
  --benchmarks iscas99__b17 \
  --checkpoint runs/planner_aligned_q_rank_v5_context_safe/best_final_horizon.pt \
  --planners greedy --score-fields q_pred_context --beam-objectives cumulative \
  --beam-widths 1 --lookahead-depths 1 --max-candidates "$max_candidates" \
  --discount-gammas 0.9 --candidate-strategies "$strategy" \
  --candidate-diversity-penalties 0.0 --candidate-diversity-depths 4 \
  --candidate-allowlist autoresearch/original-netlist-recovery-260712/exact_itc99/b17_C/exact_candidate_nodes.txt \
  --plan-device cuda --time-limit-hours 72 --out-dir "$OUT_ROOT/$tag" \
  --stream-logs "${extra[@]}" 2>&1 | tee "$OUT_ROOT/console_logs/$tag.log"
echo "[round4] done tag=$tag gpu=$gpu"
