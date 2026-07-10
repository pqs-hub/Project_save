#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/data4/pengqingsong/DFT/TPI-my.3}
cd "$REPO_ROOT"

OUT_ROOT=${OUT_ROOT:-autoresearch/improve-260704-1656/b21_gate_only}
ATALANTA_BIN=${ATALANTA_BIN:-/data4/pengqingsong/DFT/tool/atalanta_bist_with_ufaults/atalanta}
mkdir -p "$OUT_ROOT/logs"

run_gate() {
  local name=$1
  local gpu=$2
  local checkpoint=$3
  local score_field=$4
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_gmean_sweep.py \
    --checkpoint "$checkpoint" \
    --benchmarks iscas99__b21 \
    --budget-mode floor1pct \
    --benchmark-budgets '{"iscas99__b21": 628}' \
    --planners greedy \
    --score-fields "$score_field" \
    --beam-objectives cumulative \
    --beam-widths 1 \
    --lookahead-depths 1 \
    --max-candidates 96 \
    --candidate-strategies heuristic_recall_pool \
    --candidate-diversity-penalties 0.0 \
    --candidate-diversity-depths 4 \
    --candidate-sample-seeds 0 \
    --candidate-real-fault-priors autoresearch/eval8-real-priors-budget5-v1/real_fault_priors.csv \
    --patterns 300000 \
    --seed 2026 \
    --timeout-sec 14400 \
    --eval-backend atalanta-bist \
    --atalanta-bin "$ATALANTA_BIN" \
    --plan-device cuda \
    --eval-step-mode final \
    --time-limit-hours 72 \
    --stream-logs \
    --out-dir "$OUT_ROOT/$name" \
    2>&1 | tee "$OUT_ROOT/logs/${name}.log"
}

run_gate "$@"
