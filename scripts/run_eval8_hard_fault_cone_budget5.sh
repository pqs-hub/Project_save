#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/data3/pengqingsong/DFT/TPI-my.2}
cd "$REPO_ROOT"

BENCHMARKS=${BENCHMARKS:-iscas99__b15_1,iscas99__b20,iscas99__b21,iscas99__b22,epfl__random_control__i2c__i2c,epfl__arithmetic__max__max,iscas99__b17,openabcd__mem_ctrl_orig}
PRIOR_DIR=${PRIOR_DIR:-autoresearch/eval8-real-priors-budget5-v1}
OUT_DIR=${OUT_DIR:-autoresearch/eval8-hard-fault-cone-budget5-300k-v1}
CHECKPOINT=${CHECKPOINT:-autoresearch/hard-fault-cone-distill-train-smallmid-v2/best.pt}

mkdir -p "$PRIOR_DIR" "$OUT_DIR"

if [[ ! -f "$PRIOR_DIR/real_fault_priors.csv" ]]; then
  echo "[$(date -Is)] collecting eval baseline TMAX fault logs"
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6} python scripts/collect_real_fault_logs.py \
    --benchmarks "$BENCHMARKS" \
    --out-dir "$PRIOR_DIR/fault_logs" \
    --top-k 0 \
    --patterns 300000 \
    --timeout-sec 14400 \
    --parallel-jobs 2 \
    --devices 0,1 \
    --resume --force

  python scripts/build_real_fault_priors.py "$PRIOR_DIR/fault_logs" \
    --out-csv "$PRIOR_DIR/real_fault_priors.csv" \
    --out-json "$PRIOR_DIR/real_fault_priors.json"
fi

if [[ ! -f "$PRIOR_DIR/activation_priors_30k.csv" ]]; then
  echo "[$(date -Is)] building eval activation priors"
  python scripts/build_activation_priors.py \
    --benchmarks "$BENCHMARKS" \
    --patterns 30000 \
    --batch-size 2048 \
    --device cpu \
    --out-csv "$PRIOR_DIR/activation_priors_30k.csv" \
    --out-json "$PRIOR_DIR/activation_priors_30k.json"
fi

echo "[$(date -Is)] running eval8 hard_fault_cone budget=5 300k"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6} python scripts/run_gmean_sweep.py \
  --checkpoint "$CHECKPOINT" \
  --benchmarks "$BENCHMARKS" \
  --budget-mode fixed \
  --fixed-budget 5 \
  --planners beam \
  --score-fields reward_pred \
  --beam-objectives cumulative \
  --beam-widths 4 \
  --lookahead-depths 3 \
  --max-candidates 96 \
  --candidate-strategies hard_fault_cone \
  --candidate-diversity-penalties 0.0 \
  --candidate-diversity-depths 4 \
  --patterns 300000 \
  --seed 2026 \
  --timeout-sec 14400 \
  --time-limit-hours 72 \
  --real-fault-priors "$PRIOR_DIR/real_fault_priors.csv" \
  --activation-priors "$PRIOR_DIR/activation_priors_30k.csv" \
  --out-dir "$OUT_DIR"
