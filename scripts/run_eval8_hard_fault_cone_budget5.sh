#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/data4/pengqingsong/DFT/TPI-my.3}
cd "$REPO_ROOT"
export DFT_ROOT=${DFT_ROOT:-/data4/pengqingsong/DFT}

BENCHMARKS=${BENCHMARKS:-iscas99__b15_1,iscas99__b20,iscas99__b21,iscas99__b22,epfl__random_control__i2c__i2c,epfl__arithmetic__max__max,iscas99__b17,openabcd__mem_ctrl_orig}
PRIOR_DIR=${PRIOR_DIR:-autoresearch/eval8-real-priors-budget5-v1}
OUT_DIR=${OUT_DIR:-autoresearch/eval8-mainline-smallk-budget5-300k-v3}
CHECKPOINT=${CHECKPOINT:-runs/mainline_world_model_simplified/best.pt}
EVAL_BACKEND=${EVAL_BACKEND:-atalanta-bist}
ATALANTA_BIN=${ATALANTA_BIN:-/data4/pengqingsong/DFT/tool/atalanta_bist_with_ufaults/atalanta}
PLAN_DEVICE=${PLAN_DEVICE:-cuda}
PLANNERS=${PLANNERS:-beam}
BEAM_WIDTH=${BEAM_WIDTH:-2}
LOOKAHEAD_DEPTH=${LOOKAHEAD_DEPTH:-2}
K_RECALL=${K_RECALL:-96}
K_MODEL=${K_MODEL:-32}
K_PLAN=${K_PLAN:-12}
EVAL_STEP_MODE=${EVAL_STEP_MODE:-final}
SAVE_STEP_TRAINING_DATA=${SAVE_STEP_TRAINING_DATA:-0}

mkdir -p "$PRIOR_DIR" "$OUT_DIR"

EXTRA_EVAL_ARGS=(--eval-step-mode "$EVAL_STEP_MODE")
if [[ "$SAVE_STEP_TRAINING_DATA" == "1" ]]; then
  EXTRA_EVAL_ARGS+=(--save-step-training-data)
fi

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
    --backend "$EVAL_BACKEND" \
    --atalanta-bin "$ATALANTA_BIN" \
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

echo "[$(date -Is)] running eval8 mainline heuristic_recall_pool budget=5 300k"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6} python scripts/run_gmean_sweep.py \
  --checkpoint "$CHECKPOINT" \
  --benchmarks "$BENCHMARKS" \
  --budget-mode fixed \
  --fixed-budget 5 \
  --planners "$PLANNERS" \
  --score-fields reward_pred \
  --beam-objectives cumulative \
  --beam-widths "$BEAM_WIDTH" \
  --lookahead-depths "$LOOKAHEAD_DEPTH" \
  --max-candidates "$K_RECALL" \
  --k-recalls "$K_RECALL" \
  --k-models "$K_MODEL" \
  --k-plans "$K_PLAN" \
  --candidate-strategies heuristic_recall_pool \
  --candidate-diversity-penalties 0.0 \
  --candidate-diversity-depths 4 \
  --patterns 300000 \
  --seed 2026 \
  --timeout-sec 14400 \
  "${EXTRA_EVAL_ARGS[@]}" \
  --eval-backend "$EVAL_BACKEND" \
  --atalanta-bin "$ATALANTA_BIN" \
  --plan-device "$PLAN_DEVICE" \
  --time-limit-hours 72 \
  --out-dir "$OUT_DIR"
