#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/data4/pengqingsong/DFT/TPI-my.3}
cd "$REPO_ROOT"
export DFT_ROOT=${DFT_ROOT:-/data4/pengqingsong/DFT}

CHECKPOINT=${CHECKPOINT:-runs/mainline_world_model_simplified/best.pt}
OUT_DIR=${OUT_DIR:-autoresearch/mainline-ablation-eval8-260701-v2}
BENCHMARKS=${BENCHMARKS:-iscas99__b15_1,iscas99__b20,iscas99__b21,iscas99__b22,epfl__random_control__i2c__i2c,epfl__arithmetic__max__max,iscas99__b17,openabcd__mem_ctrl_orig}
SMALL_BENCHMARKS=${SMALL_BENCHMARKS:-iscas99__b15_1,epfl__random_control__i2c__i2c}
PRIOR_DIR=${PRIOR_DIR:-autoresearch/eval8-real-priors-budget5-v1}
PATTERNS=${PATTERNS:-300000}
SEED=${SEED:-2026}
TIMEOUT_SEC=${TIMEOUT_SEC:-14400}
EVAL_BACKEND=${EVAL_BACKEND:-atalanta-bist}
ATALANTA_BIN=${ATALANTA_BIN:-/data4/pengqingsong/DFT/tool/atalanta_bist_with_ufaults/atalanta}
PLAN_DEVICE=${PLAN_DEVICE:-cpu}
K_RECALL=${K_RECALL:-96}
K_MODEL=${K_MODEL:-32}
K_PLAN=${K_PLAN:-12}
EVAL_STEP_MODE=${EVAL_STEP_MODE:-final}
SAVE_STEP_TRAINING_DATA=${SAVE_STEP_TRAINING_DATA:-0}

mkdir -p "$OUT_DIR/logs"

EXTRA_EVAL_ARGS=(--eval-step-mode "$EVAL_STEP_MODE")
if [[ "$SAVE_STEP_TRAINING_DATA" == "1" ]]; then
  EXTRA_EVAL_ARGS+=(--save-step-training-data)
fi

run_logged() {
  local name=$1
  shift
  echo "[$(date -Is)] ${name}"
  "$@" 2>&1 | tee "$OUT_DIR/logs/${name}.log"
}

run_world_variant() {
  local name=$1
  local planners=$2
  local beam_widths=$3
  local depths=$4
  local benchmarks=$5
  run_logged "$name" python scripts/run_gmean_sweep.py \
    --checkpoint "$CHECKPOINT" \
    --benchmarks "$benchmarks" \
    --budget-mode fixed \
    --fixed-budget 5 \
    --planners "$planners" \
    --score-fields reward_pred \
    --beam-objectives cumulative \
    --beam-widths "$beam_widths" \
    --lookahead-depths "$depths" \
    --max-candidates "$K_RECALL" \
    --k-recalls "$K_RECALL" \
    --k-models "$K_MODEL" \
    --k-plans "$K_PLAN" \
    --candidate-strategies heuristic_recall_pool \
    --candidate-diversity-penalties 0.0 \
    --candidate-diversity-depths 4 \
    --patterns "$PATTERNS" \
    --seed "$SEED" \
    --timeout-sec "$TIMEOUT_SEC" \
    "${EXTRA_EVAL_ARGS[@]}" \
    --eval-backend "$EVAL_BACKEND" \
    --atalanta-bin "$ATALANTA_BIN" \
    --plan-device "$PLAN_DEVICE" \
    --time-limit-hours 72 \
    --out-dir "$OUT_DIR/$name"
}

run_heuristic_baseline() {
  local variant_dir="$OUT_DIR/A_heuristic_only"
  mkdir -p "$variant_dir/plans" "$variant_dir/evals" "$variant_dir/logs"
  for benchmark in ${BENCHMARKS//,/ }; do
    local plan_csv="$variant_dir/plans/${benchmark}.csv"
    run_logged "A_plan_${benchmark}" python scripts/plan_candidate_baseline.py \
      --benchmark-id "$benchmark" \
      --budget 5 \
      --candidate-strategy heuristic_recall_pool \
      --out "$plan_csv"
    run_logged "A_eval_${benchmark}" python -m tpi_jepa.evaluate_plan_tmax \
      --benchmark-id "$benchmark" \
      --plan-csv "$plan_csv" \
      --out-dir "$variant_dir/evals/$benchmark" \
      --patterns "$PATTERNS" \
      --seed "$SEED" \
      --backend "$EVAL_BACKEND" \
      --atalanta-bin "$ATALANTA_BIN" \
      --timeout-sec "$TIMEOUT_SEC" \
      "${EXTRA_EVAL_ARGS[@]}" \
      --force \
      --cleanup-workdir
  done
}

run_heuristic_baseline
run_world_variant "B_world_rerank" "greedy" "1" "1" "$BENCHMARKS"
run_world_variant "C_depth2_rollout" "beam" "2" "2" "$BENCHMARKS"
run_world_variant "D_depth3_small" "beam" "2" "3" "$SMALL_BENCHMARKS"
