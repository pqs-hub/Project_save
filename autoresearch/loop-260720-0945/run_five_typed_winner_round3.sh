#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
BASE=${BASE:-autoresearch/loop-260720-0945/model_training_round3}
WINNER="$BASE/b15_selection/winner.json"
OUT=${OUT:-autoresearch/loop-260720-0945/typed_winner_round3_five}
test -f "$WINNER"
mapfile -t selected < <(python -c \
  'import json,sys; w=json.load(open(sys.argv[1]))["winner"]; e=w.get("planner_environment") or {}; print(w["checkpoint"]); print(w["score_field"]); print(w["typed_residual_alpha"]); print(w["typed_residual_decay_steps"]); print(w.get("typed_trust_min_heads",2)); print(w.get("typed_trust_cp0_min_heads",3)); print(w.get("typed_trust_head_margin",0)); print(w.get("typed_trust_advantage_margin",0)); print(w.get("typed_reliable_marginal_weight",0.75)); print(w.get("typed_reliable_min_heads",1)); print(w.get("typed_reliable_cp0_min_heads",2)); print(w.get("ensemble_checkpoints", "")); print(w.get("ensemble_lcb_alpha",1.0)); print(w.get("max_candidates",48)); print(w.get("candidate_strategy","hard_fault_cluster")); print(w.get("adaptive_base_candidates",0)); print(w.get("adaptive_expansion_margin",0)); print(w.get("adaptive_margin_mode","absolute")); print(e.get("TPI_LATENT_REENCODE_INTERVAL",0)); print(e.get("TPI_LATENT_REENCODE_BLEND",1))' \
  "$WINNER")
CHECKPOINT=${selected[0]}
SCORE_FIELD=${selected[1]}
TPI_TYPED_RESIDUAL_ALPHA=${selected[2]}
TPI_TYPED_RESIDUAL_DECAY_STEPS=${selected[3]}
TPI_TYPED_TRUST_MIN_HEADS=${selected[4]}
TPI_TYPED_TRUST_CP0_MIN_HEADS=${selected[5]}
TPI_TYPED_TRUST_HEAD_MARGIN=${selected[6]}
TPI_TYPED_TRUST_ADVANTAGE_MARGIN=${selected[7]}
TPI_TYPED_RELIABLE_MARGINAL_WEIGHT=${selected[8]}
TPI_TYPED_RELIABLE_MIN_HEADS=${selected[9]}
TPI_TYPED_RELIABLE_CP0_MIN_HEADS=${selected[10]}
ENSEMBLE_CHECKPOINTS=${selected[11]}
ENSEMBLE_LCB_ALPHA=${selected[12]}
MAX_CANDIDATES=${selected[13]}
CANDIDATE_STRATEGY=${selected[14]}
TPI_ADAPTIVE_BASE_CANDIDATES=${selected[15]}
TPI_ADAPTIVE_EXPANSION_MARGIN=${selected[16]}
TPI_ADAPTIVE_MARGIN_MODE=${selected[17]}
TPI_LATENT_REENCODE_INTERVAL=${selected[18]}
TPI_LATENT_REENCODE_BLEND=${selected[19]}
mkdir -p "$OUT/logs"

export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 TPI_TORCH_DETERMINISTIC=1
export TPI_HARD_CLUSTER_MAX_HARD_NODES=1024 TPI_LATENT_NORM_CLIP_RATIO=4
export TPI_Q_CONTEXT_SUPPORT_ALPHA=0.45 TPI_Q_CONTEXT_DISAGREEMENT_BETA=0.10
export TPI_TYPED_RESIDUAL_ALPHA TPI_TYPED_RESIDUAL_DECAY_STEPS
export TPI_TYPED_TRUST_MIN_HEADS TPI_TYPED_TRUST_CP0_MIN_HEADS
export TPI_TYPED_TRUST_HEAD_MARGIN TPI_TYPED_TRUST_ADVANTAGE_MARGIN
export TPI_TYPED_RELIABLE_MARGINAL_WEIGHT TPI_TYPED_RELIABLE_MIN_HEADS
export TPI_TYPED_RELIABLE_CP0_MIN_HEADS
export TPI_ADAPTIVE_BASE_CANDIDATES TPI_ADAPTIVE_EXPANSION_MARGIN TPI_ADAPTIVE_MARGIN_MODE
export TPI_LATENT_REENCODE_INTERVAL TPI_LATENT_REENCODE_BLEND
export TPI_TYPED_RESIDUAL_CLIP=1.0 TPI_TYPED_RESIDUAL_DISAGREEMENT_BETA=0.25
export TPI_SCORE_QUANTIZATION=0.001 TPI_PLAN_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

circuits=(b15_C b20_C b21_C b22_C b17_C)
benchmarks=(iscas99__b15_1 iscas99__b20 iscas99__b21 iscas99__b22 iscas99__b17)
IFS=',' read -r -a gpus <<< "${GPUS_CSV:-0,1,2,4,5}"
if (( ${#gpus[@]} == 0 )); then
  echo "GPUS_CSV must contain at least one GPU" >&2
  exit 2
fi

run_one() {
  local circuit=$1 benchmark=$2 gpu=$3
  echo "[typed-five-r3] start circuit=$circuit gpu=$gpu checkpoint=$CHECKPOINT score=$SCORE_FIELD alpha=$TPI_TYPED_RESIDUAL_ALPHA decay=$TPI_TYPED_RESIDUAL_DECAY_STEPS candidates=$MAX_CANDIDATES adaptive=$TPI_ADAPTIVE_BASE_CANDIDATES/$TPI_ADAPTIVE_MARGIN_MODE/$TPI_ADAPTIVE_EXPANSION_MARGIN reencode=$TPI_LATENT_REENCODE_INTERVAL/$TPI_LATENT_REENCODE_BLEND"
  local ensemble_args=()
  if [[ -n "$ENSEMBLE_CHECKPOINTS" ]]; then
    ensemble_args=(--ensemble-checkpoints "$ENSEMBLE_CHECKPOINTS" --ensemble-lcb-alpha "$ENSEMBLE_LCB_ALPHA")
  fi
  CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/run_gmean_sweep.py \
    --eval-protocol configs/eval_protocol_coverage_only.json --protocol-keep-cli-benchmarks \
    --benchmarks "$benchmark" --checkpoint "$CHECKPOINT" \
    "${ensemble_args[@]}" \
    --planners greedy --score-fields "$SCORE_FIELD" \
    --beam-objectives cumulative --beam-widths 1 --lookahead-depths 1 \
    --max-candidates "$MAX_CANDIDATES" --discount-gammas 0.9 \
    --candidate-strategies "$CANDIDATE_STRATEGY" --candidate-diversity-penalties 0.0 \
    --candidate-diversity-depths 4 \
    --candidate-real-fault-priors autoresearch/eval8-real-priors-budget5-v1/real_fault_priors.csv \
    --candidate-allowlist "autoresearch/original-netlist-recovery-260712/exact_itc99/$circuit/exact_candidate_nodes.txt" \
    --plan-device cuda --time-limit-hours 72 --stream-logs --out-dir "$OUT/$circuit" \
    2>&1 | sed -u "s|^|[typed-five-r3/$circuit] |" | tee "$OUT/logs/$circuit.log"
  echo "[typed-five-r3] done circuit=$circuit gpu=$gpu"
}

pids=()
for index in "${!circuits[@]}"; do
  run_one "${circuits[$index]}" "${benchmarks[$index]}" "${gpus[$((index % ${#gpus[@]}))]}" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
if (( failed )); then exit 1; fi

python scripts/summarize_exact_itc99_eval.py --eval-root "$OUT"
python scripts/verify_uniform_exact_itc99.py "$OUT"
python scripts/check_deeptpi_goal.py "$OUT"
