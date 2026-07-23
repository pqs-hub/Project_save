#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

BASE=autoresearch/loop-260720-0945/model_training_round7
SELECTION="$BASE/b15_selection"
WINNER="$SELECTION/winner.json"
INCUMBENT=autoresearch/loop-260720-0945/model_training_round6_adaptive/b15_selection/winner.json
mkdir -p "$SELECTION/logs"

python scripts/select_onpolicy_typed_on_b15.py \
  --base "$BASE" --incumbent-manifest "$INCUMBENT" >/dev/null

mapfile -t selected < <(python -c \
  'import json,sys; w=json.load(open(sys.argv[1]))["winner"]; print(w["checkpoint"]); print(w["score_field"]); print(w["typed_residual_alpha"]); print(w["typed_residual_decay_steps"]); print(w.get("typed_reliable_marginal_weight",0.75)); print(w.get("typed_reliable_min_heads",1)); print(w.get("typed_reliable_cp0_min_heads",2)); print(w.get("adaptive_base_candidates",48)); print(w.get("adaptive_expansion_margin",0.003)); print(w.get("adaptive_margin_mode","relative_range")); print(w.get("max_candidates",64)); print(w.get("candidate_strategy","hard_fault_cluster"))' \
  "$WINNER")
CHECKPOINT=${selected[0]}
SCORE_FIELD=${selected[1]}
TPI_TYPED_RESIDUAL_ALPHA=${selected[2]}
TPI_TYPED_RESIDUAL_DECAY_STEPS=${selected[3]}
TPI_TYPED_RELIABLE_MARGINAL_WEIGHT=${selected[4]}
TPI_TYPED_RELIABLE_MIN_HEADS=${selected[5]}
TPI_TYPED_RELIABLE_CP0_MIN_HEADS=${selected[6]}
TPI_ADAPTIVE_BASE_CANDIDATES=${selected[7]}
TPI_ADAPTIVE_EXPANSION_MARGIN=${selected[8]}
TPI_ADAPTIVE_MARGIN_MODE=${selected[9]}
MAX_CANDIDATES=${selected[10]}
CANDIDATE_STRATEGY=${selected[11]}

export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 TPI_TORCH_DETERMINISTIC=1
export TPI_HARD_CLUSTER_MAX_HARD_NODES=1024 TPI_LATENT_NORM_CLIP_RATIO=4
export TPI_Q_CONTEXT_SUPPORT_ALPHA=0.45 TPI_Q_CONTEXT_DISAGREEMENT_BETA=0.10
export TPI_TYPED_RESIDUAL_ALPHA TPI_TYPED_RESIDUAL_DECAY_STEPS
export TPI_TYPED_RESIDUAL_CLIP=1.0 TPI_TYPED_RESIDUAL_DISAGREEMENT_BETA=0.25
export TPI_TYPED_RELIABLE_MARGINAL_WEIGHT TPI_TYPED_RELIABLE_MIN_HEADS
export TPI_TYPED_RELIABLE_CP0_MIN_HEADS
export TPI_TYPED_TRUST_MIN_HEADS=2 TPI_TYPED_TRUST_CP0_MIN_HEADS=3
export TPI_TYPED_TRUST_HEAD_MARGIN=0 TPI_TYPED_TRUST_ADVANTAGE_MARGIN=0
export TPI_ADAPTIVE_BASE_CANDIDATES TPI_ADAPTIVE_EXPANSION_MARGIN TPI_ADAPTIVE_MARGIN_MODE
export TPI_SCORE_QUANTIZATION=0.001 TPI_PLAN_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# Periodically reconstruct the latent state from the exact selected-action
# feature state.  This is a uniform world-model drift correction, not an ATPG
# feedback step; only interval and blend are selected on b15.
settings=(
  "16 0.25" "16 0.50" "16 1.00"
  "32 0.25" "32 0.50" "32 1.00"
  "64 0.25" "64 0.50" "64 1.00"
)

IFS=',' read -r -a gpus <<< "${GPUS_CSV:-0,1,2,4,5,6}"
MAX_PARALLEL=${MAX_PARALLEL:-${#gpus[@]}}
if (( ${#gpus[@]} == 0 || MAX_PARALLEL < 1 )); then
  echo "GPUS_CSV must contain at least one GPU and MAX_PARALLEL must be positive" >&2
  exit 2
fi

run_one() {
  local interval=$1 blend=$2 gpu=$3
  local blend_tag=${blend/./p}
  local out="$SELECTION/latent_reencode/interval_${interval}/blend_${blend_tag}"
  local log="$SELECTION/logs/latent_reencode__i${interval}__b${blend}.log"
  echo "[b15-reencode] start interval=$interval blend=$blend gpu=$gpu checkpoint=$CHECKPOINT"
  TPI_LATENT_REENCODE_INTERVAL="$interval" TPI_LATENT_REENCODE_BLEND="$blend" \
  CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/run_gmean_sweep.py \
    --eval-protocol configs/eval_protocol_coverage_only.json --protocol-keep-cli-benchmarks \
    --benchmarks iscas99__b15_1 --checkpoint "$CHECKPOINT" \
    --planners greedy --score-fields "$SCORE_FIELD" --beam-objectives cumulative \
    --beam-widths 1 --lookahead-depths 1 --max-candidates "$MAX_CANDIDATES" \
    --discount-gammas 0.9 --candidate-strategies "$CANDIDATE_STRATEGY" \
    --candidate-diversity-penalties 0.0 --candidate-diversity-depths 4 \
    --candidate-real-fault-priors autoresearch/eval8-real-priors-budget5-v1/real_fault_priors.csv \
    --candidate-allowlist autoresearch/original-netlist-recovery-260712/exact_itc99/b15_C/exact_candidate_nodes.txt \
    --plan-device cuda --time-limit-hours 24 --stream-logs --out-dir "$out" \
    2>&1 | sed -u "s|^|[b15-reencode/i$interval/b$blend] |" | tee "$log"
}

pids=(); failed=0; job=0
wait_batch() {
  local pid
  for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
  pids=()
}
for setting in "${settings[@]}"; do
  read -r interval blend <<< "$setting"
  gpu=${gpus[$((job % ${#gpus[@]}))]}
  run_one "$interval" "$blend" "$gpu" &
  pids+=("$!")
  job=$((job+1))
  if (( ${#pids[@]} == MAX_PARALLEL )); then wait_batch; fi
done
if (( ${#pids[@]} )); then wait_batch; fi
if (( failed )); then exit 1; fi

python scripts/select_onpolicy_typed_on_b15.py \
  --base "$BASE" --incumbent-manifest "$INCUMBENT"
