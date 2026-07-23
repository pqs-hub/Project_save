#!/usr/bin/env bash
set -euo pipefail

# Interpolate continuously between the frozen q-context policy (alpha=0) and
# the typed residual policy under the same global adaptive-c64 recall rule.
BASE=autoresearch/loop-260720-0945/model_training_round6_adaptive_alpha
OUT="$BASE/b15_selection"
CHECKPOINT=autoresearch/loop-260720-0945/model_training_round4/runs/onpolicy_balanced/best_final_horizon.pt
mkdir -p "$OUT/logs"

export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 TPI_TORCH_DETERMINISTIC=1
export TPI_HARD_CLUSTER_MAX_HARD_NODES=1024 TPI_LATENT_NORM_CLIP_RATIO=4
export TPI_Q_CONTEXT_SUPPORT_ALPHA=0.45 TPI_Q_CONTEXT_DISAGREEMENT_BETA=0.10
export TPI_TYPED_RESIDUAL_CLIP=1.0 TPI_TYPED_RESIDUAL_DISAGREEMENT_BETA=0.25
export TPI_TYPED_RESIDUAL_DECAY_STEPS=16 TPI_SCORE_QUANTIZATION=0.001
export TPI_ADAPTIVE_BASE_CANDIDATES=48 TPI_ADAPTIVE_MARGIN_MODE=relative_range
export TPI_PLAN_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

alphas=(0.00 0.01 0.02 0.03 0.05 0.075)
margins=(0.001 0.003)
IFS=',' read -r -a gpus <<< "${GPUS_CSV:-0,1,2,4,5,6}"
MAX_PARALLEL=${MAX_PARALLEL:-${#gpus[@]}}
if (( ${#gpus[@]} == 0 || MAX_PARALLEL < 1 )); then
  echo "GPUS_CSV must contain at least one GPU and MAX_PARALLEL must be positive" >&2
  exit 2
fi

run_one() {
  local alpha=$1 margin=$2 gpu=$3
  local alpha_tag=${alpha/./p} margin_tag=${margin/./p}
  local out="$OUT/alpha_${alpha_tag}_margin_${margin_tag}"
  echo "[b15-r6-adaptive-alpha] start alpha=$alpha margin=$margin gpu=$gpu"
  TPI_TYPED_RESIDUAL_ALPHA="$alpha" TPI_ADAPTIVE_EXPANSION_MARGIN="$margin" \
  CUDA_VISIBLE_DEVICES="$gpu" \
    python -u scripts/run_gmean_sweep.py \
      --eval-protocol configs/eval_protocol_coverage_only.json --protocol-keep-cli-benchmarks \
      --benchmarks iscas99__b15_1 --checkpoint "$CHECKPOINT" \
      --planners greedy --score-fields q_typed_residual_context --beam-objectives cumulative \
      --beam-widths 1 --lookahead-depths 1 --max-candidates 64 --discount-gammas 0.9 \
      --candidate-strategies hard_fault_cluster --candidate-diversity-penalties 0.0 \
      --candidate-diversity-depths 4 \
      --candidate-real-fault-priors autoresearch/eval8-real-priors-budget5-v1/real_fault_priors.csv \
      --candidate-allowlist autoresearch/original-netlist-recovery-260712/exact_itc99/b15_C/exact_candidate_nodes.txt \
      --plan-device cuda --time-limit-hours 24 --stream-logs --out-dir "$out" \
      2>&1 | sed -u "s|^|[b15-r6-adaptive-alpha/a$alpha/m$margin] |" \
      | tee "$OUT/logs/a${alpha}__m${margin}.log"
}

pids=(); failed=0; job=0
wait_batch() {
  local pid
  for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
  pids=()
}
for alpha in "${alphas[@]}"; do
  for margin in "${margins[@]}"; do
    gpu=${gpus[$((job % ${#gpus[@]}))]}
    run_one "$alpha" "$margin" "$gpu" &
    pids+=("$!")
    job=$((job+1))
    if (( ${#pids[@]} == MAX_PARALLEL )); then wait_batch; fi
  done
done
if (( ${#pids[@]} )); then wait_batch; fi
if (( failed )); then exit 1; fi

python scripts/select_onpolicy_typed_on_b15.py --base "$BASE" \
  --incumbent-manifest autoresearch/loop-260720-0945/model_training_round6_adaptive/b15_selection/winner.json

