#!/usr/bin/env bash
set -euo pipefail

BASE=autoresearch/loop-260720-0945/model_training_round6_adaptive
OUT="$BASE/b15_selection/refine_reliable"
CHECKPOINT=autoresearch/loop-260720-0945/model_training_round6/runs/cone_toplist_sa/epoch_012.pt
mkdir -p "$OUT/logs"

export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 TPI_TORCH_DETERMINISTIC=1
export TPI_HARD_CLUSTER_MAX_HARD_NODES=1024 TPI_LATENT_NORM_CLIP_RATIO=4
export TPI_Q_CONTEXT_SUPPORT_ALPHA=0.45 TPI_Q_CONTEXT_DISAGREEMENT_BETA=0.10
export TPI_TYPED_RESIDUAL_ALPHA=0.10 TPI_TYPED_RESIDUAL_DECAY_STEPS=16
export TPI_TYPED_RESIDUAL_CLIP=1.0 TPI_TYPED_RESIDUAL_DISAGREEMENT_BETA=0.25
export TPI_TYPED_RELIABLE_MARGINAL_WEIGHT=0.75
export TPI_TYPED_RELIABLE_MIN_HEADS=1 TPI_TYPED_RELIABLE_CP0_MIN_HEADS=2
export TPI_TYPED_TRUST_HEAD_MARGIN=0 TPI_TYPED_TRUST_ADVANTAGE_MARGIN=0
export TPI_SCORE_QUANTIZATION=0.001 TPI_ADAPTIVE_BASE_CANDIDATES=48
export TPI_ADAPTIVE_MARGIN_MODE=relative_range TPI_PLAN_THREADS=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

margins=(0.002 0.0025 0.00275 0.00325 0.0035 0.004)
IFS=',' read -r -a gpus <<< "${GPUS_CSV:-4,5,6}"
MAX_PARALLEL=${MAX_PARALLEL:-${#gpus[@]}}
if (( ${#gpus[@]} == 0 || MAX_PARALLEL < 1 )); then
  echo "GPUS_CSV must contain at least one GPU and MAX_PARALLEL must be positive" >&2
  exit 2
fi

run_one() {
  local margin=$1 gpu=$2 tag=${margin/./p}
  local out="$OUT/margin_$tag"
  echo "[b15-r6-adaptive-refine] start margin=$margin gpu=$gpu"
  TPI_ADAPTIVE_EXPANSION_MARGIN="$margin" CUDA_VISIBLE_DEVICES="$gpu" \
    python -u scripts/run_gmean_sweep.py \
      --eval-protocol configs/eval_protocol_coverage_only.json --protocol-keep-cli-benchmarks \
      --benchmarks iscas99__b15_1 --checkpoint "$CHECKPOINT" \
      --planners greedy --score-fields q_typed_reliable_context --beam-objectives cumulative \
      --beam-widths 1 --lookahead-depths 1 --max-candidates 64 --discount-gammas 0.9 \
      --candidate-strategies hard_fault_cluster --candidate-diversity-penalties 0.0 \
      --candidate-diversity-depths 4 \
      --candidate-real-fault-priors autoresearch/eval8-real-priors-budget5-v1/real_fault_priors.csv \
      --candidate-allowlist autoresearch/original-netlist-recovery-260712/exact_itc99/b15_C/exact_candidate_nodes.txt \
      --plan-device cuda --time-limit-hours 24 --stream-logs --out-dir "$out" \
      2>&1 | sed -u "s|^|[b15-r6-adaptive-refine/m$margin] |" \
      | tee "$OUT/logs/m${margin}.log"
}

pids=(); failed=0; job=0
wait_batch() {
  local pid
  for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
  pids=()
}
for margin in "${margins[@]}"; do
  gpu=${gpus[$((job % ${#gpus[@]}))]}
  run_one "$margin" "$gpu" &
  pids+=("$!")
  job=$((job+1))
  if (( ${#pids[@]} == MAX_PARALLEL )); then wait_batch; fi
done
if (( ${#pids[@]} )); then wait_batch; fi
if (( failed )); then exit 1; fi

python scripts/select_onpolicy_typed_on_b15.py --base "$BASE" \
  --incumbent-manifest autoresearch/loop-260720-0945/model_training_round6/b15_selection/winner.json

