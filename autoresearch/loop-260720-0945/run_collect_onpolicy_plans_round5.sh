#!/usr/bin/env bash
set -euo pipefail

ROOT=autoresearch/loop-260720-0945/model_training_round5
PLANS="$ROOT/onpolicy_plans"
CHECKPOINT=autoresearch/loop-260720-0945/model_training_round4/runs/onpolicy_balanced/best_final_horizon.pt
MANIFEST=autoresearch/loop-260720-0945/model_training_round3/structural32_current/manifest.json
mkdir -p "$PLANS/logs"

export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 TPI_TORCH_DETERMINISTIC=1
export TPI_HARD_CLUSTER_MAX_HARD_NODES=1024 TPI_LATENT_NORM_CLIP_RATIO=4
export TPI_Q_CONTEXT_SUPPORT_ALPHA=0.45 TPI_Q_CONTEXT_DISAGREEMENT_BETA=0.10
export TPI_TYPED_RESIDUAL_ALPHA=0.1 TPI_TYPED_RESIDUAL_CLIP=1.0
export TPI_TYPED_RESIDUAL_DECAY_STEPS=16 TPI_TYPED_RESIDUAL_DISAGREEMENT_BETA=0.25
export TPI_SCORE_QUANTIZATION=0.001 TPI_PLAN_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

mapfile -t benchmarks < <(python -c 'import json,sys; print(*json.load(open(sys.argv[1]))["accepted_benchmarks"], sep="\n")' "$MANIFEST")
IFS=',' read -r -a gpus <<< "${GPUS_CSV:-1,2,3,4}"
MAX_PARALLEL=${MAX_PARALLEL:-8}
if (( ${#gpus[@]} == 0 || MAX_PARALLEL < 1 )); then
  echo "GPUS_CSV must contain at least one GPU and MAX_PARALLEL must be positive" >&2
  exit 2
fi

run_plan() {
  local benchmark=$1 gpu=$2
  local out="$PLANS/$benchmark.csv"
  echo "[onpolicy-r5] start benchmark=$benchmark gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" python -u -m tpi_jepa.plan \
    --checkpoint "$CHECKPOINT" --benchmark-id "$benchmark" --budget 32 \
    --max-candidates 48 --device cuda --planner greedy --beam-width 1 \
    --lookahead-depth 1 --score-field q_typed_residual_context --beam-objective cumulative \
    --discount-gamma 1.0 --out "$out" --candidate-strategy hard_fault_cluster \
    --candidate-sample-seed 0 --candidate-diversity-penalty 0.0 --candidate-diversity-depth 4 \
    2>&1 | sed -u "s|^|[onpolicy-r5/$benchmark] |" | tee "$PLANS/logs/$benchmark.log"
  echo "[onpolicy-r5] done benchmark=$benchmark gpu=$gpu"
}

pids=()
failed=0
job=0
wait_batch() {
  local pid
  for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
  pids=()
}
for benchmark in "${benchmarks[@]}"; do
  gpu=${gpus[$((job % ${#gpus[@]}))]}
  run_plan "$benchmark" "$gpu" &
  pids+=("$!")
  job=$((job + 1))
  if (( ${#pids[@]} == MAX_PARALLEL )); then wait_batch; fi
done
if (( ${#pids[@]} )); then wait_batch; fi
if (( failed )); then exit 1; fi

echo "[onpolicy-r5] plans_complete count=${#benchmarks[@]} out=$PLANS"
