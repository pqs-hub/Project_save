#!/usr/bin/env bash
set -euo pipefail

ROOT=autoresearch/loop-260720-0945/model_training_round4
PLANS="$ROOT/onpolicy_plans"
LABELS="$ROOT/onpolicy_real_labels"
CHECKPOINT=runs/planner_aligned_q_rank_v5_context_safe/best_final_horizon.pt
MANIFEST=autoresearch/loop-260720-0945/model_training_round3/structural32_current/manifest.json
mkdir -p "$PLANS/logs" "$LABELS"

export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 TPI_TORCH_DETERMINISTIC=1
export TPI_HARD_CLUSTER_MAX_HARD_NODES=1024 TPI_LATENT_NORM_CLIP_RATIO=4
export TPI_Q_CONTEXT_SUPPORT_ALPHA=0.45 TPI_Q_CONTEXT_DISAGREEMENT_BETA=0.10
export TPI_SCORE_QUANTIZATION=0.001 TPI_PLAN_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

mapfile -t benchmarks < <(python -c 'import json,sys; print(*json.load(open(sys.argv[1]))["accepted_benchmarks"], sep="\n")' "$MANIFEST")

run_plan() {
  local benchmark=$1 gpu=$2
  local out="$PLANS/$benchmark.csv"
  echo "[onpolicy-r4] start benchmark=$benchmark gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" python -u -m tpi_jepa.plan \
    --checkpoint "$CHECKPOINT" --benchmark-id "$benchmark" --budget 32 \
    --max-candidates 48 --device cuda --planner greedy --beam-width 1 \
    --lookahead-depth 1 --score-field q_pred_context --beam-objective cumulative \
    --discount-gamma 1.0 --out "$out" --candidate-strategy hard_fault_cluster \
    --candidate-sample-seed 0 --candidate-diversity-penalty 0.0 --candidate-diversity-depth 4 \
    2>&1 | sed -u "s|^|[onpolicy-r4/$benchmark] |" | tee "$PLANS/logs/$benchmark.log"
  echo "[onpolicy-r4] done benchmark=$benchmark gpu=$gpu"
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
  gpu=$((job % 4 + 1))
  run_plan "$benchmark" "$gpu" &
  pids+=("$!")
  job=$((job + 1))
  if (( ${#pids[@]} == 8 )); then wait_batch; fi
done
if (( ${#pids[@]} )); then wait_batch; fi
if (( failed )); then exit 1; fi

python scripts/merge_plans_as_sequence_labels.py --plans-dir "$PLANS" --out "$ROOT/onpolicy_source_labels.csv"
python -u scripts/relabel_sequences_with_backend.py \
  --labels "$ROOT/onpolicy_source_labels.csv" --out-dir "$LABELS" \
  --backend atalanta-bist --patterns 100000 --seed 2026 --parallel-jobs 12 \
  --max-sequences 24 --max-steps 32 --timeout-sec 14400 --resume \
  --cleanup-workdir --drop-partial-sequences \
  2>&1 | tee "$LABELS/driver.log"
