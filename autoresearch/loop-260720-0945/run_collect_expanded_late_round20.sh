#!/usr/bin/env bash
set -euo pipefail

ROOT=autoresearch/loop-260720-0945/model_training_round20
PLANS=$ROOT/late_onpolicy_plans
MANIFEST=$ROOT/late_source_manifest.json
OUT=$ROOT/late_prefix_oracle
CHECKPOINT=autoresearch/loop-260720-0945/model_training_round8/runs/moe_joint_within/best_final_horizon.pt
mkdir -p "$PLANS/logs" "$OUT"

python -u scripts/prepare_expanded_late_round20.py | tee "$ROOT/prepare_late_sources.log"

export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 TPI_TORCH_DETERMINISTIC=1
export TPI_HARD_CLUSTER_MAX_HARD_NODES=1024 TPI_LATENT_NORM_CLIP_RATIO=4
export TPI_Q_CONTEXT_SUPPORT_ALPHA=0.45 TPI_Q_CONTEXT_DISAGREEMENT_BETA=0.10
export TPI_TYPED_RESIDUAL_ALPHA=0.10 TPI_TYPED_RESIDUAL_DECAY_STEPS=64
export TPI_TYPED_RESIDUAL_CLIP=1.0 TPI_TYPED_RESIDUAL_DISAGREEMENT_BETA=0.25
export TPI_TYPED_RELIABLE_MARGINAL_WEIGHT=0.75
export TPI_TYPED_RELIABLE_MIN_HEADS=1 TPI_TYPED_RELIABLE_CP0_MIN_HEADS=2
export TPI_TYPED_TRUST_MIN_HEADS=2 TPI_TYPED_TRUST_CP0_MIN_HEADS=3
export TPI_TYPED_TRUST_HEAD_MARGIN=0 TPI_TYPED_TRUST_ADVANTAGE_MARGIN=0
export TPI_ADAPTIVE_BASE_CANDIDATES=48 TPI_ADAPTIVE_MARGIN_MODE=relative_range
export TPI_ADAPTIVE_EXPANSION_MARGIN=0.003
export TPI_SCORE_QUANTIZATION=0.001 TPI_PLAN_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

mapfile -t benchmarks < <(python -c 'import json,sys; print(*json.load(open(sys.argv[1]))["accepted_benchmarks"], sep="\n")' "$MANIFEST")
IFS=',' read -r -a gpus <<< "${GPUS_CSV:-1,2,4,5,6,7}"
MAX_PARALLEL=${MAX_PARALLEL:-${#gpus[@]}}
if (( ${#gpus[@]} == 0 || MAX_PARALLEL < 1 )); then
  echo "GPUS_CSV must contain at least one GPU and MAX_PARALLEL must be positive" >&2
  exit 2
fi

run_plan() {
  local benchmark=$1 gpu=$2
  local out="$PLANS/$benchmark.csv"
  local old=autoresearch/loop-260720-0945/model_training_round9/late_onpolicy_plans/$benchmark.csv
  if [[ ! -s "$out" && -s "$old" ]] && (( $(wc -l < "$old") >= 257 )); then
    cp "$old" "$out"
    echo "[onpolicy-r20] reuse Round9 benchmark=$benchmark out=$out"
  fi
  if [[ -s "$out" ]] && (( $(wc -l < "$out") >= 257 )); then
    echo "[onpolicy-r20] skip complete benchmark=$benchmark out=$out"
    return 0
  fi
  echo "[onpolicy-r20] start benchmark=$benchmark gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" python -u -m tpi_jepa.plan \
    --checkpoint "$CHECKPOINT" --benchmark-id "$benchmark" --budget 256 \
    --max-candidates 64 --device cuda --planner greedy --beam-width 1 \
    --lookahead-depth 1 --score-field q_typed_reliable_context --beam-objective cumulative \
    --discount-gamma 1.0 --out "$out" --candidate-strategy hard_fault_cluster \
    --candidate-sample-seed 0 --candidate-diversity-penalty 0.0 --candidate-diversity-depth 4 \
    2>&1 | sed -u "s|^|[onpolicy-r20/$benchmark] |" | tee "$PLANS/logs/$benchmark.log"
  echo "[onpolicy-r20] done benchmark=$benchmark gpu=$gpu"
}

pids=(); failed=0; job=0
wait_batch() {
  local pid
  for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
  pids=()
}
for benchmark in "${benchmarks[@]}"; do
  gpu=${gpus[$((job % ${#gpus[@]}))]}
  run_plan "$benchmark" "$gpu" &
  pids+=("$!")
  job=$((job+1))
  if (( ${#pids[@]} == MAX_PARALLEL )); then wait_batch; fi
done
if (( ${#pids[@]} )); then wait_batch; fi
if (( failed )); then exit 1; fi
echo "[onpolicy-r20] plans_complete count=${#benchmarks[@]} out=$PLANS"

python -u scripts/collect_onpolicy_prefix_oracle.py \
  --plans-dir "$PLANS" --training-manifest "$MANIFEST" \
  --eval-protocol configs/eval_protocol_coverage_only.json \
  --prefix-steps 144,176,208,240,255 \
  --candidate-strategy hard_fault_cluster --candidate-pool-size 64 \
  --actions-per-prefix 15 --backend atalanta-bist --patterns 300000 --seed 2026 \
  --parallel-jobs "${ORACLE_JOBS:-18}" --timeout-sec 14400 --out-dir "$OUT" \
  --resume --cleanup-workdir \
  2>&1 | tee "$OUT/driver.log"
