#!/usr/bin/env bash
set -euo pipefail

LOOP=autoresearch/loop-260720-0945
TRAIN=$LOOP/model_training_round22
BASE=$LOOP/model_training_round22_b15
PREFIX_PLAN=$LOOP/typed_winner_round8_five/b15_C/best_plans/iscas99__b15_1.csv
PRIOR=$LOOP/typed_winner_round13_refresh_five/b15_C/residual_priors/real_fault_priors.csv
ALLOWLIST=autoresearch/original-netlist-recovery-260712/exact_itc99/b15_C/exact_candidate_nodes.txt
INCUMBENT=$LOOP/model_training_round21_horizon_return_b15/b15_selection.json
VARIANTS=(toptype_r21_hard toptype_r21_all toptype_r10_hard)
EPOCHS=(004 006 008)
IFS=',' read -r -a gpus <<< "${GPUS_CSV:-5,6,7}"
MAX_PARALLEL=${MAX_PARALLEL:-${#gpus[@]}}
MAX_EVAL_PARALLEL=${MAX_EVAL_PARALLEL:-6}

if (( ${#gpus[@]} == 0 || MAX_PARALLEL < 1 || MAX_EVAL_PARALLEL < 1 )); then
  echo "GPU and parallelism settings must be non-empty and positive" >&2
  exit 2
fi
mkdir -p "$BASE"/{logs,plans,evals}

export TPI_BENCH_ROOT="$PWD/autoresearch/deeptpi_table2_restored_bench"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 TPI_TORCH_DETERMINISTIC=1
export TPI_HARD_CLUSTER_MAX_HARD_NODES=1024 TPI_LATENT_NORM_CLIP_RATIO=4
export TPI_SCORE_QUANTIZATION=0.001 TPI_PLAN_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TPI_ADAPTIVE_BASE_CANDIDATES=48 TPI_ADAPTIVE_MARGIN_MODE=relative_range
export TPI_ADAPTIVE_EXPANSION_MARGIN=0.003 TPI_CANDIDATE_PRIOR_ALPHA=0

wait_for_checkpoint() {
  local checkpoint=$1
  while [[ ! -s "$checkpoint" ]]; do
    if ! pgrep -f '^python -u -m tpi_jepa.train --config .*model_training_round22' >/dev/null; then
      echo "missing checkpoint after Round22 training stopped: $checkpoint" >&2
      return 1
    fi
    echo "[r22-b15] waiting checkpoint=$checkpoint"
    sleep 10
  done
}

run_plan() {
  local variant=$1 epoch=$2 gpu=$3
  local tag=${variant}__e${epoch}
  local checkpoint=$TRAIN/runs/$variant/epoch_${epoch}.pt
  wait_for_checkpoint "$checkpoint"
  echo "[r22-b15-plan] start tag=$tag gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" python -u -m tpi_jepa.plan \
    --checkpoint "$checkpoint" --benchmark-id iscas99__b15_1 --budget 278 \
    --max-candidates 72 --score-field typed_return_pred --planner greedy \
    --candidate-strategy hard_fault_cluster --candidate-real-fault-priors "$PRIOR" \
    --candidate-allowlist "$ALLOWLIST" --prefix-plan "$PREFIX_PLAN" --prefix-steps 192 \
    --prefix-state-mode replay --device cuda --out "$BASE/plans/$tag.csv" \
    2>&1 | sed -u "s|^|[r22-b15-plan/$tag] |" | tee "$BASE/logs/plan_${tag}.log"
  echo "[r22-b15-plan] done tag=$tag gpu=$gpu"
}

pids=(); tags=(); job=0; failed=0
wait_batch() {
  local pid
  for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
  pids=()
}
for variant in "${VARIANTS[@]}"; do
  for epoch in "${EPOCHS[@]}"; do
    tag=${variant}__e${epoch}
    tags+=("$tag")
    gpu=${gpus[$((job % ${#gpus[@]}))]}
    run_plan "$variant" "$epoch" "$gpu" &
    pids+=("$!")
    job=$((job+1))
    if (( ${#pids[@]} == MAX_PARALLEL )); then wait_batch; fi
  done
done
if (( ${#pids[@]} )); then wait_batch; fi
if (( failed )); then exit 1; fi

run_eval() {
  local tag=$1
  echo "[r22-b15-eval] start tag=$tag"
  python -u -m tpi_jepa.evaluate_plan_tmax \
    --benchmark-id iscas99__b15_1 --plan-csv "$BASE/plans/$tag.csv" \
    --out-dir "$BASE/evals/$tag" --backend atalanta-bist --patterns 300000 --seed 2026 \
    --eval-step-mode final \
    2>&1 | sed -u "s|^|[r22-b15-eval/$tag] |" | tee "$BASE/logs/eval_${tag}.log"
  echo "[r22-b15-eval] done tag=$tag"
}

pids=(); failed=0
for tag in "${tags[@]}"; do
  run_eval "$tag" &
  pids+=("$!")
  if (( ${#pids[@]} == MAX_EVAL_PARALLEL )); then wait_batch; fi
done
if (( ${#pids[@]} )); then wait_batch; fi
if (( failed )); then exit 1; fi

tags_csv=$(IFS=,; echo "${tags[*]}")
python scripts/select_b15_tail_variants.py \
  --base "$BASE" --incumbent-manifest "$INCUMBENT" \
  --variants "$tags_csv" --prefer-order "$tags_csv" \
  2>&1 | tee "$BASE/logs/select.log"
