#!/usr/bin/env bash
set -euo pipefail

LOOP=autoresearch/loop-260720-0945
BASE=$LOOP/model_training_round20_round10_epoch8_direct_return_b15
PREFIX_PLAN=$LOOP/typed_winner_round8_five/b15_C/best_plans/iscas99__b15_1.csv
PRIOR=$LOOP/typed_winner_round13_refresh_five/b15_C/residual_priors/real_fault_priors.csv
ALLOWLIST=autoresearch/original-netlist-recovery-260712/exact_itc99/b15_C/exact_candidate_nodes.txt
INCUMBENT=$LOOP/model_training_round20_round10_direct_return_b15/b15_selection.json
TAGS=(within_lr2e4_epoch8 within_lr5e4_epoch8 dual_lr2e4_epoch8)
VARIANTS=(return_within_lr2e4 return_within_lr5e4 return_dual_lr2e4)
GPU=${GPU:-0}

mkdir -p "$BASE"/{logs,plans,evals}
export TPI_BENCH_ROOT="$PWD/autoresearch/deeptpi_table2_restored_bench"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 TPI_TORCH_DETERMINISTIC=1
export TPI_HARD_CLUSTER_MAX_HARD_NODES=1024 TPI_LATENT_NORM_CLIP_RATIO=4
export TPI_SCORE_QUANTIZATION=0.001 TPI_PLAN_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TPI_ADAPTIVE_BASE_CANDIDATES=48 TPI_ADAPTIVE_MARGIN_MODE=relative_range
export TPI_ADAPTIVE_EXPANSION_MARGIN=0.003 TPI_CANDIDATE_PRIOR_ALPHA=0

wait_all() {
  local failed=0 pid
  for pid in "$@"; do if ! wait "$pid"; then failed=1; fi; done
  if (( failed )); then return 1; fi
}

pids=()
for index in "${!TAGS[@]}"; do
  tag=${TAGS[$index]}
  variant=${VARIANTS[$index]}
  checkpoint=$LOOP/model_training_round10/runs/$variant/epoch_008.pt
  echo "[r20-r10-epoch8-plan] start tag=$tag gpu=$GPU"
  CUDA_VISIBLE_DEVICES="$GPU" python -u -m tpi_jepa.plan \
    --checkpoint "$checkpoint" --benchmark-id iscas99__b15_1 --budget 278 \
    --max-candidates 72 --score-field typed_return_pred --planner greedy \
    --candidate-strategy hard_fault_cluster --candidate-real-fault-priors "$PRIOR" \
    --candidate-allowlist "$ALLOWLIST" --prefix-plan "$PREFIX_PLAN" --prefix-steps 192 \
    --prefix-state-mode replay --device cuda --out "$BASE/plans/$tag.csv" \
    2>&1 | sed -u "s|^|[r20-r10-epoch8-plan/$tag] |" | tee "$BASE/logs/plan_${tag}.log" &
  pids+=("$!")
done
wait_all "${pids[@]}"

pids=()
for tag in "${TAGS[@]}"; do
  echo "[r20-r10-epoch8-eval] start tag=$tag"
  python -u -m tpi_jepa.evaluate_plan_tmax \
    --benchmark-id iscas99__b15_1 --plan-csv "$BASE/plans/$tag.csv" \
    --out-dir "$BASE/evals/$tag" --backend atalanta-bist --patterns 300000 --seed 2026 \
    --eval-step-mode final \
    2>&1 | sed -u "s|^|[r20-r10-epoch8-eval/$tag] |" | tee "$BASE/logs/eval_${tag}.log" &
  pids+=("$!")
done
wait_all "${pids[@]}"

python scripts/select_b15_tail_variants.py \
  --base "$BASE" --incumbent-manifest "$INCUMBENT" \
  --variants within_lr2e4_epoch8,within_lr5e4_epoch8,dual_lr2e4_epoch8 \
  --prefer-order dual_lr2e4_epoch8,within_lr5e4_epoch8,within_lr2e4_epoch8 \
  2>&1 | tee "$BASE/logs/select.log"
