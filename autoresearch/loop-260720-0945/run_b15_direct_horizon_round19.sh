#!/usr/bin/env bash
set -euo pipefail

LOOP=autoresearch/loop-260720-0945
BASE=$LOOP/model_training_round19_direct_horizon_b15
PREFIX_PLAN=$LOOP/typed_winner_round8_five/b15_C/best_plans/iscas99__b15_1.csv
PRIOR=$LOOP/typed_winner_round13_refresh_five/b15_C/residual_priors/real_fault_priors.csv
ALLOWLIST=autoresearch/original-netlist-recovery-260712/exact_itc99/b15_C/exact_candidate_nodes.txt
INCUMBENT=$LOOP/model_training_round14_candidate_prior_b15/b15_selection.json
TAGS=(r8_marginal r8_return r11_lr2e5 r11_lr5e5 r11_lr1e4 r11_lr1e4_return)
CHECKPOINTS=(
  "$LOOP/model_training_round8/runs/moe_joint_within/best_final_horizon.pt"
  "$LOOP/model_training_round8/runs/moe_joint_within/best_final_horizon.pt"
  "$LOOP/model_training_round11/runs/marginal_horizon_lr2e5/best_final_horizon.pt"
  "$LOOP/model_training_round11/runs/marginal_horizon_lr5e5/best_final_horizon.pt"
  "$LOOP/model_training_round11/runs/marginal_horizon_lr1e4/best_final_horizon.pt"
  "$LOOP/model_training_round11/runs/marginal_horizon_lr1e4/best_final_horizon.pt"
)
SCORES=(typed_marginal_pred typed_return_pred typed_marginal_pred typed_marginal_pred typed_marginal_pred typed_return_pred)
IFS=',' read -r -a GPUS <<< "${GPUS_CSV:-1,2,4,5,6,7}"

mkdir -p "$BASE"/{logs,plans,evals}
if (( ${#GPUS[@]} < ${#TAGS[@]} )); then
  echo "need at least ${#TAGS[@]} GPUs in GPUS_CSV" >&2
  exit 2
fi

export TPI_BENCH_ROOT="$PWD/autoresearch/deeptpi_table2_restored_bench"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 TPI_TORCH_DETERMINISTIC=1
export TPI_HARD_CLUSTER_MAX_HARD_NODES=1024 TPI_LATENT_NORM_CLIP_RATIO=4
export TPI_SCORE_QUANTIZATION=0.001 TPI_PLAN_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TPI_ADAPTIVE_BASE_CANDIDATES=48 TPI_ADAPTIVE_MARGIN_MODE=relative_range
export TPI_ADAPTIVE_EXPANSION_MARGIN=0.003

wait_all() {
  local failed=0 pid
  for pid in "$@"; do if ! wait "$pid"; then failed=1; fi; done
  if (( failed )); then return 1; fi
}

pids=()
for index in "${!TAGS[@]}"; do
  tag=${TAGS[$index]}
  checkpoint=${CHECKPOINTS[$index]}
  score=${SCORES[$index]}
  gpu=${GPUS[$index]}
  echo "[r19-horizon-plan] start tag=$tag score=$score gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" python -u -m tpi_jepa.plan \
    --checkpoint "$checkpoint" --benchmark-id iscas99__b15_1 --budget 278 \
    --max-candidates 64 --score-field "$score" --planner greedy \
    --candidate-strategy hard_fault_cluster --candidate-real-fault-priors "$PRIOR" \
    --candidate-allowlist "$ALLOWLIST" --prefix-plan "$PREFIX_PLAN" --prefix-steps 192 \
    --prefix-state-mode replay --device cuda --out "$BASE/plans/$tag.csv" \
    2>&1 | sed -u "s|^|[r19-horizon-plan/$tag] |" | tee "$BASE/logs/plan_${tag}.log" &
  pids+=("$!")
done
wait_all "${pids[@]}"

pids=()
for tag in "${TAGS[@]}"; do
  echo "[r19-horizon-eval] start tag=$tag"
  python -u -m tpi_jepa.evaluate_plan_tmax \
    --benchmark-id iscas99__b15_1 --plan-csv "$BASE/plans/$tag.csv" \
    --out-dir "$BASE/evals/$tag" --backend atalanta-bist --patterns 300000 --seed 2026 \
    --eval-step-mode final \
    2>&1 | sed -u "s|^|[r19-horizon-eval/$tag] |" | tee "$BASE/logs/eval_${tag}.log" &
  pids+=("$!")
done
wait_all "${pids[@]}"

python scripts/select_b15_tail_variants.py \
  --base "$BASE" --incumbent-manifest "$INCUMBENT" \
  --variants r8_marginal,r8_return,r11_lr2e5,r11_lr5e5,r11_lr1e4,r11_lr1e4_return \
  --prefer-order r11_lr1e4,r11_lr5e5,r11_lr2e5,r8_marginal,r11_lr1e4_return,r8_return \
  2>&1 | tee "$BASE/logs/select.log"
