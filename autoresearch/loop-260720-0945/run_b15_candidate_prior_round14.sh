#!/usr/bin/env bash
set -euo pipefail

LOOP=autoresearch/loop-260720-0945
BASE=$LOOP/model_training_round14_candidate_prior_b15
CHECKPOINT=$LOOP/model_training_round8/runs/moe_joint_within/best_final_horizon.pt
PREFIX_PLAN=$LOOP/typed_winner_round8_five/b15_C/best_plans/iscas99__b15_1.csv
PRIOR=$LOOP/typed_winner_round13_refresh_five/b15_C/residual_priors/real_fault_priors.csv
ALLOWLIST=autoresearch/original-netlist-recovery-260712/exact_itc99/b15_C/exact_candidate_nodes.txt
INCUMBENT=$LOOP/model_training_round13_restored/b15_selection.json
ALPHAS=(0 0.01 0.03 0.07 0.15 0.30)
IFS=',' read -r -a GPUS <<< "${GPUS_CSV:-0,1,2,4,5,6}"

mkdir -p "$BASE/logs" "$BASE/plans" "$BASE/evals"
if (( ${#GPUS[@]} < ${#ALPHAS[@]} )); then
  echo "need at least ${#ALPHAS[@]} GPUs in GPUS_CSV" >&2
  exit 2
fi

export TPI_BENCH_ROOT="$PWD/autoresearch/deeptpi_table2_restored_bench"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 TPI_TORCH_DETERMINISTIC=1
export TPI_HARD_CLUSTER_MAX_HARD_NODES=1024 TPI_LATENT_NORM_CLIP_RATIO=4
export TPI_Q_CONTEXT_SUPPORT_ALPHA=0.45 TPI_Q_CONTEXT_DISAGREEMENT_BETA=0.10
export TPI_TYPED_RESIDUAL_ALPHA=0.10 TPI_TYPED_RESIDUAL_DECAY_STEPS=64
export TPI_TYPED_RESIDUAL_CLIP=1.0 TPI_TYPED_RESIDUAL_DISAGREEMENT_BETA=0.25
export TPI_TYPED_RELIABLE_MARGINAL_WEIGHT=0.75
export TPI_TYPED_RELIABLE_MIN_HEADS=1 TPI_TYPED_RELIABLE_CP0_MIN_HEADS=2
export TPI_TYPED_TRUST_MIN_HEADS=2 TPI_TYPED_TRUST_CP0_MIN_HEADS=3
export TPI_TYPED_TRUST_HEAD_MARGIN=0 TPI_TYPED_TRUST_ADVANTAGE_MARGIN=0
export TPI_SCORE_QUANTIZATION=0.001 TPI_PLAN_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TPI_ADAPTIVE_BASE_CANDIDATES=48 TPI_ADAPTIVE_MARGIN_MODE=relative_range
export TPI_ADAPTIVE_EXPANSION_MARGIN=0.003

wait_all() {
  local failed=0 pid
  for pid in "$@"; do if ! wait "$pid"; then failed=1; fi; done
  if (( failed )); then return 1; fi
}

alpha_tag() { printf 'alpha_%0.3f' "$1" | tr '.' 'p'; }

pids=()
for index in "${!ALPHAS[@]}"; do
  alpha=${ALPHAS[$index]}
  gpu=${GPUS[$index]}
  tag=$(alpha_tag "$alpha")
  echo "[r14-plan] start alpha=$alpha gpu=$gpu"
  TPI_CANDIDATE_PRIOR_ALPHA="$alpha" CUDA_VISIBLE_DEVICES="$gpu" \
    python -u -m tpi_jepa.plan \
      --checkpoint "$CHECKPOINT" --benchmark-id iscas99__b15_1 --budget 278 \
      --max-candidates 64 --score-field q_typed_reliable_context --planner greedy \
      --candidate-strategy hard_fault_cluster --candidate-real-fault-priors "$PRIOR" \
      --candidate-allowlist "$ALLOWLIST" --prefix-plan "$PREFIX_PLAN" --prefix-steps 192 \
      --prefix-state-mode replay --device cuda --out "$BASE/plans/$tag.csv" \
      2>&1 | sed -u "s|^|[r14-plan/$alpha] |" | tee "$BASE/logs/plan_${tag}.log" &
  pids+=("$!")
done
wait_all "${pids[@]}"

pids=()
for alpha in "${ALPHAS[@]}"; do
  tag=$(alpha_tag "$alpha")
  echo "[r14-eval] start alpha=$alpha"
  python -u -m tpi_jepa.evaluate_plan_tmax \
    --benchmark-id iscas99__b15_1 --plan-csv "$BASE/plans/$tag.csv" \
    --out-dir "$BASE/evals/$tag" --backend atalanta-bist --patterns 300000 --seed 2026 \
    --eval-step-mode final \
    2>&1 | sed -u "s|^|[r14-eval/$alpha] |" | tee "$BASE/logs/eval_${tag}.log" &
  pids+=("$!")
done
wait_all "${pids[@]}"

python scripts/select_candidate_prior_alpha_on_b15.py \
  --base "$BASE" --incumbent-manifest "$INCUMBENT" \
  --alphas 0,0.01,0.03,0.07,0.15,0.30
