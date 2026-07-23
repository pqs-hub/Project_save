#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
LOOP=autoresearch/loop-260720-0945
SOURCE=$LOOP/typed_winner_round13_refresh_five
OUT=$LOOP/typed_winner_round19_direct_return_five
INCUMBENT=$LOOP/typed_winner_round8_five
CHECKPOINT=$LOOP/model_training_round8/runs/moe_joint_within/best_final_horizon.pt
MAPPING=autoresearch/original-netlist-recovery-260712/exact_itc99
RATIO_NUMERATOR=192
RATIO_DENOMINATOR=278

circuits=(b15_C b20_C b21_C b22_C b17_C)
benchmarks=(iscas99__b15_1 iscas99__b20 iscas99__b21 iscas99__b22 iscas99__b17)
budgets=(278 616 628 915 994)
IFS=',' read -r -a gpus <<< "${GPUS_CSV:-1,2,4,5}"
if (( ${#gpus[@]} < 4 )); then
  echo "need at least four GPUs in GPUS_CSV for b20/b21/b22/b17" >&2
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
export TPI_CANDIDATE_PRIOR_ALPHA=0
export TPI_SCORE_QUANTIZATION=0.001 TPI_PLAN_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TPI_ADAPTIVE_BASE_CANDIDATES=48 TPI_ADAPTIVE_MARGIN_MODE=relative_range
export TPI_ADAPTIVE_EXPANSION_MARGIN=0.003

wait_all() {
  local failed=0 pid
  for pid in "$@"; do if ! wait "$pid"; then failed=1; fi; done
  if (( failed )); then return 1; fi
}

mkdir -p "$OUT/logs"
for circuit in "${circuits[@]}"; do
  mkdir -p "$OUT/$circuit"/{prefix_eval,residual_priors,plans,evals/final,logs,timing}
  cp -a "$SOURCE/$circuit/prefix_eval/." "$OUT/$circuit/prefix_eval/"
  cp -a "$SOURCE/$circuit/residual_priors/." "$OUT/$circuit/residual_priors/"
  if test -f "$SOURCE/$circuit/timing/prefix_eval_sec.txt"; then
    cp "$SOURCE/$circuit/timing/prefix_eval_sec.txt" "$OUT/$circuit/timing/prefix_eval_sec.txt"
  fi
done

# b15 is the only selection circuit.  Reuse its already evaluated Round19
# typed-return/c72 artifact; the scoring head and width are now frozen.
cp "$LOOP/model_training_round19_direct_return_width_b15/plans/c72.csv" \
  "$OUT/b15_C/plans/iscas99__b15_1.csv"
cp -a "$LOOP/model_training_round19_direct_return_width_b15/evals/c72/." \
  "$OUT/b15_C/evals/final/"

pids=()
for index in 1 2 3 4; do
  circuit=${circuits[$index]}
  benchmark=${benchmarks[$index]}
  budget=${budgets[$index]}
  gpu=${gpus[$((index - 1))]}
  refresh_steps=$((budget * RATIO_NUMERATOR / RATIO_DENOMINATOR))
  prefix_plan="$INCUMBENT/$circuit/best_plans/$benchmark.csv"
  echo "[r19-five-plan] start circuit=$circuit gpu=$gpu budget=$budget refresh=$refresh_steps score=typed_return_pred"
  CUDA_VISIBLE_DEVICES="$gpu" /usr/bin/time -f '%e' -o "$OUT/$circuit/timing/plan_sec.txt" \
    python -u -m tpi_jepa.plan \
      --checkpoint "$CHECKPOINT" --benchmark-id "$benchmark" --budget "$budget" \
      --max-candidates 72 --score-field typed_return_pred --planner greedy \
      --candidate-strategy hard_fault_cluster \
      --candidate-real-fault-priors "$OUT/$circuit/residual_priors/real_fault_priors.csv" \
      --candidate-allowlist "$MAPPING/$circuit/exact_candidate_nodes.txt" \
      --prefix-plan "$prefix_plan" --prefix-steps "$refresh_steps" \
      --prefix-state-mode replay --device cuda --out "$OUT/$circuit/plans/$benchmark.csv" \
      2>&1 | sed -u "s|^|[r19-five-plan/$circuit] |" | tee "$OUT/$circuit/logs/plan.log" &
  pids+=("$!")
done
wait_all "${pids[@]}"

pids=()
for index in 1 2 3 4; do
  circuit=${circuits[$index]}
  benchmark=${benchmarks[$index]}
  echo "[r19-five-eval] start circuit=$circuit"
  /usr/bin/time -f '%e' -o "$OUT/$circuit/timing/final_eval_sec.txt" \
    python -u -m tpi_jepa.evaluate_plan_tmax \
      --benchmark-id "$benchmark" --plan-csv "$OUT/$circuit/plans/$benchmark.csv" \
      --out-dir "$OUT/$circuit/evals/final" --backend atalanta-bist \
      --patterns 300000 --seed 2026 --eval-step-mode final \
      2>&1 | sed -u "s|^|[r19-five-eval/$circuit] |" | tee "$OUT/$circuit/logs/final_eval.log" &
  pids+=("$!")
done
wait_all "${pids[@]}"

python scripts/materialize_residual_refresh_five.py \
  --root "$OUT" --checkpoint "$CHECKPOINT" --incumbent-root "$INCUMBENT" \
  --mapping-root "$MAPPING" --ratio-numerator "$RATIO_NUMERATOR" \
  --ratio-denominator "$RATIO_DENOMINATOR" --rounding floor \
  --prefix-state-mode replay --patterns 300000 --seed 2026 \
  --candidate-prior-alpha 0 --score-field typed_return_pred --max-candidates 72 \
  2>&1 | tee "$OUT/logs/materialize.log"
python scripts/summarize_exact_itc99_eval.py --eval-root "$OUT" \
  2>&1 | tee "$OUT/logs/summary.log"
python scripts/verify_uniform_exact_itc99.py "$OUT" \
  2>&1 | tee "$OUT/logs/uniform_audit.log"
python scripts/check_deeptpi_goal.py "$OUT" \
  2>&1 | tee "$OUT/logs/goal_units.log"
