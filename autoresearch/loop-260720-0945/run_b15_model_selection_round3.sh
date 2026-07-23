#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
BASE=autoresearch/loop-260720-0945/model_training_round3
OUT="$BASE/b15_selection"
mkdir -p "$OUT/logs"

export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 TPI_TORCH_DETERMINISTIC=1
export TPI_HARD_CLUSTER_MAX_HARD_NODES=1024 TPI_LATENT_NORM_CLIP_RATIO=4
export TPI_Q_CONTEXT_SUPPORT_ALPHA=0.45 TPI_Q_CONTEXT_DISAGREEMENT_BETA=0.10
export TPI_TYPED_RESIDUAL_ALPHA=0.10 TPI_TYPED_RESIDUAL_CLIP=1.0
export TPI_TYPED_RESIDUAL_DISAGREEMENT_BETA=0.25 TPI_SCORE_QUANTIZATION=0.001
export TPI_PLAN_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

variants=(long32_balanced long32_return long32_marginal)
decays=(0 32 64 128)

run_one() {
  local variant=$1 decay=$2 gpu=$3
  local checkpoint="$BASE/runs/$variant/best_final_horizon.pt"
  local tag="decay_$decay"
  local out_dir="$OUT/$variant/$tag"
  test -f "$checkpoint"
  echo "[b15-round3] start variant=$variant decay=$decay gpu=$gpu"
  TPI_TYPED_RESIDUAL_DECAY_STEPS="$decay" CUDA_VISIBLE_DEVICES="$gpu" \
    python -u scripts/run_gmean_sweep.py \
      --eval-protocol configs/eval_protocol_coverage_only.json --protocol-keep-cli-benchmarks \
      --benchmarks iscas99__b15_1 --checkpoint "$checkpoint" \
      --planners greedy --score-fields q_typed_residual_context \
      --beam-objectives cumulative --beam-widths 1 --lookahead-depths 1 \
      --max-candidates 48 --discount-gammas 0.9 \
      --candidate-strategies hard_fault_cluster --candidate-diversity-penalties 0.0 \
      --candidate-diversity-depths 4 \
      --candidate-real-fault-priors autoresearch/eval8-real-priors-budget5-v1/real_fault_priors.csv \
      --candidate-allowlist autoresearch/original-netlist-recovery-260712/exact_itc99/b15_C/exact_candidate_nodes.txt \
      --plan-device cuda --time-limit-hours 24 --stream-logs --out-dir "$out_dir" \
      2>&1 | sed -u "s|^|[b15-round3/$variant/$tag] |" | tee "$OUT/logs/${variant}__${tag}.log"
  echo "[b15-round3] done variant=$variant decay=$decay gpu=$gpu"
}

pids=()
failed=0
job=0
wait_batch() {
  local pid
  for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
  pids=()
}
for variant in "${variants[@]}"; do
  for decay in "${decays[@]}"; do
    gpu=$((job % 2 + 1))
    run_one "$variant" "$decay" "$gpu" &
    pids+=("$!")
    job=$((job + 1))
    if (( ${#pids[@]} == 6 )); then wait_batch; fi
  done
done
if (( ${#pids[@]} )); then wait_batch; fi
if (( failed )); then exit 1; fi
python scripts/select_long_typed_on_b15.py
