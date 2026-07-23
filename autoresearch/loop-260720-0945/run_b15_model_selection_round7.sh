#!/usr/bin/env bash
set -euo pipefail

BASE=autoresearch/loop-260720-0945/model_training_round7
OUT="$BASE/b15_selection"
mkdir -p "$OUT/logs"

export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 TPI_TORCH_DETERMINISTIC=1
export TPI_HARD_CLUSTER_MAX_HARD_NODES=1024 TPI_LATENT_NORM_CLIP_RATIO=4
export TPI_Q_CONTEXT_SUPPORT_ALPHA=0.45 TPI_Q_CONTEXT_DISAGREEMENT_BETA=0.10
export TPI_TYPED_RESIDUAL_CLIP=1.0 TPI_TYPED_RESIDUAL_DISAGREEMENT_BETA=0.25
export TPI_TYPED_RELIABLE_MIN_HEADS=1 TPI_TYPED_RELIABLE_CP0_MIN_HEADS=2
export TPI_TYPED_TRUST_HEAD_MARGIN=0 TPI_TYPED_TRUST_ADVANTAGE_MARGIN=0
export TPI_SCORE_QUANTIZATION=0.001 TPI_PLAN_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TPI_ADAPTIVE_BASE_CANDIDATES=48 TPI_ADAPTIVE_MARGIN_MODE=relative_range
export TPI_ADAPTIVE_EXPANSION_MARGIN=0.003

variants=(cone_long_rank cone_long_toplist)
checkpoints=(best_final_horizon epoch_010)
# score alpha marginal_weight
settings=(
  "q_typed_residual_context 0.05 0.75"
  "q_typed_residual_context 0.10 0.75"
  "q_typed_residual_context 0.20 0.75"
  "q_typed_reliable_context 0.05 0.75"
  "q_typed_reliable_context 0.10 0.75"
  "q_typed_reliable_context 0.20 0.75"
  "q_typed_reliable_context 0.10 1.00"
)

IFS=',' read -r -a gpus <<< "${GPUS_CSV:-0,1,2,4,5,6}"
MAX_PARALLEL=${MAX_PARALLEL:-${#gpus[@]}}
if (( ${#gpus[@]} == 0 || MAX_PARALLEL < 1 )); then
  echo "GPUS_CSV must contain at least one GPU and MAX_PARALLEL must be positive" >&2
  exit 2
fi

run_one() {
  local variant=$1 checkpoint_tag=$2 score=$3 alpha=$4 weight=$5 gpu=$6
  local alpha_tag=${alpha/./p} weight_tag=${weight/./p}
  local checkpoint="$BASE/runs/$variant/$checkpoint_tag.pt"
  local out="$OUT/$variant/$checkpoint_tag/$score/alpha_${alpha_tag}_weight_${weight_tag}"
  echo "[b15-r7] start variant=$variant checkpoint=$checkpoint_tag score=$score alpha=$alpha weight=$weight gpu=$gpu"
  TPI_TYPED_RESIDUAL_ALPHA="$alpha" TPI_TYPED_RESIDUAL_DECAY_STEPS=16 \
  TPI_TYPED_RELIABLE_MARGINAL_WEIGHT="$weight" CUDA_VISIBLE_DEVICES="$gpu" \
    python -u scripts/run_gmean_sweep.py \
      --eval-protocol configs/eval_protocol_coverage_only.json --protocol-keep-cli-benchmarks \
      --benchmarks iscas99__b15_1 --checkpoint "$checkpoint" \
      --planners greedy --score-fields "$score" --beam-objectives cumulative \
      --beam-widths 1 --lookahead-depths 1 --max-candidates 64 --discount-gammas 0.9 \
      --candidate-strategies hard_fault_cluster --candidate-diversity-penalties 0.0 \
      --candidate-diversity-depths 4 \
      --candidate-real-fault-priors autoresearch/eval8-real-priors-budget5-v1/real_fault_priors.csv \
      --candidate-allowlist autoresearch/original-netlist-recovery-260712/exact_itc99/b15_C/exact_candidate_nodes.txt \
      --plan-device cuda --time-limit-hours 24 --stream-logs --out-dir "$out" \
      2>&1 | sed -u "s|^|[b15-r7/$variant/$checkpoint_tag/$score/a$alpha/w$weight] |" \
      | tee "$OUT/logs/${variant}__${checkpoint_tag}__${score}__a${alpha}__w${weight}.log"
}

pids=(); failed=0; job=0
wait_batch() {
  local pid
  for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
  pids=()
}
for variant in "${variants[@]}"; do
  for checkpoint_tag in "${checkpoints[@]}"; do
    for setting in "${settings[@]}"; do
      read -r score alpha weight <<< "$setting"
      gpu=${gpus[$((job % ${#gpus[@]}))]}
      run_one "$variant" "$checkpoint_tag" "$score" "$alpha" "$weight" "$gpu" &
      pids+=("$!")
      job=$((job+1))
      if (( ${#pids[@]} == MAX_PARALLEL )); then wait_batch; fi
    done
  done
done
if (( ${#pids[@]} )); then wait_batch; fi
if (( failed )); then exit 1; fi

python scripts/select_onpolicy_typed_on_b15.py --base "$BASE" \
  --incumbent-manifest autoresearch/loop-260720-0945/model_training_round6_adaptive/b15_selection/winner.json

