#!/usr/bin/env bash
set -euo pipefail

BASE=autoresearch/loop-260720-0945/model_training_round5
OUT="$BASE/b15_selection/anchored_ensemble"
OLD=autoresearch/loop-260720-0945/model_training_round4/runs/onpolicy_balanced/best_final_horizon.pt
mkdir -p "$OUT/logs"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 TPI_TORCH_DETERMINISTIC=1
export TPI_HARD_CLUSTER_MAX_HARD_NODES=1024 TPI_LATENT_NORM_CLIP_RATIO=4
export TPI_Q_CONTEXT_SUPPORT_ALPHA=0.45 TPI_Q_CONTEXT_DISAGREEMENT_BETA=0.10
export TPI_TYPED_RESIDUAL_ALPHA=0.10 TPI_TYPED_RESIDUAL_DECAY_STEPS=16
export TPI_TYPED_RESIDUAL_CLIP=1.0 TPI_TYPED_RESIDUAL_DISAGREEMENT_BETA=0.25
export TPI_SCORE_QUANTIZATION=0.001 TPI_PLAN_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

models=(
  "prefix_cql_sa best_final_horizon"
  "prefix_cql_sa epoch_012"
  "prefix_toplist_sa best_final_horizon"
  "prefix_toplist_sa epoch_012"
)
old_copies=(1 2 3)
IFS=',' read -r -a gpus <<< "${GPUS_CSV:-7,7}"
MAX_PARALLEL=${MAX_PARALLEL:-${#gpus[@]}}
if (( ${#gpus[@]} == 0 || MAX_PARALLEL < 1 )); then
  echo "GPUS_CSV must contain at least one GPU and MAX_PARALLEL must be positive" >&2
  exit 2
fi

run_one() {
  local variant=$1 checkpoint_tag=$2 copies=$3 gpu=$4
  local new="$BASE/runs/$variant/$checkpoint_tag.pt"
  local ensemble=""
  local index
  for ((index=0; index<copies; index++)); do
    ensemble+="${ensemble:+,}$OLD"
  done
  ensemble+=",$new"
  local tag="old${copies}_new1"
  local out="$OUT/$variant/$checkpoint_tag/$tag"
  echo "[b15-anchor] start variant=$variant checkpoint=$checkpoint_tag mix=$tag gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/run_gmean_sweep.py \
    --eval-protocol configs/eval_protocol_coverage_only.json --protocol-keep-cli-benchmarks \
    --benchmarks iscas99__b15_1 --checkpoint "$OLD" \
    --ensemble-checkpoints "$ensemble" --ensemble-lcb-alpha 0 \
    --planners greedy --score-fields q_typed_residual_context --beam-objectives cumulative \
    --beam-widths 1 --lookahead-depths 1 --max-candidates 48 --discount-gammas 0.9 \
    --candidate-strategies hard_fault_cluster --candidate-diversity-penalties 0.0 \
    --candidate-diversity-depths 4 \
    --candidate-real-fault-priors autoresearch/eval8-real-priors-budget5-v1/real_fault_priors.csv \
    --candidate-allowlist autoresearch/original-netlist-recovery-260712/exact_itc99/b15_C/exact_candidate_nodes.txt \
    --plan-device cuda --time-limit-hours 24 --stream-logs --out-dir "$out" \
    2>&1 | sed -u "s|^|[b15-anchor/$variant/$checkpoint_tag/$tag] |" \
    | tee "$OUT/logs/${variant}__${checkpoint_tag}__${tag}.log"
}

pids=(); failed=0; job=0
wait_batch() {
  local pid
  for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
  pids=()
}
for model in "${models[@]}"; do
  read -r variant checkpoint_tag <<< "$model"
  for copies in "${old_copies[@]}"; do
    gpu=${gpus[$((job % ${#gpus[@]}))]}
    run_one "$variant" "$checkpoint_tag" "$copies" "$gpu" &
    pids+=("$!")
    job=$((job+1))
    if (( ${#pids[@]} == MAX_PARALLEL )); then wait_batch; fi
  done
done
if (( ${#pids[@]} )); then wait_batch; fi
if (( failed )); then exit 1; fi
echo "[b15-anchor] complete jobs=$job"
