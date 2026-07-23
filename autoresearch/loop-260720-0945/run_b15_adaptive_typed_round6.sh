#!/usr/bin/env bash
set -euo pipefail

# Test one global confidence-gated candidate-width policy on b15 only.  The
# selected policy can later be replayed unchanged on all five exact netlists.
BASE=autoresearch/loop-260720-0945/model_training_round6_adaptive
OUT="$BASE/b15_selection"
mkdir -p "$OUT/logs"

export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 TPI_TORCH_DETERMINISTIC=1
export TPI_HARD_CLUSTER_MAX_HARD_NODES=1024 TPI_LATENT_NORM_CLIP_RATIO=4
export TPI_Q_CONTEXT_SUPPORT_ALPHA=0.45 TPI_Q_CONTEXT_DISAGREEMENT_BETA=0.10
export TPI_TYPED_RESIDUAL_CLIP=1.0 TPI_SCORE_QUANTIZATION=0.001
export TPI_PLAN_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TPI_TYPED_TRUST_MIN_HEADS=2 TPI_TYPED_TRUST_CP0_MIN_HEADS=3
export TPI_TYPED_TRUST_HEAD_MARGIN=0 TPI_TYPED_TRUST_ADVANTAGE_MARGIN=0
export TPI_TYPED_RELIABLE_MIN_HEADS=1 TPI_TYPED_RELIABLE_CP0_MIN_HEADS=2

# tag checkpoint score alpha decay marginal_weight disagreement
models=(
  "incumbent autoresearch/loop-260720-0945/model_training_round4/runs/onpolicy_balanced/best_final_horizon.pt q_typed_residual_context 0.10 16 0.75 0.25"
  "cone_toplist autoresearch/loop-260720-0945/model_training_round6/runs/cone_toplist/best_final_horizon.pt q_typed_residual_context 0.20 16 0.75 0.25"
  "cone_toplist_sa autoresearch/loop-260720-0945/model_training_round6/runs/cone_toplist_sa/epoch_012.pt q_typed_reliable_context 0.10 16 0.75 0.25"
)
# max_candidates margin_mode margin.  Every policy keeps the trusted c48
# prefix and expands only on a normalized near-tie.
policies=(
  "64 relative_range 0.0005"
  "64 relative_range 0.001"
  "64 relative_range 0.003"
  "64 relative_range 0.005"
  "64 relative_iqr 0.002"
  "64 relative_iqr 0.005"
  "80 relative_range 0.001"
)

IFS=',' read -r -a gpus <<< "${GPUS_CSV:-0,1,2,4,5,6}"
MAX_PARALLEL=${MAX_PARALLEL:-${#gpus[@]}}
if (( ${#gpus[@]} == 0 || MAX_PARALLEL < 1 )); then
  echo "GPUS_CSV must contain at least one GPU and MAX_PARALLEL must be positive" >&2
  exit 2
fi

run_one() {
  local tag=$1 checkpoint=$2 score=$3 alpha=$4 decay=$5 weight=$6 disagreement=$7
  local max_candidates=$8 margin_mode=$9 margin=${10} gpu=${11}
  local margin_tag=${margin/./p}
  local mode_tag=${margin_mode#relative_}
  local out="$OUT/$tag/c${max_candidates}_${mode_tag}_m${margin_tag}"
  echo "[b15-r6-adaptive] start model=$tag c=$max_candidates mode=$margin_mode margin=$margin gpu=$gpu"
  TPI_TYPED_RESIDUAL_ALPHA="$alpha" TPI_TYPED_RESIDUAL_DECAY_STEPS="$decay" \
  TPI_TYPED_RELIABLE_MARGINAL_WEIGHT="$weight" \
  TPI_TYPED_RESIDUAL_DISAGREEMENT_BETA="$disagreement" \
  TPI_ADAPTIVE_BASE_CANDIDATES=48 TPI_ADAPTIVE_MARGIN_MODE="$margin_mode" \
  TPI_ADAPTIVE_EXPANSION_MARGIN="$margin" CUDA_VISIBLE_DEVICES="$gpu" \
    python -u scripts/run_gmean_sweep.py \
      --eval-protocol configs/eval_protocol_coverage_only.json --protocol-keep-cli-benchmarks \
      --benchmarks iscas99__b15_1 --checkpoint "$checkpoint" \
      --planners greedy --score-fields "$score" --beam-objectives cumulative \
      --beam-widths 1 --lookahead-depths 1 --max-candidates "$max_candidates" --discount-gammas 0.9 \
      --candidate-strategies hard_fault_cluster --candidate-diversity-penalties 0.0 \
      --candidate-diversity-depths 4 \
      --candidate-real-fault-priors autoresearch/eval8-real-priors-budget5-v1/real_fault_priors.csv \
      --candidate-allowlist autoresearch/original-netlist-recovery-260712/exact_itc99/b15_C/exact_candidate_nodes.txt \
      --plan-device cuda --time-limit-hours 24 --stream-logs --out-dir "$out" \
      2>&1 | sed -u "s|^|[b15-r6-adaptive/$tag/c$max_candidates/$mode_tag/$margin] |" \
      | tee "$OUT/logs/${tag}__c${max_candidates}__${mode_tag}__m${margin}.log"
}

pids=(); failed=0; job=0
wait_batch() {
  local pid
  for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
  pids=()
}
for model in "${models[@]}"; do
  read -r tag checkpoint score alpha decay weight disagreement <<< "$model"
  for policy in "${policies[@]}"; do
    read -r max_candidates margin_mode margin <<< "$policy"
    gpu=${gpus[$((job % ${#gpus[@]}))]}
    run_one "$tag" "$checkpoint" "$score" "$alpha" "$decay" "$weight" "$disagreement" \
      "$max_candidates" "$margin_mode" "$margin" "$gpu" &
    pids+=("$!")
    job=$((job+1))
    if (( ${#pids[@]} == MAX_PARALLEL )); then wait_batch; fi
  done
done
if (( ${#pids[@]} )); then wait_batch; fi
if (( failed )); then exit 1; fi

python scripts/select_onpolicy_typed_on_b15.py --base "$BASE" \
  --incumbent-manifest autoresearch/loop-260720-0945/model_training_round6/b15_selection/winner.json

