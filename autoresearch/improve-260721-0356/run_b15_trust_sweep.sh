#!/usr/bin/env bash
set -euo pipefail

ROOT=autoresearch/improve-260721-0356/b15_trust_selection
CHECKPOINT=autoresearch/loop-260720-0945/model_training_round4/runs/onpolicy_balanced/best_final_horizon.pt
mkdir -p "$ROOT/settings" "$ROOT/logs"

export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 TPI_TORCH_DETERMINISTIC=1
export TPI_HARD_CLUSTER_MAX_HARD_NODES=1024 TPI_LATENT_NORM_CLIP_RATIO=4
export TPI_Q_CONTEXT_SUPPORT_ALPHA=0.45 TPI_Q_CONTEXT_DISAGREEMENT_BETA=0.10
export TPI_TYPED_RESIDUAL_CLIP=1.0 TPI_TYPED_RESIDUAL_DISAGREEMENT_BETA=0.25
export TPI_SCORE_QUANTIZATION=0.001 TPI_PLAN_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# alpha decay regular_heads cp0_heads head_margin advantage_margin
settings=(
  "0.00 16 3 3 0.00 0.00"
  "0.05 16 2 3 0.00 0.00"
  "0.10 8 2 3 0.00 0.00"
  "0.10 16 2 3 0.00 0.00"
  "0.10 32 2 3 0.00 0.00"
  "0.20 16 2 3 0.00 0.00"
  "0.10 16 3 3 0.00 0.00"
  "0.10 16 2 3 0.10 0.05"
)
gpus=(0 1 2 4 5 6)

run_one() {
  local alpha=$1 decay=$2 heads=$3 cp0_heads=$4 head_margin=$5 advantage_margin=$6 gpu=$7
  local id="a${alpha/./p}_d${decay}_h${heads}_c${cp0_heads}_m${head_margin/./p}_v${advantage_margin/./p}"
  local out="$ROOT/settings/$id"
  echo "[b15-trust] start id=$id gpu=$gpu"
  TPI_TYPED_RESIDUAL_ALPHA="$alpha" \
  TPI_TYPED_RESIDUAL_DECAY_STEPS="$decay" \
  TPI_TYPED_TRUST_MIN_HEADS="$heads" \
  TPI_TYPED_TRUST_CP0_MIN_HEADS="$cp0_heads" \
  TPI_TYPED_TRUST_HEAD_MARGIN="$head_margin" \
  TPI_TYPED_TRUST_ADVANTAGE_MARGIN="$advantage_margin" \
  CUDA_VISIBLE_DEVICES="$gpu" \
    python -u scripts/run_gmean_sweep.py \
      --eval-protocol configs/eval_protocol_coverage_only.json --protocol-keep-cli-benchmarks \
      --benchmarks iscas99__b15_1 --checkpoint "$CHECKPOINT" \
      --planners greedy --score-fields q_typed_trust_context --beam-objectives cumulative \
      --beam-widths 1 --lookahead-depths 1 --max-candidates 48 --discount-gammas 0.9 \
      --candidate-strategies hard_fault_cluster --candidate-diversity-penalties 0.0 \
      --candidate-diversity-depths 4 \
      --candidate-real-fault-priors autoresearch/eval8-real-priors-budget5-v1/real_fault_priors.csv \
      --candidate-allowlist autoresearch/original-netlist-recovery-260712/exact_itc99/b15_C/exact_candidate_nodes.txt \
      --plan-device cuda --time-limit-hours 24 --stream-logs --out-dir "$out" \
      2>&1 | sed -u "s|^|[b15-trust/$id] |" | tee "$ROOT/logs/$id.log"
  echo "[b15-trust] done id=$id gpu=$gpu"
}

pids=()
failed=0
job=0
wait_batch() {
  local pid
  for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
  pids=()
}
for setting in "${settings[@]}"; do
  read -r alpha decay heads cp0_heads head_margin advantage_margin <<< "$setting"
  gpu=${gpus[$((job % ${#gpus[@]}))]}
  run_one "$alpha" "$decay" "$heads" "$cp0_heads" "$head_margin" "$advantage_margin" "$gpu" &
  pids+=("$!")
  job=$((job + 1))
  if (( ${#pids[@]} == ${#gpus[@]} )); then wait_batch; fi
done
if (( ${#pids[@]} )); then wait_batch; fi
if (( failed )); then exit 1; fi

python autoresearch/improve-260721-0356/select_trust_on_b15.py \
  --root "$ROOT" --checkpoint "$CHECKPOINT"

