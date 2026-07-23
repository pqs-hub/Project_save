#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
OUT_ROOT="autoresearch/loop-260718-0223/structural_rollout_round9"
LABELS="autoresearch/loop-260718-0223/structural_rollout64/labels.csv"
INIT="runs/planner_aligned_q_rank_v5_context_safe/best_final_horizon.pt"
mkdir -p "$OUT_ROOT/configs" "$OUT_ROOT/logs" "$OUT_ROOT/evals"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TPI_PLAN_THREADS=1

variants=(h16_strong h32_strong h32_weak h32_jepa)
horizons=(16 32 32 32)
start_horizons=(4 8 8 8)
increments=(4 8 8 8)
q_ranks=(0.38 0.38 0.10 0.0)
q_values=(0.08 0.08 0.02 0.0)
candidate_weights=(0.18 0.18 0.05 0.0)
context_weights=(0.25 0.25 0.05 0.0)
gpus=(0 2 4 5)

build_config() {
    local index="$1" variant="${variants[$1]}"
    local config="$OUT_ROOT/configs/$variant.json"
    jq \
        --arg labels "$LABELS" \
        --arg run_dir "$OUT_ROOT/runs/$variant" \
        --arg init "$INIT" \
        --argjson horizon "${horizons[$index]}" \
        --argjson start_horizon "${start_horizons[$index]}" \
        --argjson increment "${increments[$index]}" \
        --argjson q_rank "${q_ranks[$index]}" \
        --argjson q_value "${q_values[$index]}" \
        --argjson candidate_weight "${candidate_weights[$index]}" \
        --argjson context_weight "${context_weights[$index]}" \
        '.labels=$labels
         | .run_dir=$run_dir
         | .init_checkpoint=$init
         | .init_checkpoint_strict=true
         | .epochs=4
         | .lr=0.00005
         | .max_train_samples=8000
         | .max_train_steps_per_epoch=120
         | .max_val_samples=2048
         | .max_val_steps=32
         | .lambda_jepa=1.0
         | .lambda_scoap=0.0
         | .lambda_delta_scoap=0.0
         | .lambda_hard=0.0
         | .lambda_hard_rank=0.0
         | .lambda_hard_brier=0.0
         | .lambda_hard_soft_f1=0.0
         | .lambda_hard_count=0.0
         | .lambda_hard_reduction=0.0
         | .lambda_fc=0.0
         | .lambda_pattern=0.0
         | .lambda_return=0.0
         | .lambda_q_rank=$q_rank
         | .lambda_q_value=$q_value
         | .lambda_candidate=$candidate_weight
         | .lambda_context_rank=$context_weight
         | .lambda_ndcg_rank=0.0
         | .lambda_conservative_q=0.0
         | .rollout_training=true
         | .rollout_max_horizon=$horizon
         | .rollout_start_epoch=1
         | .rollout_increase_every=1
         | .rollout_start_horizon=$start_horizon
         | .rollout_horizon_increment=$increment
         | .require_full_horizon=true
         | .repeat_train_samples=false
         | .device="cuda"' \
        configs/planner_aligned_q_rank_v5_context_safe.json > "$config"
}

train_one() {
    local index="$1" variant="${variants[$1]}" gpu="${gpus[$1]}"
    local config="$OUT_ROOT/configs/$variant.json"
    local log="$OUT_ROOT/logs/$variant.train.log"
    echo "[round9] train_start variant=$variant gpu=$gpu horizon=${horizons[$index]}"
    CUDA_VISIBLE_DEVICES="$gpu" python -m tpi_jepa.train --config "$config" 2>&1 \
        | sed -u "s|^|[round9/train/$variant] |" | tee "$log"
    echo "[round9] train_done variant=$variant gpu=$gpu"
}

eval_one() {
    local index="$1" variant="${variants[$1]}" gpu="${gpus[$1]}"
    local checkpoint="$OUT_ROOT/runs/$variant/best_final_horizon.pt"
    local out="$OUT_ROOT/evals/$variant/b15_C"
    local log="$OUT_ROOT/logs/$variant.b15.log"
    echo "[round9] b15_start variant=$variant gpu=$gpu"
    TPI_HARD_CLUSTER_MAX_HARD_NODES=1024 TPI_LATENT_NORM_CLIP_RATIO=0 \
    TPI_Q_CONTEXT_SUPPORT_ALPHA=0.35 TPI_Q_CONTEXT_DISAGREEMENT_BETA=0.15 \
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_gmean_sweep.py \
        --eval-protocol configs/eval_protocol_coverage_only.json --protocol-keep-cli-benchmarks \
        --benchmarks iscas99__b15_1 --checkpoint "$checkpoint" \
        --planners greedy --score-fields q_pred_context --beam-objectives cumulative \
        --beam-widths 1 --lookahead-depths 1 --max-candidates 48 --discount-gammas 0.9 \
        --candidate-strategies hard_fault_cluster --candidate-diversity-penalties 0.0 \
        --candidate-diversity-depths 4 \
        --candidate-real-fault-priors autoresearch/eval8-real-priors-budget5-v1/real_fault_priors.csv \
        --candidate-allowlist autoresearch/original-netlist-recovery-260712/exact_itc99/b15_C/exact_candidate_nodes.txt \
        --plan-device cuda --time-limit-hours 72 --out-dir "$out" 2>&1 \
        | sed -u "s|^|[round9/b15/$variant] |" | tee "$log"
    echo "[round9] b15_done variant=$variant gpu=$gpu"
}

for index in "${!variants[@]}"; do build_config "$index"; done

pids=()
for index in "${!variants[@]}"; do train_one "$index" & pids+=("$!"); done
failed=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
if (( failed )); then exit 1; fi

pids=()
for index in "${!variants[@]}"; do eval_one "$index" & pids+=("$!"); done
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
exit "$failed"
