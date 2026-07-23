#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
OUT_ROOT="autoresearch/loop-260718-0223/qcontext_weight_b15_round8"
mkdir -p "$OUT_ROOT/logs"
export TPI_HARD_CLUSTER_MAX_HARD_NODES=1024
export TPI_LATENT_REENCODE_INTERVAL=0 TPI_PLAN_THREADS=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

variants=(
    a015_b000 a015_b010
    a025_b000 a025_b010 a025_b025
    a045_b000 a045_b010 a045_b025
    a055_b000 a055_b010
    a025_b010_clip4 a045_b010_clip4
)
alphas=(0.15 0.15 0.25 0.25 0.25 0.45 0.45 0.45 0.55 0.55 0.25 0.45)
betas=(0.00 0.10 0.00 0.10 0.25 0.00 0.10 0.25 0.00 0.10 0.10 0.10)
clips=(0 0 0 0 0 0 0 0 0 0 4 4)
gpus=(0 1 2 4 5 6)

run_one() {
    local variant="$1" alpha="$2" beta="$3" clip="$4" gpu="$5"
    local out="$OUT_ROOT/$variant/b15_C"
    local log="$OUT_ROOT/logs/${variant}__b15_C.log"
    echo "[round8-b15] start variant=$variant gpu=$gpu alpha=$alpha beta=$beta clip=$clip"
    TPI_Q_CONTEXT_SUPPORT_ALPHA="$alpha" \
    TPI_Q_CONTEXT_DISAGREEMENT_BETA="$beta" \
    TPI_LATENT_NORM_CLIP_RATIO="$clip" \
    CUDA_VISIBLE_DEVICES="$gpu" \
    python scripts/run_gmean_sweep.py \
        --eval-protocol configs/eval_protocol_coverage_only.json --protocol-keep-cli-benchmarks \
        --benchmarks iscas99__b15_1 \
        --checkpoint runs/planner_aligned_q_rank_v5_context_safe/best_final_horizon.pt \
        --planners greedy --score-fields q_pred_context --beam-objectives cumulative \
        --beam-widths 1 --lookahead-depths 1 --max-candidates 48 --discount-gammas 0.9 \
        --candidate-strategies hard_fault_cluster --candidate-diversity-penalties 0.0 \
        --candidate-diversity-depths 4 \
        --candidate-real-fault-priors autoresearch/eval8-real-priors-budget5-v1/real_fault_priors.csv \
        --candidate-allowlist autoresearch/original-netlist-recovery-260712/exact_itc99/b15_C/exact_candidate_nodes.txt \
        --plan-device cuda --time-limit-hours 72 --out-dir "$out" 2>&1 \
        | sed -u "s|^|[round8-b15/$variant] |" | tee "$log"
    echo "[round8-b15] done variant=$variant gpu=$gpu"
}

pids=()
for index in "${!variants[@]}"; do
    gpu="${gpus[$((index % ${#gpus[@]}))]}"
    run_one "${variants[$index]}" "${alphas[$index]}" "${betas[$index]}" \
        "${clips[$index]}" "$gpu" &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
done
exit "$failed"
