#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
ROUND9="autoresearch/loop-260718-0223/structural_rollout_round9"
OUT_ROOT="$ROUND9/b15_head_selection"
mkdir -p "$OUT_ROOT/logs"
export TPI_HARD_CLUSTER_MAX_HARD_NODES=1024 TPI_LATENT_NORM_CLIP_RATIO=0
export TPI_Q_CONTEXT_SUPPORT_ALPHA=0.35 TPI_Q_CONTEXT_DISAGREEMENT_BETA=0.15
export TPI_PLAN_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

variants=(h16_strong h32_strong h32_weak h32_jepa)
scores=(
    reward_pred q_pred q_pred_context reward_pred_context
    guarded_reward_context bounded_residual_hybrid_pred_context
)
gpus=(0 1 2 4 5 6)

run_one() {
    local variant="$1" score="$2" gpu="$3"
    local checkpoint="$ROUND9/runs/$variant/best_final_horizon.pt"
    local tag="${variant}__${score}"
    local out="$OUT_ROOT/$tag/b15_C"
    local log="$OUT_ROOT/logs/$tag.log"
    echo "[round9-head] start variant=$variant score=$score gpu=$gpu"
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_gmean_sweep.py \
        --eval-protocol configs/eval_protocol_coverage_only.json --protocol-keep-cli-benchmarks \
        --benchmarks iscas99__b15_1 --checkpoint "$checkpoint" \
        --planners greedy --score-fields "$score" --beam-objectives cumulative \
        --beam-widths 1 --lookahead-depths 1 --max-candidates 48 --discount-gammas 0.9 \
        --candidate-strategies hard_fault_cluster --candidate-diversity-penalties 0.0 \
        --candidate-diversity-depths 4 \
        --candidate-real-fault-priors autoresearch/eval8-real-priors-budget5-v1/real_fault_priors.csv \
        --candidate-allowlist autoresearch/original-netlist-recovery-260712/exact_itc99/b15_C/exact_candidate_nodes.txt \
        --plan-device cuda --time-limit-hours 72 --out-dir "$out" 2>&1 \
        | sed -u "s|^|[round9-head/$tag] |" | tee "$log"
    echo "[round9-head] done variant=$variant score=$score gpu=$gpu"
}

pids=()
labels=()
job=0
for variant in "${variants[@]}"; do
    for score in "${scores[@]}"; do
        gpu="${gpus[$((job % ${#gpus[@]}))]}"
        run_one "$variant" "$score" "$gpu" &
        pids+=("$!")
        labels+=("$variant/$score")
        job=$((job + 1))
    done
done

failed=0
for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
        echo "[round9-head] failed job=${labels[$index]}"
        failed=1
    fi
done
exit "$failed"
