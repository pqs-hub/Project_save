#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
OUT_ROOT="autoresearch/loop-260718-0223/stable_recall_b15_round15"
mkdir -p "$OUT_ROOT/logs"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TPI_PLAN_THREADS=1
export TPI_LATENT_NORM_CLIP_RATIO=4
export TPI_Q_CONTEXT_SUPPORT_ALPHA=0.45
export TPI_Q_CONTEXT_DISAGREEMENT_BETA=0.10
export TPI_SCORE_QUANTIZATION=0.001

seeds=(512 512 512 1024 1024 1024 1536 1536 1536 2048 2048 2048)
widths=(32 48 64 32 48 64 32 48 64 32 48 64)
gpus=(0 1 2 4 5 6)

run_one() {
    local seeds_count="$1" width="$2" gpu="$3"
    local variant="s${seeds_count}_c${width}"
    local out="$OUT_ROOT/$variant/b15_C"
    local log="$OUT_ROOT/logs/${variant}__b15_C.log"
    echo "[round15] start variant=$variant gpu=$gpu"
    TPI_HARD_CLUSTER_MAX_HARD_NODES="$seeds_count" CUDA_VISIBLE_DEVICES="$gpu" \
    python scripts/run_gmean_sweep.py \
        --eval-protocol configs/eval_protocol_coverage_only.json --protocol-keep-cli-benchmarks \
        --benchmarks iscas99__b15_1 \
        --checkpoint runs/planner_aligned_q_rank_v5_context_safe/best_final_horizon.pt \
        --planners greedy --score-fields q_pred_context --beam-objectives cumulative \
        --beam-widths 1 --lookahead-depths 1 --max-candidates "$width" --discount-gammas 0.9 \
        --candidate-strategies hard_fault_cluster --candidate-diversity-penalties 0.0 \
        --candidate-diversity-depths 4 \
        --candidate-real-fault-priors autoresearch/eval8-real-priors-budget5-v1/real_fault_priors.csv \
        --candidate-allowlist autoresearch/original-netlist-recovery-260712/exact_itc99/b15_C/exact_candidate_nodes.txt \
        --plan-device cuda --time-limit-hours 72 --out-dir "$out" 2>&1 \
        | sed -u "s|^|[round15/$variant] |" | tee "$log"
    echo "[round15] done variant=$variant gpu=$gpu"
}

pids=()
labels=()
for index in "${!seeds[@]}"; do
    labels+=("s${seeds[$index]}_c${widths[$index]}")
    run_one "${seeds[$index]}" "${widths[$index]}" \
        "${gpus[$((index % ${#gpus[@]}))]}" &
    pids+=("$!")
done

failed=0
for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
        echo "[round15] failed variant=${labels[$index]}"
        failed=1
    fi
done
if (( failed )); then exit 1; fi

for variant in "${labels[@]}"; do
    jq -r --arg variant "$variant" '[ $variant, .macro_mean_delta_tc ] | @tsv' \
        "$OUT_ROOT/$variant/b15_C/best.json"
done | sort -t $'\t' -k2,2nr | tee "$OUT_ROOT/b15_selection.tsv"
