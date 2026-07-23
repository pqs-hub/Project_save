#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
OUT_ROOT="autoresearch/loop-260718-0223/quantized_repeats_b15_round13"
mkdir -p "$OUT_ROOT/logs"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TPI_PLAN_THREADS=1
export TPI_HARD_CLUSTER_MAX_HARD_NODES=1024
export TPI_LATENT_NORM_CLIP_RATIO=4
export TPI_Q_CONTEXT_SUPPORT_ALPHA=0.45
export TPI_Q_CONTEXT_DISAGREEMENT_BETA=0.10

quantums=(0.00001 0.00001 0.0001 0.0001 0.001 0.001)
penalties=(0.0 0.8 0.0 0.8 0.0 0.8)
depths=(4 8 4 8 4 8)
gpus=(0 1 2 4 5 6)

run_one() {
    local method="$1" replica="$2" quantum="$3" penalty="$4" depth="$5" gpu="$6"
    local tag="${method}__r${replica}"
    local out="$OUT_ROOT/$tag/b15_C"
    local log="$OUT_ROOT/logs/${tag}__b15_C.log"
    echo "[round13] start tag=$tag gpu=$gpu quantum=$quantum penalty=$penalty depth=$depth"
    TPI_SCORE_QUANTIZATION="$quantum" CUDA_VISIBLE_DEVICES="$gpu" \
    python scripts/run_gmean_sweep.py \
        --eval-protocol configs/eval_protocol_coverage_only.json --protocol-keep-cli-benchmarks \
        --benchmarks iscas99__b15_1 \
        --checkpoint runs/planner_aligned_q_rank_v5_context_safe/best_final_horizon.pt \
        --planners greedy --score-fields q_pred_context --beam-objectives cumulative \
        --beam-widths 1 --lookahead-depths 1 --max-candidates 48 --discount-gammas 0.9 \
        --candidate-strategies hard_fault_cluster \
        --candidate-diversity-penalties "$penalty" --candidate-diversity-depths "$depth" \
        --candidate-real-fault-priors autoresearch/eval8-real-priors-budget5-v1/real_fault_priors.csv \
        --candidate-allowlist autoresearch/original-netlist-recovery-260712/exact_itc99/b15_C/exact_candidate_nodes.txt \
        --plan-device cuda --time-limit-hours 72 --out-dir "$out" 2>&1 \
        | sed -u "s|^|[round13/$tag] |" | tee "$log"
    echo "[round13] done tag=$tag gpu=$gpu"
}

pids=()
tags=()
job=0
for index in "${!quantums[@]}"; do
    qtag="${quantums[$index]//./p}"
    ptag="${penalties[$index]//./p}"
    method="q${qtag}_p${ptag}_d${depths[$index]}"
    for replica in 1 2; do
        tags+=("${method}__r${replica}")
        run_one "$method" "$replica" "${quantums[$index]}" \
            "${penalties[$index]}" "${depths[$index]}" \
            "${gpus[$((job % ${#gpus[@]}))]}" &
        pids+=("$!")
        job=$((job + 1))
    done
done

failed=0
for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
        echo "[round13] failed tag=${tags[$index]}"
        failed=1
    fi
done
if (( failed )); then exit 1; fi

for tag in "${tags[@]}"; do
    result="$OUT_ROOT/$tag/b15_C/best.json"
    plan=$(find "$OUT_ROOT/$tag/b15_C/best_plans" -name '*.csv' -print -quit)
    tc=$(jq -r '.macro_mean_delta_tc' "$result")
    digest=$(sha256sum "$plan" | cut -d' ' -f1)
    printf '%s\t%s\t%s\n' "$tag" "$tc" "$digest"
done | sort | tee "$OUT_ROOT/repeats.tsv"
