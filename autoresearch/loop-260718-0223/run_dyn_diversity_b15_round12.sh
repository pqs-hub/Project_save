#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
OUT_ROOT="autoresearch/loop-260718-0223/dyn_diversity_b15_round12"
CHECKPOINT="autoresearch/loop-260718-0223/predictor_only_round10/runs/dyn_weak/best_final_horizon.pt"
mkdir -p "$OUT_ROOT/logs"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TPI_PLAN_THREADS=1
export TPI_HARD_CLUSTER_MAX_HARD_NODES=1024
export TPI_LATENT_NORM_CLIP_RATIO=0
export TPI_Q_CONTEXT_SUPPORT_ALPHA=0.35
export TPI_Q_CONTEXT_DISAGREEMENT_BETA=0.15

depths=(2 2 2 2 4 4 4 4 8 8 8 8)
penalties=(0.10 0.25 0.50 0.80 0.10 0.25 0.50 0.80 0.10 0.25 0.50 0.80)
gpus=(0 1 2 4 5 6)

run_one() {
    local depth="$1" penalty="$2" gpu="$3"
    local penalty_tag="${penalty/./p}"
    local variant="d${depth}_p${penalty_tag}"
    local out="$OUT_ROOT/$variant/b15_C"
    local log="$OUT_ROOT/logs/${variant}__b15_C.log"
    echo "[round12] start variant=$variant gpu=$gpu depth=$depth penalty=$penalty"
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_gmean_sweep.py \
        --eval-protocol configs/eval_protocol_coverage_only.json --protocol-keep-cli-benchmarks \
        --benchmarks iscas99__b15_1 --checkpoint "$CHECKPOINT" \
        --planners greedy --score-fields reward_pred --beam-objectives cumulative \
        --beam-widths 1 --lookahead-depths 1 --max-candidates 48 --discount-gammas 0.9 \
        --candidate-strategies hard_fault_cluster \
        --candidate-diversity-penalties "$penalty" \
        --candidate-diversity-depths "$depth" \
        --candidate-real-fault-priors autoresearch/eval8-real-priors-budget5-v1/real_fault_priors.csv \
        --candidate-allowlist autoresearch/original-netlist-recovery-260712/exact_itc99/b15_C/exact_candidate_nodes.txt \
        --plan-device cuda --time-limit-hours 72 --out-dir "$out" 2>&1 \
        | sed -u "s|^|[round12/$variant] |" | tee "$log"
    echo "[round12] done variant=$variant gpu=$gpu"
}

pids=()
labels=()
for index in "${!depths[@]}"; do
    penalty_tag="${penalties[$index]/./p}"
    labels+=("d${depths[$index]}_p${penalty_tag}")
    run_one "${depths[$index]}" "${penalties[$index]}" \
        "${gpus[$((index % ${#gpus[@]}))]}" &
    pids+=("$!")
done

failed=0
for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
        echo "[round12] failed variant=${labels[$index]}"
        failed=1
    fi
done
if (( failed )); then exit 1; fi

for variant in "${labels[@]}"; do
    result="$OUT_ROOT/$variant/b15_C/best.json"
    jq -r --arg variant "$variant" '[ $variant, .macro_mean_delta_tc ] | @tsv' "$result"
done | sort -t $'\t' -k2,2nr | tee "$OUT_ROOT/b15_selection.tsv"
