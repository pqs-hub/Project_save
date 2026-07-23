#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
OUT_ROOT="autoresearch/loop-260718-0223/relative_gate_b15_round18"
mkdir -p "$OUT_ROOT/logs"

export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 TPI_TORCH_DETERMINISTIC=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TPI_PLAN_THREADS=1
export TPI_HARD_CLUSTER_MAX_HARD_NODES=1024 TPI_LATENT_NORM_CLIP_RATIO=4
export TPI_Q_CONTEXT_SUPPORT_ALPHA=0.45 TPI_Q_CONTEXT_DISAGREEMENT_BETA=0.10
export TPI_SCORE_QUANTIZATION=0.001 TPI_ADAPTIVE_BASE_CANDIDATES=48

modes=(relative_iqr relative_iqr relative_iqr relative_iqr relative_iqr relative_iqr relative_range relative_range relative_range relative_range relative_range relative_range)
margins=(0.001 0.002 0.005 0.01 0.02 0.05 0.001 0.002 0.005 0.01 0.02 0.05)
gpus=(0 1 2 4 5 6)

run_one() {
    local mode="$1" margin="$2" gpu="$3"
    local tag="${mode#relative_}_m${margin/./p}"
    local out="$OUT_ROOT/$tag/b15_C"
    echo "[round18] start variant=$tag gpu=$gpu"
    TPI_ADAPTIVE_MARGIN_MODE="$mode" TPI_ADAPTIVE_EXPANSION_MARGIN="$margin" \
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_gmean_sweep.py \
        --eval-protocol configs/eval_protocol_coverage_only.json --protocol-keep-cli-benchmarks \
        --benchmarks iscas99__b15_1 \
        --checkpoint runs/planner_aligned_q_rank_v5_context_safe/best_final_horizon.pt \
        --planners greedy --score-fields q_pred_context --beam-objectives cumulative \
        --beam-widths 1 --lookahead-depths 1 --max-candidates 64 --discount-gammas 0.9 \
        --candidate-strategies hard_fault_cluster --candidate-diversity-penalties 0.0 \
        --candidate-diversity-depths 4 \
        --candidate-real-fault-priors autoresearch/eval8-real-priors-budget5-v1/real_fault_priors.csv \
        --candidate-allowlist autoresearch/original-netlist-recovery-260712/exact_itc99/b15_C/exact_candidate_nodes.txt \
        --plan-device cuda --time-limit-hours 72 --out-dir "$out" 2>&1 \
        | sed -u "s|^|[round18/$tag] |" | tee "$OUT_ROOT/logs/$tag.log"
    echo "[round18] done variant=$tag"
}

pids=()
labels=()
for index in "${!modes[@]}"; do
    tag="${modes[$index]#relative_}_m${margins[$index]/./p}"
    labels+=("$tag")
    run_one "${modes[$index]}" "${margins[$index]}" "${gpus[$((index % ${#gpus[@]}))]}" &
    pids+=("$!")
done
failed=0
for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then echo "[round18] failed variant=${labels[$index]}"; failed=1; fi
done
if (( failed )); then exit 1; fi

for variant in "${labels[@]}"; do
    result="$OUT_ROOT/$variant/b15_C/best.json"
    plan=$(find "$OUT_ROOT/$variant/b15_C/best_plans" -name '*.csv' -print -quit)
    expansions=$(python -c 'import csv,sys; r=list(csv.DictReader(open(sys.argv[1]))); print(sum(x.get("adaptive_expanded","").lower()=="true" for x in r))' "$plan")
    jq -r --arg variant "$variant" --arg expansions "$expansions" \
        '[ $variant, .macro_mean_delta_tc, $expansions ] | @tsv' "$result"
done | sort -t $'\t' -k2,2nr | tee "$OUT_ROOT/b15_selection.tsv"
