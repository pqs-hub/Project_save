#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT_ROOT="autoresearch/loop-260718-0223"
PROTOCOL="configs/eval_protocol_coverage_only.json"
CHECKPOINT="runs/planner_aligned_q_rank_v5_context_safe/best_final_horizon.pt"
PRIOR="autoresearch/eval8-real-priors-budget5-v1/real_fault_priors.csv"
MAPPING_ROOT="autoresearch/original-netlist-recovery-260712/exact_itc99"
mkdir -p "$OUT_ROOT/logs"

export TPI_HARD_CLUSTER_MAX_HARD_NODES=1024
export TPI_PLAN_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

circuits=(b15_C b20_C b21_C b22_C b17_C)
benchmarks=(iscas99__b15_1 iscas99__b20 iscas99__b21 iscas99__b22 iscas99__b17)
variants=(v5_q_context v5_q_type_context v5_consensus_type_context)
scores=(q_pred_context q_pred_type_context consensus_pred_type_context)
gpus=(0 1 2 4 5 6)

run_one() {
    local variant="$1" score="$2" circuit="$3" benchmark="$4" gpu="$5"
    local out_dir="$OUT_ROOT/$variant/$circuit"
    local log="$OUT_ROOT/logs/${variant}__${circuit}.log"
    echo "[explore] start variant=$variant circuit=$circuit gpu=$gpu score=$score"
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_gmean_sweep.py \
        --eval-protocol "$PROTOCOL" \
        --protocol-keep-cli-benchmarks \
        --benchmarks "$benchmark" \
        --checkpoint "$CHECKPOINT" \
        --planners greedy \
        --score-fields "$score" \
        --beam-objectives cumulative \
        --beam-widths 1 \
        --lookahead-depths 1 \
        --max-candidates 48 \
        --discount-gammas 0.9 \
        --candidate-strategies hard_fault_cluster \
        --candidate-diversity-penalties 0.0 \
        --candidate-diversity-depths 4 \
        --candidate-real-fault-priors "$PRIOR" \
        --candidate-allowlist "$MAPPING_ROOT/$circuit/exact_candidate_nodes.txt" \
        --plan-device cuda \
        --time-limit-hours 72 \
        --out-dir "$out_dir" 2>&1 \
        | sed -u "s|^|[$variant/$circuit] |" \
        | tee "$log"
    echo "[explore] done variant=$variant circuit=$circuit gpu=$gpu"
}

pids=()
labels=()
job=0
for variant_index in "${!variants[@]}"; do
    for circuit_index in "${!circuits[@]}"; do
        variant="${variants[$variant_index]}"
        score="${scores[$variant_index]}"
        circuit="${circuits[$circuit_index]}"
        benchmark="${benchmarks[$circuit_index]}"
        gpu="${gpus[$((job % ${#gpus[@]}))]}"
        run_one "$variant" "$score" "$circuit" "$benchmark" "$gpu" &
        pids+=("$!")
        labels+=("$variant/$circuit")
        job=$((job + 1))
    done
done

failed=0
for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
        echo "[explore] failed job=${labels[$index]}"
        failed=1
    fi
done

if (( failed )); then
    exit 1
fi

for variant in "${variants[@]}"; do
    python scripts/summarize_exact_itc99_eval.py --eval-root "$OUT_ROOT/$variant"
    score="$(python scripts/score_exact_itc99_vs_deeptpi.py "$OUT_ROOT/$variant/summary.json")"
    echo "[explore] summary variant=$variant metric=$score"
done

echo "[explore] all 15 jobs completed"
