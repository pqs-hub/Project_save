#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
OUT_ROOT="autoresearch/loop-260718-0223/bottleneck_round2"
mkdir -p "$OUT_ROOT/logs"
export TPI_PLAN_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

variants=(clip1 clip2 clip4 seeds1536 seeds2048 recall_union_c96)
clips=(1 2 4 0 0 0)
hard_seeds=(1024 1024 1024 1536 2048 1024)
candidates=(48 48 48 48 48 96)
strategies=(hard_fault_cluster hard_fault_cluster hard_fault_cluster hard_fault_cluster hard_fault_cluster hard_fault_recall_union)
circuits=(b22_C b17_C)
benchmarks=(iscas99__b22 iscas99__b17)
gpus=(1 2 4 5 6)

run_one() {
    local variant="$1" clip="$2" seeds="$3" count="$4" strategy="$5"
    local circuit="$6" benchmark="$7" gpu="$8"
    local out="$OUT_ROOT/$variant/$circuit"
    local log="$OUT_ROOT/logs/${variant}__${circuit}.log"
    echo "[round2] start variant=$variant circuit=$circuit gpu=$gpu clip=$clip seeds=$seeds"
    TPI_LATENT_NORM_CLIP_RATIO="$clip" TPI_HARD_CLUSTER_MAX_HARD_NODES="$seeds" \
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_gmean_sweep.py \
        --eval-protocol configs/eval_protocol_coverage_only.json --protocol-keep-cli-benchmarks \
        --benchmarks "$benchmark" \
        --checkpoint runs/planner_aligned_q_rank_v5_context_safe/best_final_horizon.pt \
        --planners greedy --score-fields q_pred_context --beam-objectives cumulative \
        --beam-widths 1 --lookahead-depths 1 --max-candidates "$count" --discount-gammas 0.9 \
        --candidate-strategies "$strategy" --candidate-diversity-penalties 0.0 \
        --candidate-diversity-depths 4 \
        --candidate-real-fault-priors autoresearch/eval8-real-priors-budget5-v1/real_fault_priors.csv \
        --candidate-allowlist "autoresearch/original-netlist-recovery-260712/exact_itc99/$circuit/exact_candidate_nodes.txt" \
        --plan-device cuda --time-limit-hours 72 --out-dir "$out" 2>&1 \
        | sed -u "s|^|[round2/$variant/$circuit] |" | tee "$log"
    echo "[round2] done variant=$variant circuit=$circuit gpu=$gpu"
}

pids=()
labels=()
job=0
for variant_index in "${!variants[@]}"; do
    for circuit_index in "${!circuits[@]}"; do
        gpu="${gpus[$((job % ${#gpus[@]}))]}"
        run_one "${variants[$variant_index]}" "${clips[$variant_index]}" \
            "${hard_seeds[$variant_index]}" "${candidates[$variant_index]}" \
            "${strategies[$variant_index]}" "${circuits[$circuit_index]}" \
            "${benchmarks[$circuit_index]}" "$gpu" &
        pids+=("$!")
        labels+=("${variants[$variant_index]}/${circuits[$circuit_index]}")
        job=$((job + 1))
    done
done

failed=0
for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
        echo "[round2] failed job=${labels[$index]}"
        failed=1
    fi
done
if (( failed )); then exit 1; fi
python scripts/summarize_bottleneck_search.py "$OUT_ROOT"

