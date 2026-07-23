#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
OUT_ROOT="autoresearch/loop-260718-0223/reencode_round6"
mkdir -p "$OUT_ROOT/logs"
export TPI_HARD_CLUSTER_MAX_HARD_NODES=1024 TPI_LATENT_NORM_CLIP_RATIO=0
export TPI_PLAN_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

variants=(reencode8 reencode16 reencode32 reencode64)
intervals=(8 16 32 64)
circuits=(b22_C b17_C)
benchmarks=(iscas99__b22 iscas99__b17)
gpus=(5 6 1 2)

run_one() {
    local variant="$1" interval="$2" circuit="$3" benchmark="$4" gpu="$5"
    local out="$OUT_ROOT/$variant/$circuit"
    local log="$OUT_ROOT/logs/${variant}__${circuit}.log"
    echo "[round6] start variant=$variant circuit=$circuit gpu=$gpu interval=$interval"
    TPI_LATENT_REENCODE_INTERVAL="$interval" CUDA_VISIBLE_DEVICES="$gpu" \
    python scripts/run_gmean_sweep.py \
        --eval-protocol configs/eval_protocol_coverage_only.json --protocol-keep-cli-benchmarks \
        --benchmarks "$benchmark" \
        --checkpoint runs/planner_aligned_q_rank_v5_context_safe/best_final_horizon.pt \
        --planners greedy --score-fields q_pred_context --beam-objectives cumulative \
        --beam-widths 1 --lookahead-depths 1 --max-candidates 48 --discount-gammas 0.9 \
        --candidate-strategies hard_fault_cluster --candidate-diversity-penalties 0.0 \
        --candidate-diversity-depths 4 \
        --candidate-real-fault-priors autoresearch/eval8-real-priors-budget5-v1/real_fault_priors.csv \
        --candidate-allowlist "autoresearch/original-netlist-recovery-260712/exact_itc99/$circuit/exact_candidate_nodes.txt" \
        --plan-device cuda --time-limit-hours 72 --out-dir "$out" 2>&1 \
        | sed -u "s|^|[round6/$variant/$circuit] |" | tee "$log"
    echo "[round6] done variant=$variant circuit=$circuit gpu=$gpu"
}

pids=()
labels=()
job=0
for variant_index in "${!variants[@]}"; do
    for circuit_index in "${!circuits[@]}"; do
        gpu="${gpus[$((job % ${#gpus[@]}))]}"
        run_one "${variants[$variant_index]}" "${intervals[$variant_index]}" \
            "${circuits[$circuit_index]}" "${benchmarks[$circuit_index]}" "$gpu" &
        pids+=("$!")
        labels+=("${variants[$variant_index]}/${circuits[$circuit_index]}")
        job=$((job + 1))
    done
done

failed=0
for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
        echo "[round6] failed job=${labels[$index]}"
        failed=1
    fi
done
if (( failed )); then exit 1; fi
python scripts/summarize_bottleneck_search.py "$OUT_ROOT"

