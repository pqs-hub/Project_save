#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
OUT_ROOT="autoresearch/loop-260718-0223"
export TPI_HARD_CLUSTER_MAX_HARD_NODES=1024 TPI_PLAN_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

run_one() {
    local variant="$1" score="$2" circuit="$3" benchmark="$4" gpu="$5"
    local log="$OUT_ROOT/logs/${variant}__${circuit}.retry.log"
    echo "[retry] start variant=$variant circuit=$circuit gpu=$gpu"
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_gmean_sweep.py \
        --eval-protocol configs/eval_protocol_coverage_only.json --protocol-keep-cli-benchmarks \
        --benchmarks "$benchmark" \
        --checkpoint runs/planner_aligned_q_rank_v5_context_safe/best_final_horizon.pt \
        --planners greedy --score-fields "$score" --beam-objectives cumulative \
        --beam-widths 1 --lookahead-depths 1 --max-candidates 48 --discount-gammas 0.9 \
        --candidate-strategies hard_fault_cluster --candidate-diversity-penalties 0.0 \
        --candidate-diversity-depths 4 \
        --candidate-real-fault-priors autoresearch/eval8-real-priors-budget5-v1/real_fault_priors.csv \
        --candidate-allowlist "autoresearch/original-netlist-recovery-260712/exact_itc99/$circuit/exact_candidate_nodes.txt" \
        --plan-device cuda --time-limit-hours 72 --out-dir "$OUT_ROOT/$variant/$circuit" 2>&1 \
        | sed -u "s|^|[retry/$variant/$circuit] |" | tee "$log"
    echo "[retry] done variant=$variant circuit=$circuit gpu=$gpu"
}

run_one v5_q_type_context q_pred_type_context b20_C iscas99__b20 6 & p1=$!
run_one v5_consensus_type_context consensus_pred_type_context b21_C iscas99__b21 5 & p2=$!
failed=0
for pid in "$p1" "$p2"; do if ! wait "$pid"; then failed=1; fi; done
exit "$failed"

