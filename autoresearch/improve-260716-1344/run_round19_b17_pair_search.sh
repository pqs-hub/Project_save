#!/usr/bin/env bash
set -euo pipefail

root="autoresearch/improve-260716-1344/round19_b17_pair_search"
primary="autoresearch/improve-260716-1344/round5_b17/context_cluster_c48_seed1024_prior/plans/iscas99__b17__greedy__cumulative__q_pred_context__bw1__d1__c48__g1p0__candhard_fault_cluster__div0p0__s0.csv"
secondary="autoresearch/improve-260716-1344/round3_b17/context_cluster_c48/plans/iscas99__b17__greedy__cumulative__q_pred_context__bw1__d1__c48__g1p0__candhard_fault_cluster__div0p0__s0.csv"
mkdir -p "$root/console_logs"

run_pair() {
    local secondary_rank="$1"
    local tag="pair_0_${secondary_rank}"
    local run_dir="$root/$tag"
    mkdir -p "$run_dir/plans"
    python scripts/splice_plan_suffix.py \
        --primary "$primary" \
        --secondary "$secondary" \
        --budget 994 \
        --replace 2 \
        --secondary-indices "0,$secondary_rank" \
        --out "$run_dir/plans/iscas99__b17.csv"
    python scripts/evaluate_existing_plans.py \
        --benchmarks iscas99__b17 \
        --plan-dir "$run_dir/plans" \
        --out-dir "$run_dir/evaluation" \
        --backend atalanta-bist \
        --atalanta-bin /data4/pengqingsong/DFT/tool/atalanta_bist_with_ufaults/atalanta \
        --patterns 300000 \
        --seed 2026 \
        --parallel-jobs 1
}

for secondary_rank in 9 10 11 12; do
    run_pair "$secondary_rank" \
        2>&1 | tee "$root/console_logs/pair_0_${secondary_rank}.log" &
done
wait
