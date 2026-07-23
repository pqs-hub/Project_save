#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT_ROOT="autoresearch/unified-b15-reward-c48-exact-260718"
mkdir -p "$OUT_ROOT/console_logs"
export TPI_HARD_CLUSTER_MAX_HARD_NODES=512

run_one() {
    local circuit="$1"
    local benchmark="$2"
    local gpu="$3"
    local out_dir="$OUT_ROOT/$circuit"
    local log="$OUT_ROOT/console_logs/$circuit.log"
    echo "[unified-reward] start circuit=$circuit benchmark=$benchmark gpu=$gpu hard_seeds=512"
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_gmean_sweep.py \
        --eval-protocol configs/eval_protocol_coverage_only.json \
        --protocol-keep-cli-benchmarks \
        --benchmarks "$benchmark" \
        --checkpoint runs/rollout_loss_A_reward_only/epoch_009.pt \
        --planners greedy \
        --score-fields reward_pred \
        --beam-objectives cumulative \
        --beam-widths 1 \
        --lookahead-depths 1 \
        --max-candidates 48 \
        --discount-gammas 0.9 \
        --candidate-strategies hard_fault_cluster \
        --candidate-diversity-penalties 0.0 \
        --candidate-diversity-depths 4 \
        --candidate-allowlist "autoresearch/original-netlist-recovery-260712/exact_itc99/$circuit/exact_candidate_nodes.txt" \
        --plan-device cuda \
        --time-limit-hours 72 \
        --out-dir "$out_dir" \
        --stream-logs 2>&1 | tee "$log"
    echo "[unified-reward] done circuit=$circuit gpu=$gpu"
}

run_one b20_C iscas99__b20 0 & p20=$!
run_one b21_C iscas99__b21 1 & p21=$!
run_one b22_C iscas99__b22 2 & p22=$!

failed=0
for pid in "$p20" "$p21" "$p22"; do
    if ! wait "$pid"; then
        failed=1
    fi
done
exit "$failed"
