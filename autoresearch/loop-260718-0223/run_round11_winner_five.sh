#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
OUT_ROOT="autoresearch/loop-260718-0223/round11_winner_five"
CHECKPOINT="runs/planner_aligned_q_rank_v5_context_safe/best_final_horizon.pt"
mkdir -p "$OUT_ROOT/logs"
export TPI_HARD_CLUSTER_MAX_HARD_NODES=1024 TPI_LATENT_NORM_CLIP_RATIO=4
export TPI_Q_CONTEXT_SUPPORT_ALPHA=0.45 TPI_Q_CONTEXT_DISAGREEMENT_BETA=0.10
export TPI_PLAN_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

circuits=(b15_C b20_C b21_C b22_C b17_C)
benchmarks=(iscas99__b15_1 iscas99__b20 iscas99__b21 iscas99__b22 iscas99__b17)
gpus=(0 1 2 4 5)

run_one() {
    local circuit="$1" benchmark="$2" gpu="$3"
    echo "[round11-five] start circuit=$circuit gpu=$gpu"
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_gmean_sweep.py \
        --eval-protocol configs/eval_protocol_coverage_only.json --protocol-keep-cli-benchmarks \
        --benchmarks "$benchmark" --checkpoint "$CHECKPOINT" \
        --planners greedy --score-fields q_pred_context --beam-objectives cumulative \
        --beam-widths 1 --lookahead-depths 1 --max-candidates 48 --discount-gammas 0.9 \
        --candidate-strategies hard_fault_cluster --candidate-diversity-penalties 0.8 \
        --candidate-diversity-depths 8 \
        --candidate-real-fault-priors autoresearch/eval8-real-priors-budget5-v1/real_fault_priors.csv \
        --candidate-allowlist "autoresearch/original-netlist-recovery-260712/exact_itc99/$circuit/exact_candidate_nodes.txt" \
        --plan-device cuda --time-limit-hours 72 --out-dir "$OUT_ROOT/$circuit" 2>&1 \
        | sed -u "s|^|[round11-five/$circuit] |" | tee "$OUT_ROOT/logs/$circuit.log"
    echo "[round11-five] done circuit=$circuit gpu=$gpu"
}

pids=()
for index in "${!circuits[@]}"; do
    run_one "${circuits[$index]}" "${benchmarks[$index]}" "${gpus[$index]}" &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
if (( failed )); then exit 1; fi

python scripts/summarize_exact_itc99_eval.py --eval-root "$OUT_ROOT"
python scripts/verify_uniform_exact_itc99.py "$OUT_ROOT"
