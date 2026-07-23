#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
BASE="autoresearch/loop-260720-0945/model_training_round1"
OUT="$BASE/b15_selection"
mkdir -p "$OUT/logs"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export TPI_TORCH_DETERMINISTIC=1
export TPI_HARD_CLUSTER_MAX_HARD_NODES=1024
export TPI_LATENT_NORM_CLIP_RATIO=4
export TPI_Q_CONTEXT_SUPPORT_ALPHA=0.45
export TPI_Q_CONTEXT_DISAGREEMENT_BETA=0.10
export TPI_SCORE_QUANTIZATION=0.001
export TPI_PLAN_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

variants=(frozen_balanced frozen_long joint_low_lr)
gpus=(1 2 1)

run_one() {
    local variant="$1" gpu="$2"
    local checkpoint="$BASE/runs/$variant/best_final_horizon.pt"
    if [[ ! -f "$checkpoint" ]]; then
        echo "[b15-select/$variant] missing checkpoint: $checkpoint" >&2
        return 1
    fi
    echo "[b15-select] start variant=$variant gpu=$gpu"
    CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/run_gmean_sweep.py \
        --eval-protocol configs/eval_protocol_coverage_only.json --protocol-keep-cli-benchmarks \
        --benchmarks iscas99__b15_1 --checkpoint "$checkpoint" \
        --planners greedy \
        --score-fields typed_marginal_pred,typed_return_pred,q_pred_context \
        --beam-objectives cumulative --beam-widths 1 --lookahead-depths 1 \
        --max-candidates 48 --discount-gammas 0.9 \
        --candidate-strategies hard_fault_cluster \
        --candidate-diversity-penalties 0.0 --candidate-diversity-depths 4 \
        --candidate-real-fault-priors autoresearch/eval8-real-priors-budget5-v1/real_fault_priors.csv \
        --candidate-allowlist autoresearch/original-netlist-recovery-260712/exact_itc99/b15_C/exact_candidate_nodes.txt \
        --plan-device cuda --time-limit-hours 24 --stream-logs \
        --out-dir "$OUT/$variant" 2>&1 \
        | sed -u "s|^|[b15-select/$variant] |" \
        | tee "$OUT/logs/$variant.log"
    echo "[b15-select] done variant=$variant gpu=$gpu"
}

pids=()
for index in "${!variants[@]}"; do
    run_one "${variants[$index]}" "${gpus[$index]}" &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
done
if (( failed )); then exit 1; fi
python scripts/select_typed_world_model_on_b15.py
