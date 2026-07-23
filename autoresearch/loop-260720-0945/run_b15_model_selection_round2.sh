#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
BASE="autoresearch/loop-260720-0945/model_training_round2"
OUT="$BASE/b15_selection"
mkdir -p "$OUT/logs"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export TPI_TORCH_DETERMINISTIC=1
export TPI_HARD_CLUSTER_MAX_HARD_NODES=1024
export TPI_LATENT_NORM_CLIP_RATIO=4
export TPI_Q_CONTEXT_SUPPORT_ALPHA=0.45
export TPI_Q_CONTEXT_DISAGREEMENT_BETA=0.10
export TPI_TYPED_RESIDUAL_CLIP=1.0
export TPI_TYPED_RESIDUAL_DISAGREEMENT_BETA=0.25
export TPI_SCORE_QUANTIZATION=0.001
export TPI_PLAN_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

variants=(typed_balanced typed_long typed_rank)
alphas=(0.00 0.05 0.10 0.20 0.35 0.50)

run_one() {
    local variant="$1" alpha="$2" gpu="$3"
    local checkpoint="$BASE/runs/$variant/best_final_horizon.pt"
    local tag="alpha_${alpha/./p}"
    local out_dir="$OUT/$variant/$tag"
    if [[ ! -f "$checkpoint" ]]; then
        echo "[b15-round2] missing checkpoint: $checkpoint" >&2
        return 1
    fi
    echo "[b15-round2] start variant=$variant alpha=$alpha gpu=$gpu"
    TPI_TYPED_RESIDUAL_ALPHA="$alpha" CUDA_VISIBLE_DEVICES="$gpu" \
        python -u scripts/run_gmean_sweep.py \
        --eval-protocol configs/eval_protocol_coverage_only.json --protocol-keep-cli-benchmarks \
        --benchmarks iscas99__b15_1 --checkpoint "$checkpoint" \
        --planners greedy --score-fields q_typed_residual_context \
        --beam-objectives cumulative --beam-widths 1 --lookahead-depths 1 \
        --max-candidates 48 --discount-gammas 0.9 \
        --candidate-strategies hard_fault_cluster \
        --candidate-diversity-penalties 0.0 --candidate-diversity-depths 4 \
        --candidate-real-fault-priors autoresearch/eval8-real-priors-budget5-v1/real_fault_priors.csv \
        --candidate-allowlist autoresearch/original-netlist-recovery-260712/exact_itc99/b15_C/exact_candidate_nodes.txt \
        --plan-device cuda --time-limit-hours 24 --stream-logs \
        --out-dir "$out_dir" 2>&1 \
        | sed -u "s|^|[b15-round2/$variant/$tag] |" \
        | tee "$OUT/logs/${variant}__${tag}.log"
    echo "[b15-round2] done variant=$variant alpha=$alpha gpu=$gpu"
}

failed=0
batch_pids=()
job=0
wait_batch() {
    local pid
    for pid in "${batch_pids[@]}"; do
        if ! wait "$pid"; then failed=1; fi
    done
    batch_pids=()
}

for variant in "${variants[@]}"; do
    for alpha in "${alphas[@]}"; do
        gpu=$((job % 2 + 1))
        run_one "$variant" "$alpha" "$gpu" &
        batch_pids+=("$!")
        job=$((job + 1))
        if (( ${#batch_pids[@]} == 6 )); then wait_batch; fi
    done
done
if (( ${#batch_pids[@]} > 0 )); then wait_batch; fi
if (( failed )); then exit 1; fi
python scripts/select_conservative_typed_on_b15.py
