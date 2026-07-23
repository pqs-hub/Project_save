#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
OUT_ROOT="autoresearch/loop-260718-0223/real_label_calibration_round14"
INIT="runs/planner_aligned_q_rank_v5_context_safe/best_final_horizon.pt"
BASE_CONFIG="configs/planner_aligned_q_rank_v5_context_safe.json"
mkdir -p "$OUT_ROOT/configs" "$OUT_ROOT/logs" "$OUT_ROOT/b15"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TPI_PLAN_THREADS=1

variants=(balanced reward rank gentle)
lrs=(0.00002 0.00002 0.00002 0.000005)
jepas=(0.25 0.25 0.25 0.25)
fcs=(0.22 0.80 0.10 0.22)
q_ranks=(0.38 0.15 0.80 0.38)
q_values=(0.08 0.10 0.10 0.08)
candidates=(0.18 0.10 0.25 0.18)
contexts=(0.25 0.10 0.35 0.25)
gpus=(0 1 2 4)
scores=(reward_pred q_pred_context guarded_reward_context)

build_config() {
    local index="$1" variant="${variants[$1]}"
    jq \
        --arg run_dir "$OUT_ROOT/runs/$variant" --arg init "$INIT" \
        --argjson lr "${lrs[$index]}" --argjson jepa "${jepas[$index]}" \
        --argjson fc "${fcs[$index]}" --argjson q_rank "${q_ranks[$index]}" \
        --argjson q_value "${q_values[$index]}" \
        --argjson candidate "${candidates[$index]}" --argjson context "${contexts[$index]}" \
        '.run_dir=$run_dir | .init_checkpoint=$init | .init_checkpoint_strict=true
         | .trainable_modules="rollout_dynamics" | .epochs=3 | .lr=$lr
         | .max_train_samples=12000 | .max_train_steps_per_epoch=300
         | .max_val_samples=3072 | .max_val_steps=64
         | .lambda_jepa=$jepa | .lambda_fc=$fc
         | .lambda_q_rank=$q_rank | .lambda_q_value=$q_value
         | .lambda_candidate=$candidate | .lambda_context_rank=$context
         | .rollout_training=true | .rollout_max_horizon=5
         | .rollout_start_epoch=1 | .rollout_increase_every=1
         | .rollout_start_horizon=2 | .rollout_horizon_increment=2
         | .require_full_horizon=true | .repeat_train_samples=false | .device="cuda"' \
        "$BASE_CONFIG" > "$OUT_ROOT/configs/$variant.json"
}

train_one() {
    local index="$1" variant="${variants[$1]}" gpu="${gpus[$1]}"
    echo "[round14] train_start variant=$variant gpu=$gpu"
    CUDA_VISIBLE_DEVICES="$gpu" python -m tpi_jepa.train --config "$OUT_ROOT/configs/$variant.json" 2>&1 \
        | sed -u "s|^|[round14/train/$variant] |" | tee "$OUT_ROOT/logs/$variant.train.log"
    echo "[round14] train_done variant=$variant gpu=$gpu"
}

eval_one() {
    local variant="$1" score="$2" gpu="$3" tag="${variant}__${score}"
    echo "[round14] b15_start tag=$tag gpu=$gpu"
    TPI_HARD_CLUSTER_MAX_HARD_NODES=1024 TPI_LATENT_NORM_CLIP_RATIO=4 \
    TPI_Q_CONTEXT_SUPPORT_ALPHA=0.45 TPI_Q_CONTEXT_DISAGREEMENT_BETA=0.10 \
    TPI_SCORE_QUANTIZATION=0.001 CUDA_VISIBLE_DEVICES="$gpu" \
    python scripts/run_gmean_sweep.py \
        --eval-protocol configs/eval_protocol_coverage_only.json --protocol-keep-cli-benchmarks \
        --benchmarks iscas99__b15_1 --checkpoint "$OUT_ROOT/runs/$variant/best_final_horizon.pt" \
        --planners greedy --score-fields "$score" --beam-objectives cumulative \
        --beam-widths 1 --lookahead-depths 1 --max-candidates 48 --discount-gammas 0.9 \
        --candidate-strategies hard_fault_cluster --candidate-diversity-penalties 0.0 \
        --candidate-diversity-depths 4 \
        --candidate-real-fault-priors autoresearch/eval8-real-priors-budget5-v1/real_fault_priors.csv \
        --candidate-allowlist autoresearch/original-netlist-recovery-260712/exact_itc99/b15_C/exact_candidate_nodes.txt \
        --plan-device cuda --time-limit-hours 72 --out-dir "$OUT_ROOT/b15/$tag/b15_C" 2>&1 \
        | sed -u "s|^|[round14/b15/$tag] |" | tee "$OUT_ROOT/logs/$tag.b15.log"
    echo "[round14] b15_done tag=$tag gpu=$gpu"
}

for index in "${!variants[@]}"; do build_config "$index"; done
pids=()
for index in "${!variants[@]}"; do train_one "$index" & pids+=("$!"); done
failed=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
if (( failed )); then exit 1; fi

pids=()
tags=()
job=0
for variant in "${variants[@]}"; do
    for score in "${scores[@]}"; do
        tags+=("${variant}__${score}")
        eval_one "$variant" "$score" "${gpus[$((job % ${#gpus[@]}))]}" &
        pids+=("$!")
        job=$((job + 1))
    done
done
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
if (( failed )); then exit 1; fi

for tag in "${tags[@]}"; do
    jq -r --arg tag "$tag" '[ $tag, .macro_mean_delta_tc ] | @tsv' \
        "$OUT_ROOT/b15/$tag/b15_C/best.json"
done | sort -t $'\t' -k2,2nr | tee "$OUT_ROOT/b15_selection.tsv"
