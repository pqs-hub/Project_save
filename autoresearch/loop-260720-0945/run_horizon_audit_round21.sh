#!/usr/bin/env bash
set -euo pipefail

LOOP=autoresearch/loop-260720-0945
TRAIN=$LOOP/model_training_round21
ORACLE=$TRAIN/ultralong_prefix_oracle/oracle_actions.tsv
OUT=$TRAIN/horizon_prefix_audit_heldout
VARIANTS=(horizon_r8_pairwise horizon_r8_hybrid horizon_r10_pairwise horizon_r10_hybrid)
EPOCHS=(006 008 010)
GPU=${GPU:-0}

mkdir -p "$OUT"
args=(
  --oracle-actions "$ORACLE"
  --checkpoint "round8_incumbent=$LOOP/model_training_round8/runs/moe_joint_within/best_final_horizon.pt"
  --checkpoint "round10_b15_selected=$LOOP/model_training_round10/runs/return_within_lr5e4/epoch_008.pt"
)
for variant in "${VARIANTS[@]}"; do
  for epoch in "${EPOCHS[@]}"; do
    checkpoint=$TRAIN/runs/$variant/epoch_${epoch}.pt
    test -s "$checkpoint"
    args+=(--checkpoint "${variant}_e${epoch}=$checkpoint")
  done
done

CUDA_VISIBLE_DEVICES="$GPU" python -u scripts/evaluate_prefix_oracle_ranking.py \
  "${args[@]}" \
  --score-fields typed_return_pred --include-benchmarks subckt_0360,subckt_0230 \
  --latent-norm-clip-ratio 4 --plan-device cuda --out-dir "$OUT" \
  2>&1 | tee "$OUT/run.log"
