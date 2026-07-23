#!/usr/bin/env bash
set -euo pipefail

LOOP=autoresearch/loop-260720-0945
BASE=$LOOP/model_training_round20
ORACLE=$BASE/late_prefix_oracle/oracle_actions.tsv
OUT=$BASE/prefix_oracle_audit_expanded
GPU=${AUDIT_GPU:-0}

specs=(
  --checkpoint "round8_incumbent=$LOOP/model_training_round8/runs/moe_joint_within/best_final_horizon.pt"
  --checkpoint "round10_b15_selected=$LOOP/model_training_round10/runs/return_within_lr5e4/epoch_008.pt"
)
for variant in return_pairwise_expanded return_hybrid_listwise return_top_listwise; do
  for epoch in 006 008 010; do
    checkpoint="$BASE/runs/$variant/epoch_${epoch}.pt"
    if [[ ! -f "$checkpoint" ]]; then
      echo "missing checkpoint: $checkpoint" >&2
      exit 1
    fi
    specs+=(--checkpoint "${variant}_e${epoch}=$checkpoint")
  done
done

mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES="$GPU" python -u scripts/evaluate_prefix_oracle_ranking.py \
  --oracle-actions "$ORACLE" \
  "${specs[@]}" \
  --score-fields typed_return_pred \
  --plan-device cuda \
  --out-dir "$OUT" \
  2>&1 | tee "$OUT/run.log"
