#!/usr/bin/env bash
set -euo pipefail

LOOP=autoresearch/loop-260720-0945
BASE=$LOOP/model_training_round10
LONG_ORACLE=$LOOP/model_training_round7/long_prefix_oracle/oracle_actions.tsv
LATE_ORACLE=$LOOP/model_training_round9/late_prefix_oracle/oracle_actions.tsv
GPU=${AUDIT_GPU:-7}

specs=(
  --checkpoint "round8_incumbent=$LOOP/model_training_round8/runs/moe_joint_within/best_final_horizon.pt"
)
for variant in return_within_lr5e5 return_within_lr1e4 return_dual_lr5e5; do
  checkpoint="$BASE/runs/$variant/epoch_008.pt"
  if [[ ! -f "$checkpoint" ]]; then
    echo "missing checkpoint: $checkpoint" >&2
    exit 1
  fi
  specs+=(--checkpoint "$variant=$checkpoint")
done

for split in long late; do
  if [[ "$split" == long ]]; then oracle=$LONG_ORACLE; else oracle=$LATE_ORACLE; fi
  out="$BASE/prefix_oracle_audit_$split"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$GPU" python -u scripts/evaluate_prefix_oracle_ranking.py \
    --oracle-actions "$oracle" \
    "${specs[@]}" \
    --score-fields typed_marginal_pred,typed_return_pred \
    --plan-device cuda \
    --out-dir "$out" \
    2>&1 | tee "$out/run.log"
done
