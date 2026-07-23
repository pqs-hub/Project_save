#!/usr/bin/env bash
set -euo pipefail

LOOP=autoresearch/loop-260720-0945
BASE=$LOOP/model_training_round8
SHORT_ORACLE=$LOOP/model_training_round5/onpolicy_prefix_oracle/oracle_actions.tsv
LONG_ORACLE=$LOOP/model_training_round7/long_prefix_oracle/oracle_actions.tsv
GPU=${AUDIT_GPU:-0}

specs=(
  --checkpoint "round7_incumbent=$LOOP/model_training_round7/runs/cone_long_rank/best_final_horizon.pt"
)
for variant in moe_experts_within moe_joint_within; do
  for checkpoint_tag in best_final_horizon epoch_010; do
    checkpoint="$BASE/runs/$variant/$checkpoint_tag.pt"
    if [[ ! -f "$checkpoint" ]]; then
      echo "missing checkpoint: $checkpoint" >&2
      exit 1
    fi
    specs+=(--checkpoint "${variant}_${checkpoint_tag}=$checkpoint")
  done
done

for split in short long; do
  if [[ "$split" == short ]]; then oracle=$SHORT_ORACLE; else oracle=$LONG_ORACLE; fi
  out="$BASE/prefix_oracle_audit_$split"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$GPU" python -u scripts/evaluate_prefix_oracle_ranking.py \
    --oracle-actions "$oracle" \
    "${specs[@]}" \
    --score-fields typed_marginal_pred,typed_return_pred,typed_sa_reduction_total_pred \
    --plan-device cuda \
    --out-dir "$out" \
    2>&1 | tee "$out/run.log"
done
