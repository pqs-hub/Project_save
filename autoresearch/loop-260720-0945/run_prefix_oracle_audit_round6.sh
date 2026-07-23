#!/usr/bin/env bash
set -euo pipefail

BASE=autoresearch/loop-260720-0945/model_training_round6
ORACLE=autoresearch/loop-260720-0945/model_training_round5/onpolicy_prefix_oracle/oracle_actions.tsv
OUT="$BASE/prefix_oracle_audit"
mkdir -p "$OUT"

specs=()
for variant in cone_rank cone_toplist cone_toplist_sa; do
  for checkpoint_tag in best_final_horizon epoch_012; do
    checkpoint="$BASE/runs/$variant/$checkpoint_tag.pt"
    if [[ ! -f "$checkpoint" ]]; then
      echo "missing checkpoint: $checkpoint" >&2
      exit 1
    fi
    specs+=(--checkpoint "${variant}_${checkpoint_tag}=$checkpoint")
  done
done

CUDA_VISIBLE_DEVICES="${AUDIT_GPU:-1}" python -u scripts/evaluate_prefix_oracle_ranking.py \
  --oracle-actions "$ORACLE" \
  "${specs[@]}" \
  --score-fields typed_marginal_pred,typed_return_pred,typed_sa_reduction_total_pred \
  --plan-device cuda \
  --out-dir "$OUT" \
  2>&1 | tee "$OUT/run.log"
