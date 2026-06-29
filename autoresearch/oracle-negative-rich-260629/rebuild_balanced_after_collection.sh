#!/usr/bin/env bash
set -euo pipefail

inputs=(
  --input autoresearch/oracle-action-probe-260629-expanded-subckt-train/oracle_actions.tsv
)

if [[ -s autoresearch/oracle-negative-rich-260629/pilot/oracle_actions.tsv ]]; then
  inputs+=(--input autoresearch/oracle-negative-rich-260629/pilot/oracle_actions.tsv)
fi
if [[ -s autoresearch/oracle-negative-rich-260629/topup/oracle_actions.tsv ]]; then
  inputs+=(--input autoresearch/oracle-negative-rich-260629/topup/oracle_actions.tsv)
fi

python scripts/merge_oracle_action_tsv.py \
  "${inputs[@]}" \
  --out-tsv autoresearch/oracle-negative-rich-260629/merged_train_oracle_actions.tsv \
  --out-report autoresearch/oracle-negative-rich-260629/merge_report.md

python scripts/build_balanced_oracle_action_subset.py \
  --train-oracle autoresearch/oracle-negative-rich-260629/merged_train_oracle_actions.tsv \
  --val-oracle autoresearch/oracle-action-probe-260629-expanded-subckt-val/oracle_actions.tsv \
  --transfer-oracle autoresearch/oracle-action-probe-260629-smoke/oracle_actions.tsv \
  --min-negatives-per-group 3 \
  --min-positives-per-group 3 \
  --prefer-negative-types control0,control1 \
  --max-actions-per-group 0 \
  --min-train-groups 120 \
  --min-val-groups 24 \
  --out-dir autoresearch/oracle-balanced-negative-rich-260629

