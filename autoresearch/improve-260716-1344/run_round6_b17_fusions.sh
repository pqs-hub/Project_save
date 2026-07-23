#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
out_root="autoresearch/improve-260716-1344/round6_b17_fusions"
log_root="$out_root/console_logs"
mkdir -p "$log_root"
c48="$(find autoresearch/improve-260716-1344/round3_b17/context_cluster_c48/plans -name '*.csv' -print -quit)"
c96="$(find autoresearch/improve-260716-1344/round4_b17/context_cluster_c96_prior/plans -name '*.csv' -print -quit)"

for spec in equal:1:1 favor_c48:3:1 favor_c96:1:3; do
  IFS=: read -r tag w48 w96 <<<"$spec"
  plan_dir="$out_root/$tag/plans"
  mkdir -p "$plan_dir"
  python scripts/fuse_ranked_plans.py --plans "$c48" "$c96" --weights "$w48" "$w96" \
    --budget 994 --rrf-k 60 --out "$plan_dir/iscas99__b17.csv"
done

pids=()
for tag in equal favor_c48 favor_c96; do
  echo "[round6] start tag=$tag"
  (
    python scripts/evaluate_existing_plans.py --benchmarks iscas99__b17 \
      --plan-dir "$out_root/$tag/plans" --out-dir "$out_root/$tag/evaluation" \
      --backend atalanta-bist \
      --atalanta-bin /data4/pengqingsong/DFT/tool/atalanta_bist_with_ufaults/atalanta \
      --patterns 300000 --seed 2026 --parallel-jobs 1 2>&1 |
      tee "$log_root/$tag.log"
    echo "[round6] done tag=$tag"
  ) &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
