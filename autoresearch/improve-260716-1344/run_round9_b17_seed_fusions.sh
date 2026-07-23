#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
out_root="autoresearch/improve-260716-1344/round9_b17_seed_fusions"
log_root="$out_root/console_logs"
mkdir -p "$log_root"
seed512="$(find autoresearch/improve-260716-1344/round3_b17/context_cluster_c48/plans -name '*.csv' -print -quit)"
seed1024="$(find autoresearch/improve-260716-1344/round5_b17/context_cluster_c48_seed1024_prior/plans -name '*.csv' -print -quit)"
atalanta="/data4/pengqingsong/DFT/tool/atalanta_bist_with_ufaults/atalanta"

for spec in equal:1:1 favor_seed1024:1:3 favor_seed512:3:1; do
  IFS=: read -r tag w512 w1024 <<<"$spec"
  python scripts/fuse_ranked_plans.py --plans "$seed512" "$seed1024" \
    --weights "$w512" "$w1024" --budget 994 --rrf-k 60 \
    --out "$out_root/$tag/plans/iscas99__b17.csv"
done

pids=()
for tag in equal favor_seed1024 favor_seed512; do
  echo "[round9] start tag=$tag"
  (
    python scripts/evaluate_existing_plans.py --benchmarks iscas99__b17 \
      --plan-dir "$out_root/$tag/plans" --out-dir "$out_root/$tag/evaluation" \
      --backend atalanta-bist --atalanta-bin "$atalanta" \
      --patterns 300000 --seed 2026 --parallel-jobs 1 2>&1 | tee "$log_root/$tag.log"
    echo "[round9] done tag=$tag"
  ) &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
exit "$status"
