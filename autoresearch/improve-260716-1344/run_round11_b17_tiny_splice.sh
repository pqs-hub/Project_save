#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
out_root="autoresearch/improve-260716-1344/round11_b17_tiny_splice"
log_root="$out_root/console_logs"
mkdir -p "$log_root"
seed512="$(find autoresearch/improve-260716-1344/round3_b17/context_cluster_c48/plans -name '*.csv' -print -quit)"
seed1024="$(find autoresearch/improve-260716-1344/round5_b17/context_cluster_c48_seed1024_prior/plans -name '*.csv' -print -quit)"
atalanta="/data4/pengqingsong/DFT/tool/atalanta_bist_with_ufaults/atalanta"

for count in 1 2 4 8 16; do
  tag="replace_${count}"
  python scripts/splice_plan_suffix.py --primary "$seed1024" --secondary "$seed512" \
    --budget 994 --replace "$count" --out "$out_root/$tag/plans/iscas99__b17.csv"
done

pids=()
for tag in replace_1 replace_2 replace_4 replace_8 replace_16; do
  echo "[round11] start tag=$tag"
  (
    python scripts/evaluate_existing_plans.py --benchmarks iscas99__b17 \
      --plan-dir "$out_root/$tag/plans" --out-dir "$out_root/$tag/evaluation" \
      --backend atalanta-bist --atalanta-bin "$atalanta" \
      --patterns 300000 --seed 2026 --parallel-jobs 1 2>&1 | tee "$log_root/$tag.log"
    echo "[round11] done tag=$tag"
  ) &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
exit "$status"
