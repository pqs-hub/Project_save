#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
out_root="autoresearch/improve-260716-1344/round8_b17_type_ablation"
log_root="$out_root/console_logs"
mkdir -p "$log_root"
source_plan="$(find autoresearch/improve-260716-1344/round3_b17/context_cluster_c48/plans -name '*.csv' -print -quit)"
atalanta="/data4/pengqingsong/DFT/tool/atalanta_bist_with_ufaults/atalanta"

make_plan() {
  local tag="$1" from_types="$2" min_step="$3"
  python scripts/rewrite_plan_action_types.py --plan "$source_plan" \
    --from-types "$from_types" --to-type observe --min-step "$min_step" \
    --out "$out_root/$tag/plans/iscas99__b17.csv"
}
make_plan late_controls_to_observe control0,control1 500
make_plan late_control0_to_observe control0 500
make_plan late_control1_to_observe control1 500
make_plan all_control1_to_observe control1 1

pids=()
for tag in late_controls_to_observe late_control0_to_observe late_control1_to_observe all_control1_to_observe; do
  echo "[round8] start tag=$tag"
  (
    python scripts/evaluate_existing_plans.py --benchmarks iscas99__b17 \
      --plan-dir "$out_root/$tag/plans" --out-dir "$out_root/$tag/evaluation" \
      --backend atalanta-bist --atalanta-bin "$atalanta" \
      --patterns 300000 --seed 2026 --parallel-jobs 1 2>&1 | tee "$log_root/$tag.log"
    echo "[round8] done tag=$tag"
  ) &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
exit "$status"
