#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/data4/pengqingsong/DFT/TPI-my.3}
cd "$REPO_ROOT"
export DFT_ROOT=${DFT_ROOT:-/data4/pengqingsong/DFT}

OUT_ROOT=${OUT_ROOT:-autoresearch/improve-260704-1656/planner_aligned_variants}
ATALANTA_BIN=${ATALANTA_BIN:-/data4/pengqingsong/DFT/tool/atalanta_bist_with_ufaults/atalanta}
mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/gates"

run_one() {
  local name=$1
  local gpu=$2
  local config=$3
  local checkpoint=$4
  local score_field=$5
  local log="$OUT_ROOT/logs/${name}.log"

  {
    echo "[$(date -Is)] train_start name=${name} gpu=${gpu} config=${config}"
    CUDA_VISIBLE_DEVICES="$gpu" python -m tpi_jepa.train --config "$config"
    echo "[$(date -Is)] train_done name=${name} checkpoint=${checkpoint}"
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_gmean_sweep.py \
      --checkpoint "$checkpoint" \
      --benchmarks iscas99__b21 \
      --budget-mode floor1pct \
      --benchmark-budgets '{"iscas99__b21": 628}' \
      --planners greedy \
      --score-fields "$score_field" \
      --beam-objectives cumulative \
      --beam-widths 1 \
      --lookahead-depths 1 \
      --max-candidates 96 \
      --candidate-strategies heuristic_recall_pool \
      --candidate-diversity-penalties 0.0 \
      --candidate-diversity-depths 4 \
      --candidate-sample-seeds 0 \
      --candidate-real-fault-priors autoresearch/eval8-real-priors-budget5-v1/real_fault_priors.csv \
      --patterns 300000 \
      --seed 2026 \
      --timeout-sec 14400 \
      --eval-backend atalanta-bist \
      --atalanta-bin "$ATALANTA_BIN" \
      --plan-device cuda \
      --eval-step-mode final \
      --time-limit-hours 72 \
      --stream-logs \
      --out-dir "$OUT_ROOT/gates/$name"
    echo "[$(date -Is)] gate_done name=${name}"
  } 2>&1 | sed -u "s/^/[${name} gpu${gpu}] /" | tee "$log"
}

pids=()
run_one "reward_rank_v1" 4 "configs/planner_aligned_reward_rank_v1.json" "runs/planner_aligned_reward_rank_v1/best_final_horizon.pt" "reward_pred" &
pids+=($!)
run_one "reward_rank_v2_strong" 5 "configs/planner_aligned_reward_rank_v2_strong.json" "runs/planner_aligned_reward_rank_v2_strong/best_final_horizon.pt" "reward_pred" &
pids+=($!)
run_one "q_rank_v1" 6 "configs/planner_aligned_q_rank_v1.json" "runs/planner_aligned_q_rank_v1/best_final_horizon.pt" "q_pred" &
pids+=($!)

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done

python - <<'PY'
import csv
import math
from pathlib import Path

root = Path("autoresearch/improve-260704-1656/planner_aligned_variants")
rows = []

old_path = Path("autoresearch/eval8-b21-hybrid-recall-k96-realfault-300k-260702-203127/iscas99__b21/B_world_rerank/results.tsv")
if old_path.exists():
    with old_path.open() as f:
        row = next(csv.DictReader(f, delimiter="\t"))
    row["variant"] = "old_rollout_loss_A_epoch009"
    row["score_nan_rows"] = "0"
    row["first_score_nan_step"] = ""
    rows.append(row)

for name in ("reward_rank_v1", "reward_rank_v2_strong", "q_rank_v1"):
    result_path = root / "gates" / name / "results.tsv"
    if not result_path.exists():
        rows.append({"variant": name, "status": "missing_results", "error": str(result_path)})
        continue
    with result_path.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            plan = Path(row["plan_csv"])
            score_field = row["score_field"]
            bad = []
            if plan.exists():
                with plan.open() as pf:
                    for idx, plan_row in enumerate(csv.DictReader(pf), start=1):
                        try:
                            value = float(plan_row.get(score_field, "nan"))
                            adjusted = float(plan_row.get("score_adjusted", value))
                            sequence = float(plan_row.get("sequence_score", adjusted))
                            if not (math.isfinite(value) and math.isfinite(adjusted) and math.isfinite(sequence)):
                                bad.append(idx)
                        except Exception:
                            bad.append(idx)
            row["variant"] = name
            row["score_nan_rows"] = str(len(bad))
            row["first_score_nan_step"] = str(bad[0]) if bad else ""
            rows.append(row)

out = root / "summary.tsv"
fields = [
    "variant",
    "status",
    "benchmark_id",
    "budget",
    "planner",
    "score_field",
    "candidate_strategy",
    "delta_test_coverage",
    "delta_fault_coverage",
    "score_nan_rows",
    "first_score_nan_step",
    "plan_elapsed_sec",
    "eval_elapsed_sec",
    "elapsed_sec",
    "plan_csv",
    "eval_dir",
    "error",
]
with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
print(out)
PY

exit "$status"
