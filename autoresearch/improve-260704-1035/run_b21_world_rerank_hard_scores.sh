#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/data4/pengqingsong/DFT/TPI-my.3}
cd "$REPO_ROOT"
export DFT_ROOT=${DFT_ROOT:-/data4/pengqingsong/DFT}

OUT_ROOT=${OUT_ROOT:-autoresearch/improve-260704-1035/b21_world_rerank_hard_scores}
ATALANTA_BIN=${ATALANTA_BIN:-/data4/pengqingsong/DFT/tool/atalanta_bist_with_ufaults/atalanta}
mkdir -p "$OUT_ROOT/logs"

run_variant() {
  local name=$1
  local gpu=$2
  local checkpoint=$3
  local score_field=$4
  local out_dir="$OUT_ROOT/$name"
  local top_log="$OUT_ROOT/logs/${name}.log"

  mkdir -p "$out_dir" "$OUT_ROOT/logs"
  {
    echo "[$(date -Is)] start name=${name} gpu=${gpu} score_field=${score_field} checkpoint=${checkpoint}"
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
      --out-dir "$out_dir"
    echo "[$(date -Is)] done name=${name}"
  } 2>&1 | sed -u "s/^/[${name} gpu${gpu}] /" | tee "$top_log"
}

pids=()
run_variant "v4_hard_reduction_total" 4 "runs/mainline_accuracy_improve_v4_reduction_sign/best.pt" "hard_reduction_total_pred" &
pids+=($!)
run_variant "v4_derived_hard_reduction_hybrid" 5 "runs/mainline_accuracy_improve_v4_reduction_sign/best.pt" "derived_hard_reduction_hybrid_pred" &
pids+=($!)
run_variant "v4_hybrid" 6 "runs/mainline_accuracy_improve_v4_reduction_sign/best.pt" "hybrid_pred" &
pids+=($!)
run_variant "v3_hard_reduction_total" 7 "runs/mainline_accuracy_improve_v3_hard_precision/best.pt" "hard_reduction_total_pred" &
pids+=($!)
run_variant "v3_derived_hard_reduction_hybrid" 4 "runs/mainline_accuracy_improve_v3_hard_precision/best.pt" "derived_hard_reduction_hybrid_pred" &
pids+=($!)
run_variant "v3_hybrid" 5 "runs/mainline_accuracy_improve_v3_hard_precision/best.pt" "hybrid_pred" &
pids+=($!)

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done

python - <<'PY'
import csv
from pathlib import Path

root = Path("autoresearch/improve-260704-1035/b21_world_rerank_hard_scores")
rows = []
for name in (
    "v4_hard_reduction_total",
    "v4_derived_hard_reduction_hybrid",
    "v4_hybrid",
    "v3_hard_reduction_total",
    "v3_derived_hard_reduction_hybrid",
    "v3_hybrid",
):
    result_path = root / name / "results.tsv"
    plan_paths = sorted((root / name / "plans").glob("*.csv")) if (root / name / "plans").exists() else []
    nan_rows = ""
    first_nan_step = ""
    if plan_paths:
        with plan_paths[0].open() as f:
            plan_rows = list(csv.DictReader(f))
        bad = []
        for idx, row in enumerate(plan_rows, start=1):
            if any(str(row.get(field, "")).lower() == "nan" for field in ("score_adjusted", "sequence_score", "reward_pred", "hard_reduction_total_pred", "hybrid_pred", "derived_hard_reduction_hybrid_pred")):
                bad.append(idx)
        nan_rows = str(len(bad))
        first_nan_step = str(bad[0]) if bad else ""
    if not result_path.exists():
        rows.append({
            "variant": name,
            "status": "missing_results",
            "results": str(result_path),
            "nan_rows": nan_rows,
            "first_nan_step": first_nan_step,
        })
        continue
    with result_path.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            row = dict(row)
            row["variant"] = name
            row["nan_rows"] = nan_rows
            row["first_nan_step"] = first_nan_step
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
    "nan_rows",
    "first_nan_step",
    "plan_elapsed_sec",
    "eval_elapsed_sec",
    "elapsed_sec",
    "plan_csv",
    "eval_dir",
    "error",
    "results",
]
with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
print(out)
PY

exit "$status"
