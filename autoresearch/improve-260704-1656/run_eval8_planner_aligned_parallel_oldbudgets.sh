#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/data4/pengqingsong/DFT/TPI-my.3}
cd "$REPO_ROOT"

OUT_ROOT=${OUT_ROOT:-autoresearch/improve-260704-1656/eval8_planner_aligned_parallel_oldbudgets}
ATALANTA_BIN=${ATALANTA_BIN:-/data4/pengqingsong/DFT/tool/atalanta_bist_with_ufaults/atalanta}
EVAL_PROTOCOL=${EVAL_PROTOCOL:-configs/eval_protocol_coverage_only.json}
GPUS_CSV=${GPUS_CSV:-0,1,2,3,4,5,6,7}
FORCE=${FORCE:-0}

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "GPUS_CSV must contain at least one GPU id" >&2
  exit 2
fi

mkdir -p "$OUT_ROOT/logs"
python scripts/validate_eval_protocol.py --protocol "$EVAL_PROTOCOL"

BENCHMARKS=(
  epfl__arithmetic__max__max
  epfl__random_control__i2c__i2c
  iscas99__b15_1
  iscas99__b17
  iscas99__b20
  iscas99__b21
  iscas99__b22
  openabcd__mem_ctrl_orig
)

run_one() {
  local name=$1
  local ckpt=$2
  local score=$3
  local bench=$4
  local gpu=$5
  local out_dir="$OUT_ROOT/${name}/${bench}"
  local result_file="$out_dir/results.tsv"
  local log_file="$OUT_ROOT/logs/${name}__${bench}.log"

  if [[ "$FORCE" != "1" && -s "$result_file" ]]; then
    echo "[$(date -Is)] skip existing name=${name} bench=${bench} result=${result_file}" | tee -a "$log_file"
    return 0
  fi

  echo "[$(date -Is)] start name=${name} bench=${bench} gpu=${gpu} score=${score}" | tee "$log_file"
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_gmean_sweep.py \
    --eval-protocol "$EVAL_PROTOCOL" \
    --protocol-keep-cli-benchmarks \
    --checkpoint "$ckpt" \
    --benchmarks "$bench" \
    --planners greedy \
    --score-fields "$score" \
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
    --out-dir "$out_dir" \
    2>&1 | tee -a "$log_file"
  echo "[$(date -Is)] done name=${name} bench=${bench}" | tee -a "$log_file"
}

pids=()
i=0
for bench in "${BENCHMARKS[@]}"; do
  gpu=${GPUS[$((i % ${#GPUS[@]}))]}
  run_one q_rank_v1 runs/planner_aligned_q_rank_v1/best_final_horizon.pt q_pred "$bench" "$gpu" &
  pids+=($!)
  i=$((i + 1))

  gpu=${GPUS[$((i % ${#GPUS[@]}))]}
  run_one reward_rank_v1 runs/planner_aligned_reward_rank_v1/best_final_horizon.pt reward_pred "$bench" "$gpu" &
  pids+=($!)
  i=$((i + 1))
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done

OUT_ROOT="$OUT_ROOT" python - <<'PY'
import csv
import math
import os
from pathlib import Path

root = Path(os.environ["OUT_ROOT"])
rows = []

for path in sorted(root.glob("*/*/results.tsv")):
    model = path.parts[-3]
    with path.open(newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            row = dict(row)
            row["model"] = model
            rows.append(row)

out = root / "summary.tsv"
fields = [
    "model",
    "benchmark_id",
    "status",
    "logic_gates",
    "budget",
    "score_field",
    "delta_test_coverage",
    "delta_fault_coverage",
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

for model in sorted({r["model"] for r in rows}):
    vals = [
        float(r["delta_test_coverage"])
        for r in rows
        if r.get("model") == model
        and r.get("status") == "ok"
        and r.get("delta_test_coverage") not in ("", "NA")
        and math.isfinite(float(r["delta_test_coverage"]))
    ]
    if vals:
        print(
            f"{model}\tn={len(vals)}\tmacro_delta_tc={sum(vals) / len(vals):.6f}"
            f"\tmin={min(vals):.6f}\tmax={max(vals):.6f}"
        )

print(f"summary: {out}")
PY

python scripts/validate_eval_protocol.py --protocol "$EVAL_PROTOCOL" --results "$OUT_ROOT" || status=1

exit "$status"
