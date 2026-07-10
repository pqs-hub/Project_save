#!/usr/bin/env bash
set -u -o pipefail

ROOT=${ROOT:-/data4/pengqingsong/DFT/TPI-my.3}
cd "$ROOT"

CHECKPOINT=${CHECKPOINT:-runs/rollout_loss_A_reward_only/epoch_009.pt}
BENCH_ROOT=${BENCH_ROOT:-$ROOT/autoresearch/deeptpi_table2_restored_bench}
ATALANTA_BIN=${ATALANTA_BIN:-/data4/pengqingsong/DFT/tool/atalanta_bist_with_ufaults/atalanta}
OUT_ROOT=${OUT_ROOT:-autoresearch/eval8-restored-table2-rollout-loss-A-epoch009-final-300k-parallel-$(date +%y%m%d-%H%M%S)}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}

IFS=, read -r -a GPU_LIST <<< "$GPUS"

BENCHMARKS=(
  iscas99__b15_1
  iscas99__b20
  iscas99__b21
  iscas99__b22
  epfl__random_control__i2c__i2c
  epfl__arithmetic__max__max
  iscas99__b17
  openabcd__mem_ctrl_orig
)

BUDGETS=(
  278
  616
  628
  915
  34
  94
  994
  1273
)

if (( ${#GPU_LIST[@]} < ${#BENCHMARKS[@]} )); then
  echo "need ${#BENCHMARKS[@]} GPUs, got ${#GPU_LIST[@]} from GPUS=$GPUS" >&2
  exit 2
fi

mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/status"

run_one() {
  local idx=$1
  local bench=${BENCHMARKS[$idx]}
  local budget=${BUDGETS[$idx]}
  local gpu=${GPU_LIST[$idx]}
  local bench_out="$OUT_ROOT/$bench"
  local log="$OUT_ROOT/logs/$bench.log"
  local status_file="$OUT_ROOT/status/$bench.status"

  mkdir -p "$bench_out"
  echo "[launcher] start benchmark=$bench budget=$budget gpu=$gpu out=$bench_out"

  (
    export TPI_BENCH_ROOT="$BENCH_ROOT"
    export CUDA_VISIBLE_DEVICES="$gpu"
    stdbuf -oL -eL python -u scripts/run_gmean_sweep.py \
      --checkpoint "$CHECKPOINT" \
      --benchmarks "$bench" \
      --benchmark-budgets "{\"$bench\": $budget}" \
      --planners beam \
      --score-fields reward_pred \
      --beam-objectives cumulative \
      --beam-widths 2 \
      --lookahead-depths 2 \
      --max-candidates 96 \
      --candidate-strategies hard_fault_cone \
      --candidate-diversity-penalties 0.0 \
      --candidate-diversity-depths 4 \
      --candidate-sample-seeds 0 \
      --plan-device cuda \
      --eval-backend atalanta-bist \
      --atalanta-bin "$ATALANTA_BIN" \
      --patterns 300000 \
      --seed 2026 \
      --timeout-sec 14400 \
      --time-limit-hours 72 \
      --prior-setup-elapsed-sec 0 \
      --eval-step-mode final \
      --stream-logs \
      --out-dir "$bench_out" \
      2>&1
  ) | sed -u "s/^/[$bench gpu$gpu] /" | tee "$log"

  local status=${PIPESTATUS[0]}
  echo "$status" > "$status_file"
  echo "[launcher] done benchmark=$bench status=$status log=$log"
  return "$status"
}

pids=()
for idx in "${!BENCHMARKS[@]}"; do
  run_one "$idx" &
  pids+=("$!")
done

overall=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    overall=1
  fi
done

OUT_ROOT="$OUT_ROOT" python - <<'PY'
import csv
import json
import os
from pathlib import Path
from statistics import mean

out = Path(os.environ["OUT_ROOT"])
rows = []
fields = None
for path in sorted(out.glob("*/results.tsv")):
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fields = fields or reader.fieldnames
        rows.extend(reader)

if fields:
    merged = out / "merged_results.tsv"
    with merged.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

ok_rows = [row for row in rows if row.get("status") == "ok"]
def number(row, key):
    try:
        return float(row.get(key, "nan"))
    except ValueError:
        return float("nan")

deltas = [number(row, "delta_test_coverage") for row in ok_rows]
summary = {
    "out_root": str(out),
    "completed": len(ok_rows),
    "total": len(rows),
    "macro_mean_delta_tc": mean(deltas) if deltas else None,
    "min_delta_tc": min(deltas) if deltas else None,
    "positive_count": sum(1 for value in deltas if value > 0),
    "negative_count": sum(1 for value in deltas if value < 0),
    "plan_elapsed_sec": sum(number(row, "plan_elapsed_sec") for row in ok_rows),
    "eval_elapsed_sec": sum(number(row, "eval_elapsed_sec") for row in ok_rows),
    "elapsed_sec_sum": sum(number(row, "elapsed_sec") for row in ok_rows),
}
(out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

exit "$overall"
