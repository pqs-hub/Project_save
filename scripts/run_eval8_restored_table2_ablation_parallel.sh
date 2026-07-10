#!/usr/bin/env bash
set -u -o pipefail

ROOT=${ROOT:-/data4/pengqingsong/DFT/TPI-my.3}
cd "$ROOT"

CHECKPOINT=${CHECKPOINT:-runs/rollout_loss_A_reward_only/epoch_009.pt}
BENCH_ROOT=${BENCH_ROOT:-$ROOT/autoresearch/deeptpi_table2_restored_bench}
ATALANTA_BIN=${ATALANTA_BIN:-/data4/pengqingsong/DFT/tool/atalanta_bist_with_ufaults/atalanta}
OUT_ROOT=${OUT_ROOT:-autoresearch/eval8-restored-table2-ablation-hardcone96-300k-parallel-$(date +%y%m%d-%H%M%S)}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
PATTERNS=${PATTERNS:-300000}
SEED=${SEED:-2026}
TIMEOUT_SEC=${TIMEOUT_SEC:-14400}
MAX_CANDIDATES=${MAX_CANDIDATES:-96}
BEAM_WIDTH=${BEAM_WIDTH:-2}
LOOKAHEAD_DEPTH=${LOOKAHEAD_DEPTH:-2}
PLAN_DEVICE=${PLAN_DEVICE:-cuda}
EVAL_BACKEND=${EVAL_BACKEND:-atalanta-bist}
CANDIDATE_STRATEGY=${CANDIDATE_STRATEGY:-hard_fault_cone}
CANDIDATE_REAL_FAULT_PRIORS=${CANDIDATE_REAL_FAULT_PRIORS:-}
EVAL_STEP_MODE=${EVAL_STEP_MODE:-final}
BENCHMARK_FILTER=${BENCHMARK_FILTER:-}
METHOD_FILTER=${METHOD_FILTER:-A_heuristic_only,B_world_rerank,C_depth2_rollout}

IFS=, read -r -a GPU_LIST <<< "$GPUS"
IFS=, read -r -a METHOD_LIST <<< "$METHOD_FILTER"

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

# Table IV / restored Table II TP budgets.
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

if [[ -n "$BENCHMARK_FILTER" ]]; then
  FILTERED_BENCHMARKS=()
  FILTERED_BUDGETS=()
  IFS=, read -r -a FILTER_ITEMS <<< "$BENCHMARK_FILTER"
  for idx in "${!BENCHMARKS[@]}"; do
    bench=${BENCHMARKS[$idx]}
    alias=$bench
    case "$bench" in
      iscas99__b15_1) alias=b15_C ;;
      iscas99__b20) alias=b20_C ;;
      iscas99__b21) alias=b21_C ;;
      iscas99__b22) alias=b22_C ;;
      epfl__random_control__i2c__i2c) alias=i2c_aig ;;
      epfl__arithmetic__max__max) alias=max_aig ;;
      iscas99__b17) alias=b17_C ;;
      openabcd__mem_ctrl_orig) alias=mem_ctrl_aig ;;
    esac
    for item in "${FILTER_ITEMS[@]}"; do
      if [[ "$bench" == "$item" || "$alias" == "$item" ]]; then
        FILTERED_BENCHMARKS+=("$bench")
        FILTERED_BUDGETS+=("${BUDGETS[$idx]}")
        break
      fi
    done
  done
  BENCHMARKS=("${FILTERED_BENCHMARKS[@]}")
  BUDGETS=("${FILTERED_BUDGETS[@]}")
  if (( ${#BENCHMARKS[@]} == 0 )); then
    echo "BENCHMARK_FILTER=$BENCHMARK_FILTER matched no benchmarks" >&2
    exit 2
  fi
fi

if (( ${#GPU_LIST[@]} < ${#BENCHMARKS[@]} )); then
  echo "need ${#BENCHMARKS[@]} GPUs, got ${#GPU_LIST[@]} from GPUS=$GPUS" >&2
  exit 2
fi

echo "[launcher] benchmarks=${BENCHMARKS[*]} gpus=${GPU_LIST[*]} out=$OUT_ROOT"
echo "[launcher] methods=${METHOD_LIST[*]}"
mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/status"

method_enabled() {
  local want=$1
  local method
  for method in "${METHOD_LIST[@]}"; do
    if [[ "$method" == "$want" ]]; then
      return 0
    fi
  done
  return 1
}

run_world_method() {
  local method=$1
  local bench=$2
  local budget=$3
  local bench_out=$4
  local planner=$5
  local beam_width=$6
  local depth=$7
  local prior_args=()
  if [[ -n "$CANDIDATE_REAL_FAULT_PRIORS" ]]; then
    prior_args+=(--candidate-real-fault-priors "$CANDIDATE_REAL_FAULT_PRIORS")
  fi

  stdbuf -oL -eL python -u scripts/run_gmean_sweep.py \
    --checkpoint "$CHECKPOINT" \
    --benchmarks "$bench" \
    --benchmark-budgets "{\"$bench\": $budget}" \
    --planners "$planner" \
    --score-fields reward_pred \
    --beam-objectives cumulative \
    --beam-widths "$beam_width" \
    --lookahead-depths "$depth" \
    --max-candidates "$MAX_CANDIDATES" \
    --candidate-strategies "$CANDIDATE_STRATEGY" \
    --candidate-diversity-penalties 0.0 \
    --candidate-diversity-depths 4 \
    --candidate-sample-seeds 0 \
    --plan-device "$PLAN_DEVICE" \
    --eval-backend "$EVAL_BACKEND" \
    --atalanta-bin "$ATALANTA_BIN" \
    --patterns "$PATTERNS" \
    --seed "$SEED" \
    --timeout-sec "$TIMEOUT_SEC" \
    --time-limit-hours 72 \
    --prior-setup-elapsed-sec 0 \
    --eval-step-mode "$EVAL_STEP_MODE" \
    "${prior_args[@]}" \
    --stream-logs \
    --out-dir "$bench_out/$method"
}

write_heuristic_result() {
  local bench=$1
  local budget=$2
  local method_out=$3
  local plan_csv=$4
  local eval_dir=$5
  local plan_sec=$6
  local eval_sec=$7
  local status=$8
  local error=${9:-}

  BENCH="$bench" \
  BUDGET="$budget" \
  METHOD_OUT="$method_out" \
  PLAN_CSV="$plan_csv" \
  EVAL_DIR="$eval_dir" \
  PLAN_SEC="$plan_sec" \
  EVAL_SEC="$eval_sec" \
  STATUS_VALUE="$status" \
  ERROR_VALUE="$error" \
  PATTERNS_VALUE="$PATTERNS" \
  SEED_VALUE="$SEED" \
  MAX_CANDIDATES_VALUE="$MAX_CANDIDATES" \
  CANDIDATE_STRATEGY_VALUE="$CANDIDATE_STRATEGY" \
  python - <<'PY'
import csv
import os
from pathlib import Path

from scripts.run_gmean_sweep import RESULT_FIELDS, final_eval_metrics, logic_gate_count

bench = os.environ["BENCH"]
budget = int(os.environ["BUDGET"])
method_out = Path(os.environ["METHOD_OUT"])
eval_dir = Path(os.environ["EVAL_DIR"])
status = os.environ["STATUS_VALUE"]
error = os.environ["ERROR_VALUE"]
metrics = {}
labels_csv = eval_dir / "labels.csv"
if status == "ok" and labels_csv.exists():
    metrics = final_eval_metrics(labels_csv)

def metric(name: str) -> str:
    return str(metrics.get(name, ""))

row = {
    "timestamp": "",
    "variant_id": (
        "heuristic_iterative__cumulative__none__bw0__d0__"
        f"c{os.environ['MAX_CANDIDATES_VALUE']}__g1p0__"
        f"cand{os.environ['CANDIDATE_STRATEGY_VALUE']}__div0p0__s0"
    ),
    "status": status,
    "benchmark_id": bench,
    "logic_gates": logic_gate_count(bench),
    "budget_mode": "table4",
    "budget": budget,
    "planner": "heuristic_iterative_first",
    "beam_objective": "cumulative",
    "score_field": "heuristic",
    "beam_width": 0,
    "lookahead_depth": 0,
    "max_candidates": os.environ["MAX_CANDIDATES_VALUE"],
    "k_recall": "",
    "k_model": "",
    "k_plan": "",
    "discount_gamma": 1.0,
    "candidate_strategy": os.environ["CANDIDATE_STRATEGY_VALUE"],
    "candidate_diversity_penalty": 0.0,
    "candidate_diversity_depth": 4,
    "candidate_sample_seed": 0,
    "patterns": os.environ["PATTERNS_VALUE"],
    "seed": os.environ["SEED_VALUE"],
    "plan_score_sum": "",
    "plan_reward_sum": "",
    "plan_fc_sum": "",
    "plan_return_sum": "",
    "plan_sequence_sum": "",
    "plan_objective_sum": "",
    "delta_test_coverage": metric("delta_test_coverage"),
    "delta_fault_coverage": metric("delta_fault_coverage"),
    "delta_pattern_count": metric("delta_pattern_count"),
    "plan_csv": os.environ["PLAN_CSV"],
    "eval_dir": os.environ["EVAL_DIR"],
    "prior_setup_elapsed_sec": 0.0,
    "plan_elapsed_sec": os.environ["PLAN_SEC"],
    "eval_elapsed_sec": os.environ["EVAL_SEC"],
    "elapsed_sec": float(os.environ["PLAN_SEC"]) + float(os.environ["EVAL_SEC"]),
    "error": error,
}

method_out.mkdir(parents=True, exist_ok=True)
with (method_out / "results.tsv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, delimiter="\t")
    writer.writeheader()
    writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})
PY
}

run_heuristic_method() {
  local bench=$1
  local budget=$2
  local bench_out=$3
  local method=A_heuristic_only
  local method_out="$bench_out/$method"
  local plan_csv="$method_out/plans/${bench}.csv"
  local eval_dir="$method_out/evals/${bench}"
  local start end plan_sec eval_sec status error
  local prior_args=()

  mkdir -p "$method_out/plans" "$method_out/evals" "$method_out/logs"
  if [[ -n "$CANDIDATE_REAL_FAULT_PRIORS" ]]; then
    prior_args+=(--real-fault-priors "$CANDIDATE_REAL_FAULT_PRIORS")
  fi

  start=$(date +%s)
  if python scripts/plan_candidate_baseline.py \
    --benchmark-id "$bench" \
    --budget "$budget" \
    --max-candidates "$MAX_CANDIDATES" \
    --iterative-first \
    --candidate-strategy "$CANDIDATE_STRATEGY" \
    "${prior_args[@]}" \
    --out "$plan_csv"; then
    status=ok
    error=""
  else
    status=error
    error="heuristic plan failed"
  fi
  end=$(date +%s)
  plan_sec=$((end - start))

  eval_sec=0
  if [[ "$status" == "ok" ]]; then
    start=$(date +%s)
    if python -m tpi_jepa.evaluate_plan_tmax \
      --benchmark-id "$bench" \
      --plan-csv "$plan_csv" \
      --out-dir "$eval_dir" \
      --patterns "$PATTERNS" \
      --seed "$SEED" \
      --backend "$EVAL_BACKEND" \
      --atalanta-bin "$ATALANTA_BIN" \
      --timeout-sec "$TIMEOUT_SEC" \
      --eval-step-mode "$EVAL_STEP_MODE" \
      --force \
      --cleanup-workdir; then
      status=ok
      error=""
    else
      status=error
      error="heuristic eval failed"
    fi
    end=$(date +%s)
    eval_sec=$((end - start))
  fi

  write_heuristic_result "$bench" "$budget" "$method_out" "$plan_csv" "$eval_dir" "$plan_sec" "$eval_sec" "$status" "$error"
  [[ "$status" == "ok" ]]
}

run_one_benchmark() {
  local idx=$1
  local bench=${BENCHMARKS[$idx]}
  local budget=${BUDGETS[$idx]}
  local gpu=${GPU_LIST[$idx]}
  local bench_out="$OUT_ROOT/$bench"
  local log="$OUT_ROOT/logs/$bench.log"
  local status_file="$OUT_ROOT/status/$bench.status"
  local overall=0

  mkdir -p "$bench_out"
  echo "[launcher] start benchmark=$bench budget=$budget gpu=$gpu out=$bench_out"

  (
    export TPI_BENCH_ROOT="$BENCH_ROOT"
    export CUDA_VISIBLE_DEVICES="$gpu"

    if method_enabled A_heuristic_only; then
      echo "[method] A_heuristic_only benchmark=$bench budget=$budget"
      run_heuristic_method "$bench" "$budget" "$bench_out" || overall=1
    fi

    if method_enabled B_world_rerank; then
      echo "[method] B_world_rerank benchmark=$bench budget=$budget"
      run_world_method "B_world_rerank" "$bench" "$budget" "$bench_out" greedy 1 1 || overall=1
    fi

    if method_enabled C_depth2_rollout; then
      echo "[method] C_depth2_rollout benchmark=$bench budget=$budget"
      run_world_method "C_depth2_rollout" "$bench" "$budget" "$bench_out" beam "$BEAM_WIDTH" "$LOOKAHEAD_DEPTH" || overall=1
    fi

    exit "$overall"
  ) 2>&1 | sed -u "s/^/[$bench gpu$gpu] /" | tee "$log"

  local status=${PIPESTATUS[0]}
  echo "$status" > "$status_file"
  echo "[launcher] done benchmark=$bench status=$status log=$log"
  return "$status"
}

pids=()
for idx in "${!BENCHMARKS[@]}"; do
  run_one_benchmark "$idx" &
  pids+=("$!")
done

overall=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    overall=1
  fi
done

OUT_ROOT="$OUT_ROOT" EXPECTED_COUNT="${#BENCHMARKS[@]}" METHOD_FILTER="$METHOD_FILTER" python - <<'PY'
import csv
import json
import os
from pathlib import Path
from statistics import mean

out = Path(os.environ["OUT_ROOT"])
expected_count = int(os.environ.get("EXPECTED_COUNT", "8"))
methods = [item for item in os.environ.get("METHOD_FILTER", "").split(",") if item]
bench_names = {
    "iscas99__b15_1": "b15_C",
    "iscas99__b20": "b20_C",
    "iscas99__b21": "b21_C",
    "iscas99__b22": "b22_C",
    "epfl__random_control__i2c__i2c": "i2c_aig",
    "epfl__arithmetic__max__max": "max_aig",
    "iscas99__b17": "b17_C",
    "openabcd__mem_ctrl_orig": "mem_ctrl_aig",
}

rows = []
fields = None
for method in methods:
    method_rows = []
    for path in sorted(out.glob(f"*/{method}/results.tsv")):
        with path.open(newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            fields = fields or reader.fieldnames
            for row in reader:
                row = dict(row)
                row["method"] = method
                method_rows.append(row)
                rows.append(row)
    if fields:
        method_dir = out / method
        method_dir.mkdir(exist_ok=True)
        with (method_dir / "results.tsv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerows([{field: row.get(field, "") for field in fields} for row in method_rows])

if fields:
    merged_fields = ["method", *fields]
    with (out / "merged_results.tsv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=merged_fields, delimiter="\t")
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in merged_fields} for row in rows])

def num(row, key, default=0.0):
    try:
        value = row.get(key, "")
        return default if value == "" else float(value)
    except (TypeError, ValueError):
        return default

summary = {}
for method in methods:
    ok = [row for row in rows if row["method"] == method and row.get("status") == "ok"]
    deltas = [num(row, "delta_test_coverage") for row in ok]
    summary[method] = {
        "completed": len(ok),
        "macro_mean_delta_tc": mean(deltas) if deltas else None,
        "min_delta_tc": min(deltas) if deltas else None,
        "positive_count": sum(1 for value in deltas if value > 0),
        "negative_count": sum(1 for value in deltas if value < 0),
        "plan_elapsed_sec": sum(num(row, "plan_elapsed_sec") for row in ok),
        "eval_elapsed_sec": sum(num(row, "eval_elapsed_sec") for row in ok),
        "elapsed_sec_sum": sum(num(row, "elapsed_sec") for row in ok),
    }

(out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

wide = {}
for row in rows:
    bench = row.get("benchmark_id", "")
    method = row.get("method", "")
    wide.setdefault(bench, {"benchmark_id": bench, "circuit": bench_names.get(bench, bench), "budget": row.get("budget", "")})
    wide[bench][f"{method}_delta_pct"] = f"{100.0 * num(row, 'delta_test_coverage'):.3f}" if row.get("status") == "ok" else "ERR"
    wide[bench][f"{method}_status"] = row.get("status", "")

wide_fields = ["circuit", "benchmark_id", "budget"]
for method in methods:
    wide_fields.extend([f"{method}_delta_pct", f"{method}_status"])
with (out / "ablation_delta_wide.tsv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=wide_fields, delimiter="\t")
    writer.writeheader()
    for bench in bench_names:
        if bench in wide:
            writer.writerow({field: wide[bench].get(field, "") for field in wide_fields})

lines = ["# Restored Table II Table-IV-Budget Ablation", ""]
lines.append("| Method | Done | Macro ΔTC | Min ΔTC | Pos | Neg | Plan sec | Eval sec |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
for method in methods:
    item = summary.get(method, {})
    macro = item.get("macro_mean_delta_tc")
    min_delta = item.get("min_delta_tc")
    lines.append(
        f"| {method} | {item.get('completed', 0)}/{expected_count} | "
        f"{100.0 * macro:.3f}% | {100.0 * min_delta:.3f}% | "
        f"{item.get('positive_count', 0)} | {item.get('negative_count', 0)} | "
        f"{item.get('plan_elapsed_sec', 0.0):.1f} | {item.get('eval_elapsed_sec', 0.0):.1f} |"
        if macro is not None and min_delta is not None
        else f"| {method} | 0/{expected_count} |  |  |  |  |  |  |"
    )
lines.append("")
lines.append("Per-circuit deltas are in `ablation_delta_wide.tsv`.")
(out / "analysis_table.md").write_text("\n".join(lines) + "\n")

print(json.dumps(summary, indent=2, sort_keys=True))
print(f"[merge] wrote {out / 'merged_results.tsv'}")
print(f"[merge] wrote {out / 'ablation_delta_wide.tsv'}")
print(f"[merge] wrote {out / 'analysis_table.md'}")
PY

exit "$overall"
