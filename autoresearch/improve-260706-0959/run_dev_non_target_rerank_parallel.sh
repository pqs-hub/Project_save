#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/data4/pengqingsong/DFT/TPI-my.3}
cd "$REPO_ROOT"

OUT_ROOT=${OUT_ROOT:-autoresearch/improve-260706-0959/dev_non_target_rerank}
ATALANTA_BIN=${ATALANTA_BIN:-/data4/pengqingsong/DFT/tool/atalanta_bist_with_ufaults/atalanta}
GPUS_CSV=${GPUS_CSV:-0,1,2,3,4,5,6,7}
DEFAULT_DEV_BENCHMARK=${DEFAULT_DEV_BENCHMARK:-subckt_0001}
if [[ "${DEV_BENCHMARK+x}" == "x" && -n "${DEV_BENCHMARK}" ]]; then
  DEV_BENCHMARK_SOURCE=env
else
  DEV_BENCHMARK=$DEFAULT_DEV_BENCHMARK
  DEV_BENCHMARK_SOURCE=default
fi
PATTERNS=${PATTERNS:-300000}
FORCE=${FORCE:-0}
MODEL_FILTER=${MODEL_FILTER:-}

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "GPUS_CSV must contain at least one GPU id" >&2
  exit 2
fi
MAX_PARALLEL=${MAX_PARALLEL:-${#GPUS[@]}}

case "$DEV_BENCHMARK" in
  epfl__arithmetic__max__max|epfl__random_control__i2c__i2c|iscas99__b15_1|iscas99__b17|iscas99__b20|iscas99__b21|iscas99__b22|openabcd__mem_ctrl_orig|max|max_aig|i2c|i2c_aig|b15_C|b17_C|b20_C|b21_C|b22_C|mem_ctrl|mem_ctrl_aig)
    echo "Refusing non-target dev rerank on held-out target benchmark: $DEV_BENCHMARK" >&2
    exit 2
    ;;
esac

python - "$DEV_BENCHMARK" "$DEV_BENCHMARK_SOURCE" "$DEFAULT_DEV_BENCHMARK" <<'PY'
import sys
from tpi_jepa.labels import find_bench_path

benchmark = sys.argv[1]
source = sys.argv[2]
default = sys.argv[3]
try:
    path = find_bench_path(benchmark)
except FileNotFoundError as exc:
    print(f"Cannot find BENCH for DEV_BENCHMARK={benchmark!r} (source={source}): {exc}", file=sys.stderr)
    print(f"Hint: rerun with DEV_BENCHMARK={default} or unset DEV_BENCHMARK.", file=sys.stderr)
    raise SystemExit(2) from exc
print(f"dev_benchmark={benchmark} source={source} bench={path}", flush=True)
PY

mkdir -p "$OUT_ROOT/logs"

MODELS=(
  "q_rank_v1 runs/planner_aligned_q_rank_v1/best_final_horizon.pt q_pred"
  "reward_rank_v1 runs/planner_aligned_reward_rank_v1/best_final_horizon.pt reward_pred"
  "q_rank_v2_safe runs/planner_aligned_q_rank_v2_safe/best_final_horizon.pt q_pred"
  "q_rank_v2_seed2_safe runs/planner_aligned_q_rank_v2_seed2_safe/best_final_horizon.pt q_pred"
  "q_rank_v3_ndcg_safe runs/planner_aligned_q_rank_v3_ndcg_safe/best_final_horizon.pt q_pred"
  "q_rank_v4_conservative_safe runs/planner_aligned_q_rank_v4_conservative_safe/best_final_horizon.pt q_pred"
  "q_context_v1 runs/planner_aligned_q_rank_v1/best_final_horizon.pt q_pred_context"
  "q_rank_v5_context_safe runs/planner_aligned_q_rank_v5_context_safe/best_final_horizon.pt q_pred_context"
  "q_lcb_ensemble_safe runs/planner_aligned_q_rank_v1/best_final_horizon.pt q_pred_lcb runs/planner_aligned_q_rank_v1/best_final_horizon.pt,runs/planner_aligned_q_rank_v2_safe/best_final_horizon.pt,runs/planner_aligned_q_rank_v2_seed2_safe/best_final_horizon.pt 0.75"
  "reward_rank_v3_safe runs/planner_aligned_reward_rank_v3_safe/best_final_horizon.pt reward_pred"
  "reward_rank_v3_seed2_safe runs/planner_aligned_reward_rank_v3_seed2_safe/best_final_horizon.pt reward_pred"
  "reward_rank_v4_ndcg_safe runs/planner_aligned_reward_rank_v4_ndcg_safe/best_final_horizon.pt reward_pred"
  "reward_context_v1 runs/planner_aligned_reward_rank_v1/best_final_horizon.pt reward_pred_context"
  "reward_rank_v5_context_safe runs/planner_aligned_reward_rank_v5_context_safe/best_final_horizon.pt reward_pred_context"
  "reward_lcb_ensemble_safe runs/planner_aligned_reward_rank_v1/best_final_horizon.pt reward_pred_lcb runs/planner_aligned_reward_rank_v1/best_final_horizon.pt,runs/planner_aligned_reward_rank_v3_safe/best_final_horizon.pt,runs/planner_aligned_reward_rank_v3_seed2_safe/best_final_horizon.pt 0.75"
  "guarded_reward_rank_v1_safe runs/planner_aligned_guarded_reward_rank_v1_safe/best_final_horizon.pt guarded_reward"
  "guarded_reward_rank_v1_seed2_safe runs/planner_aligned_guarded_reward_rank_v1_seed2_safe/best_final_horizon.pt guarded_reward"
  "hybrid_rank_v1_safe runs/planner_aligned_hybrid_rank_v1_safe/best_final_horizon.pt hybrid_pred"
  "hybrid_rank_v1_seed2_safe runs/planner_aligned_hybrid_rank_v1_seed2_safe/best_final_horizon.pt hybrid_pred"
)

run_one() {
  local name=$1
  local ckpt=$2
  local score=$3
  local gpu=$4
  local ensemble=${5:-}
  local ensemble_alpha=${6:-1.0}
  local out_dir="$OUT_ROOT/$name"
  local result_file="$out_dir/results.tsv"
  local log_file="$OUT_ROOT/logs/${name}.log"

  if [[ ! -s "$ckpt" ]]; then
    echo "[$(date -Is)] skip missing checkpoint name=${name} ckpt=${ckpt}" | tee "$log_file"
    return 0
  fi
  if [[ -n "$ensemble" && "$ensemble" != "-" ]]; then
    IFS=',' read -r -a ensemble_paths <<< "$ensemble"
    for ensemble_ckpt in "${ensemble_paths[@]}"; do
      if [[ ! -s "$ensemble_ckpt" ]]; then
        echo "[$(date -Is)] skip missing ensemble checkpoint name=${name} ckpt=${ensemble_ckpt}" | tee "$log_file"
        return 0
      fi
    done
  fi
  if [[ "$FORCE" != "1" && -s "$result_file" ]]; then
    echo "[$(date -Is)] skip existing name=${name} result=${result_file}" | tee -a "$log_file"
    return 0
  fi

  echo "[$(date -Is)] start name=${name} dev=${DEV_BENCHMARK} gpu=${gpu} score=${score}" | tee "$log_file"
  extra_args=()
  if [[ -n "$ensemble" && "$ensemble" != "-" ]]; then
    extra_args+=(--ensemble-checkpoints "$ensemble" --ensemble-lcb-alpha "$ensemble_alpha")
  fi

  CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_gmean_sweep.py \
    --checkpoint "$ckpt" \
    "${extra_args[@]}" \
    --benchmarks "$DEV_BENCHMARK" \
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
    --patterns "$PATTERNS" \
    --seed 2026 \
    --timeout-sec 14400 \
    --eval-backend atalanta-bist \
    --atalanta-bin "$ATALANTA_BIN" \
    --plan-device cuda \
    --eval-step-mode final \
    --time-limit-hours 24 \
    --stream-logs \
    --out-dir "$out_dir" \
    2>&1 | tee -a "$log_file"
  echo "[$(date -Is)] done name=${name}" | tee -a "$log_file"
}

active=0
status=0
i=0
for entry in "${MODELS[@]}"; do
  read -r name ckpt score ensemble alpha <<< "$entry"
  if [[ -n "$MODEL_FILTER" && "$name" != *"$MODEL_FILTER"* ]]; then
    continue
  fi
  gpu=${GPUS[$((i % ${#GPUS[@]}))]}
  run_one "$name" "$ckpt" "$score" "$gpu" "${ensemble:-}" "${alpha:-1.0}" &
  active=$((active + 1))
  i=$((i + 1))
  if (( active >= MAX_PARALLEL )); then
    wait -n || status=$?
    active=$((active - 1))
  fi
done

while (( active > 0 )); do
  wait -n || status=$?
  active=$((active - 1))
done

OUT_ROOT="$OUT_ROOT" python - <<'PY'
import csv
import math
import os
from pathlib import Path

root = Path(os.environ["OUT_ROOT"])
rows = []
for path in sorted(root.glob("*/results.tsv")):
    model = path.parts[-2]
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
for row in rows:
    if row.get("status") == "ok" and row.get("delta_test_coverage") not in ("", "NA"):
        value = float(row["delta_test_coverage"])
        if math.isfinite(value):
            print(f"{row['model']}\t{row['benchmark_id']}\t{row['score_field']}\tdelta_tc={value * 100.0:.3f}%")
print(f"summary: {out}")
PY

exit "$status"
