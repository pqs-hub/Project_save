#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/data4/pengqingsong/DFT/TPI-my.3}
cd "$REPO_ROOT"

if [[ "${CONFIRM_EVAL8_HELDOUT:-0}" != "1" ]]; then
  echo "Refusing to touch the 8 held-out target circuits unless CONFIRM_EVAL8_HELDOUT=1 is set." >&2
  echo "Training and dev scripts do not read eval8 target circuits." >&2
  exit 2
fi

OUT_ROOT=${OUT_ROOT:-autoresearch/improve-260706-0959/eval8_final_heldout_oldbudget}
ATALANTA_BIN=${ATALANTA_BIN:-/data4/pengqingsong/DFT/tool/atalanta_bist_with_ufaults/atalanta}
EVAL_PROTOCOL=${EVAL_PROTOCOL:-configs/eval_protocol_coverage_only.json}
GPUS_CSV=${GPUS_CSV:-0,1,2,3,4,5,6,7}
PRIOR_MODE=${PRIOR_MODE:-none}
CANDIDATE_REAL_FAULT_PRIORS=${CANDIDATE_REAL_FAULT_PRIORS:-autoresearch/eval8-real-priors-budget5-v1/real_fault_priors.csv}
FORCE=${FORCE:-0}
MODEL_FILTER=${MODEL_FILTER:-}
BENCH_FILTER=${BENCH_FILTER:-}
BENCHMARKS_CSV=${BENCHMARKS_CSV:-}

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "GPUS_CSV must contain at least one GPU id" >&2
  exit 2
fi
MAX_PARALLEL=${MAX_PARALLEL:-${#GPUS[@]}}

prior_args=()
case "$PRIOR_MODE" in
  old)
    prior_args+=(--candidate-real-fault-priors "$CANDIDATE_REAL_FAULT_PRIORS")
    echo "Using old candidate real-fault priors for apples-to-apples comparison: $CANDIDATE_REAL_FAULT_PRIORS"
    ;;
  none)
    echo "Running without candidate real-fault priors; this is not apples-to-apples with old C_depth2."
    ;;
  *)
    echo "PRIOR_MODE must be old or none, got: $PRIOR_MODE" >&2
    exit 2
    ;;
esac

mkdir -p "$OUT_ROOT/logs"
python scripts/validate_eval_protocol.py --protocol "$EVAL_PROTOCOL"

if [[ -n "$BENCHMARKS_CSV" ]]; then
  IFS=',' read -r -a BENCHMARKS <<< "$BENCHMARKS_CSV"
else
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
fi

bench_selected() {
  local bench=$1
  if [[ -z "$BENCH_FILTER" ]]; then
    return 0
  fi
  local item
  IFS=',' read -r -a filter_items <<< "$BENCH_FILTER"
  for item in "${filter_items[@]}"; do
    item=${item//[[:space:]]/}
    if [[ "$bench" == "$item" ]]; then
      return 0
    fi
  done
  return 1
}

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
  local bench=$4
  local gpu=$5
  local ensemble=${6:-}
  local ensemble_alpha=${7:-1.0}
  local out_dir="$OUT_ROOT/${name}/${bench}"
  local result_file="$out_dir/results.tsv"
  local log_file="$OUT_ROOT/logs/${name}__${bench}.log"

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
    echo "[$(date -Is)] skip existing name=${name} bench=${bench} result=${result_file}" | tee -a "$log_file"
    return 0
  fi

  echo "[$(date -Is)] start name=${name} bench=${bench} gpu=${gpu} score=${score}" | tee "$log_file"
  extra_args=()
  if [[ -n "$ensemble" && "$ensemble" != "-" ]]; then
    extra_args+=(--ensemble-checkpoints "$ensemble" --ensemble-lcb-alpha "$ensemble_alpha")
  fi

  CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_gmean_sweep.py \
    --eval-protocol "$EVAL_PROTOCOL" \
    --protocol-keep-cli-benchmarks \
    --checkpoint "$ckpt" \
    "${extra_args[@]}" \
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
    "${prior_args[@]}" \
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

active=0
status=0
i=0
for bench in "${BENCHMARKS[@]}"; do
  if ! bench_selected "$bench"; then
    continue
  fi
  for entry in "${MODELS[@]}"; do
    read -r name ckpt score ensemble alpha <<< "$entry"
    if [[ -n "$MODEL_FILTER" && "$name" != *"$MODEL_FILTER"* ]]; then
      continue
    fi
    gpu=${GPUS[$((i % ${#GPUS[@]}))]}
    run_one "$name" "$ckpt" "$score" "$bench" "$gpu" "${ensemble:-}" "${alpha:-1.0}" &
    active=$((active + 1))
    i=$((i + 1))
    if (( active >= MAX_PARALLEL )); then
      wait -n || status=$?
      active=$((active - 1))
    fi
  done
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
        print(f"{model}\tn={len(vals)}\tmacro_delta_tc={sum(vals) / len(vals):.6f}\tmin={min(vals):.6f}\tmax={max(vals):.6f}")
print(f"summary: {out}")
PY

python autoresearch/improve-260706-0959/summarize_vs_old.py "$OUT_ROOT/summary.tsv" || status=1
python scripts/validate_eval_protocol.py --protocol "$EVAL_PROTOCOL" --results "$OUT_ROOT" || status=1

exit "$status"
