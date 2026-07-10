#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data4/pengqingsong/DFT/TPI-my.3}
cd "$ROOT"

CHECKPOINT=${CHECKPOINT:-runs/rollout_loss_A_reward_only/epoch_009.pt}
BENCH_ROOT=${BENCH_ROOT:-$ROOT/autoresearch/deeptpi_table2_restored_bench}
REAL_FAULT_PRIORS=${REAL_FAULT_PRIORS:-autoresearch/eval8-real-priors-budget5-v1/real_fault_priors.csv}
OUT_ROOT=${OUT_ROOT:-autoresearch/exact-rank-table2-hybrid-k96-realfault-300k-$(date +%y%m%d-%H%M%S)}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
CANDIDATE_STRATEGY=${CANDIDATE_STRATEGY:-heuristic_recall_pool}
MAX_CANDIDATES=${MAX_CANDIDATES:-96}
PATTERNS=${PATTERNS:-300000}
SCORE_FIELDS=${SCORE_FIELDS:-reward_pred}
TOP_KS=${TOP_KS:-1,8,16,32,48,96}
ORACLE_TOP_M=${ORACLE_TOP_M:-5}
BENCHMARK_FILTER=${BENCHMARK_FILTER:-}
RESUME_FLAG=${RESUME_FLAG:-}
CLEANUP_WORKDIR=${CLEANUP_WORKDIR:-1}

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

if [[ -n "$BENCHMARK_FILTER" ]]; then
  FILTERED=()
  IFS=, read -r -a FILTER_ITEMS <<< "$BENCHMARK_FILTER"
  for bench in "${BENCHMARKS[@]}"; do
    alias="$bench"
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
      if [[ "$item" == "$bench" || "$item" == "$alias" ]]; then
        FILTERED+=("$bench")
        break
      fi
    done
  done
  BENCHMARKS=("${FILTERED[@]}")
fi

if (( ${#BENCHMARKS[@]} == 0 )); then
  echo "no benchmarks selected" >&2
  exit 1
fi
if (( ${#GPU_LIST[@]} < ${#BENCHMARKS[@]} )); then
  echo "need ${#BENCHMARKS[@]} GPUs, got ${#GPU_LIST[@]} from GPUS=$GPUS" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT/logs"
echo "[launcher] out=$OUT_ROOT"
echo "[launcher] checkpoint=$CHECKPOINT strategy=$CANDIDATE_STRATEGY K=$MAX_CANDIDATES patterns=$PATTERNS"
echo "[launcher] benchmarks=${BENCHMARKS[*]} gpus=${GPU_LIST[*]}"

pids=()
for idx in "${!BENCHMARKS[@]}"; do
  bench=${BENCHMARKS[$idx]}
  gpu=${GPU_LIST[$idx]}
  bench_out="$OUT_ROOT/$bench"
  log="$OUT_ROOT/logs/$bench.log"
  mkdir -p "$bench_out"
  echo "[launcher] start $bench gpu=$gpu log=$log"
  (
    set -euo pipefail
    export TPI_BENCH_ROOT="$BENCH_ROOT"
    export CUDA_VISIBLE_DEVICES="$gpu"
    args=()
    if [[ "$RESUME_FLAG" == "1" || "$RESUME_FLAG" == "true" ]]; then
      args+=(--resume)
    fi
    if [[ "$CLEANUP_WORKDIR" == "1" || "$CLEANUP_WORKDIR" == "true" ]]; then
      args+=(--cleanup-workdir)
    fi
    python scripts/measure_world_model_precision.py \
      --checkpoint "$CHECKPOINT" \
      --benchmarks "$bench" \
      --candidate-strategies "$CANDIDATE_STRATEGY" \
      --max-candidates "$MAX_CANDIDATES" \
      --score-fields "$SCORE_FIELDS" \
      --top-ks "$TOP_KS" \
      --oracle-top-m "$ORACLE_TOP_M" \
      --patterns "$PATTERNS" \
      --real-fault-priors "$REAL_FAULT_PRIORS" \
      --out-dir "$bench_out" \
      --skip-reward-accuracy \
      "${args[@]}"
  ) 2>&1 | tee "$log" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

OUT_ROOT="$OUT_ROOT" EXPECTED_COUNT="${#BENCHMARKS[@]}" python - <<'PY'
import csv
import json
import math
import os
from pathlib import Path

out = Path(os.environ["OUT_ROOT"])
expected = int(os.environ["EXPECTED_COUNT"])

def read_tsv(path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))

def write_tsv(path, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in fields} for r in rows)

def f(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

merged_actions = []
merged_pred = []
merged_rank = []
merged_state = []
merged_groups = []
for child in sorted(p for p in out.iterdir() if p.is_dir() and p.name != "logs"):
    merged_actions.extend(read_tsv(child / "oracle_actions.tsv"))
    merged_pred.extend(read_tsv(child / "prediction_metrics.tsv"))
    merged_rank.extend(read_tsv(child / "rank_metrics.tsv"))
    merged_state.extend(read_tsv(child / "state_summary.tsv"))
    merged_groups.extend(read_tsv(child / "oracle_groups.tsv"))

write_tsv(out / "merged_oracle_actions.tsv", merged_actions)
write_tsv(out / "merged_prediction_metrics.tsv", merged_pred)
write_tsv(out / "merged_rank_metrics.tsv", merged_rank)
write_tsv(out / "merged_state_summary.tsv", merged_state)
write_tsv(out / "merged_oracle_groups.tsv", merged_groups)

reward_rows = [r for r in merged_pred if r.get("score_field") == "reward_pred"]
rank_reward_rows = [r for r in merged_rank if r.get("score_field") == "reward_pred"]

summary = {
    "completed_groups": len(merged_groups),
    "expected_groups": expected,
    "oracle_actions": len(merged_actions),
    "prediction_metric_rows": len(merged_pred),
    "rank_metric_rows": len(merged_rank),
}
if reward_rows:
    for key in ["spearman", "kendall_tau", "pearson", "sign_accuracy", "top1_real_delta_tc", "top1_regret", "negative_top1"]:
        vals = [f(r.get(key)) for r in reward_rows]
        vals = [v for v in vals if math.isfinite(v)]
        if vals:
            summary[f"mean_{key}"] = sum(vals) / len(vals)
    summary["negative_top1_count"] = sum(1 for r in reward_rows if int(float(r.get("negative_top1") or 0)) == 1)

topk = {}
for r in rank_reward_rows:
    k = r.get("k")
    if not k:
        continue
    topk.setdefault(k, []).append(r)
summary["topk"] = {}
for k, rows in sorted(topk.items(), key=lambda kv: int(kv[0])):
    entry = {}
    for key in ["oracle_action_recall", "oracle_node_recall", "regret", "best_in_topk_delta_tc"]:
        vals = [f(r.get(key)) for r in rows]
        vals = [v for v in vals if math.isfinite(v)]
        if vals:
            entry[f"mean_{key}"] = sum(vals) / len(vals)
    summary["topk"][k] = entry

(out / "rank_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

exit "$status"
