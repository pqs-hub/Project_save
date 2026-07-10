#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/data4/pengqingsong/DFT/TPI-my.3}
cd "$REPO_ROOT"

OUT_ROOT=${OUT_ROOT:-autoresearch/improve-260706-0959/train_safe_parallel}
GPUS_CSV=${GPUS_CSV:-0,1,2,3,4,5,6,7}
FORCE=${FORCE:-0}
MODEL_FILTER=${MODEL_FILTER:-}

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "GPUS_CSV must contain at least one GPU id" >&2
  exit 2
fi
MAX_PARALLEL=${MAX_PARALLEL:-${#GPUS[@]}}

mkdir -p "$OUT_ROOT/logs"

CONFIGS=(
  "q_rank_v2_safe configs/planner_aligned_q_rank_v2_safe.json"
  "q_rank_v2_seed2_safe configs/planner_aligned_q_rank_v2_seed2_safe.json"
  "q_rank_v3_ndcg_safe configs/planner_aligned_q_rank_v3_ndcg_safe.json"
  "q_rank_v4_conservative_safe configs/planner_aligned_q_rank_v4_conservative_safe.json"
  "q_rank_v5_context_safe configs/planner_aligned_q_rank_v5_context_safe.json"
  "reward_rank_v3_safe configs/planner_aligned_reward_rank_v3_safe.json"
  "reward_rank_v3_seed2_safe configs/planner_aligned_reward_rank_v3_seed2_safe.json"
  "reward_rank_v4_ndcg_safe configs/planner_aligned_reward_rank_v4_ndcg_safe.json"
  "reward_rank_v5_context_safe configs/planner_aligned_reward_rank_v5_context_safe.json"
  "guarded_reward_rank_v1_safe configs/planner_aligned_guarded_reward_rank_v1_safe.json"
  "guarded_reward_rank_v1_seed2_safe configs/planner_aligned_guarded_reward_rank_v1_seed2_safe.json"
  "hybrid_rank_v1_safe configs/planner_aligned_hybrid_rank_v1_safe.json"
  "hybrid_rank_v1_seed2_safe configs/planner_aligned_hybrid_rank_v1_seed2_safe.json"
)

python - <<'PY'
import csv
import json
from pathlib import Path

configs = [
    "configs/planner_aligned_q_rank_v2_safe.json",
    "configs/planner_aligned_q_rank_v2_seed2_safe.json",
    "configs/planner_aligned_q_rank_v3_ndcg_safe.json",
    "configs/planner_aligned_q_rank_v4_conservative_safe.json",
    "configs/planner_aligned_q_rank_v5_context_safe.json",
    "configs/planner_aligned_reward_rank_v3_safe.json",
    "configs/planner_aligned_reward_rank_v3_seed2_safe.json",
    "configs/planner_aligned_reward_rank_v4_ndcg_safe.json",
    "configs/planner_aligned_reward_rank_v5_context_safe.json",
    "configs/planner_aligned_guarded_reward_rank_v1_safe.json",
    "configs/planner_aligned_guarded_reward_rank_v1_seed2_safe.json",
    "configs/planner_aligned_hybrid_rank_v1_safe.json",
    "configs/planner_aligned_hybrid_rank_v1_seed2_safe.json",
]
target_ids = {
    "epfl__arithmetic__max__max",
    "epfl__random_control__i2c__i2c",
    "iscas99__b15_1",
    "iscas99__b17",
    "iscas99__b20",
    "iscas99__b21",
    "iscas99__b22",
    "openabcd__mem_ctrl_orig",
    "max",
    "max_aig",
    "i2c",
    "i2c_aig",
    "b15_C",
    "b17_C",
    "b20_C",
    "b21_C",
    "b22_C",
    "mem_ctrl",
    "mem_ctrl_aig",
}
blocked_path_parts = {
    "exact-rank-table2",
    "oracle-action-probe",
    "deeptpi_table2",
}

def action_paths(value):
    if isinstance(value, str):
        return [value]
    paths = []
    for item in value:
        paths.append(item["path"] if isinstance(item, dict) else item)
    return paths

for cfg_path in configs:
    cfg = json.loads(Path(cfg_path).read_text())
    missing = target_ids - set(cfg.get("oracle_forbidden_benchmarks", []))
    if missing:
        raise SystemExit(f"{cfg_path}: missing forbidden target ids: {sorted(missing)}")
    if cfg.get("exclude_eval_protocol") != "configs/eval_protocol_coverage_only.json":
        raise SystemExit(f"{cfg_path}: exclude_eval_protocol must protect eval8")
    for oracle_path in action_paths(cfg.get("oracle_actions", [])):
        if any(part in oracle_path for part in blocked_path_parts):
            raise SystemExit(f"{cfg_path}: oracle path is not allowed: {oracle_path}")
        with Path(oracle_path).open(newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                bid = (row.get("benchmark_id") or "").strip()
                if bid in target_ids:
                    raise SystemExit(f"{cfg_path}: oracle {oracle_path} contains forbidden benchmark {bid}")
    label_path = Path(cfg["labels"])
    with label_path.open(newline="") as f:
        for row in csv.DictReader(f):
            bid = (row.get("benchmark_id") or "").strip()
            if bid in target_ids:
                raise SystemExit(f"{cfg_path}: labels contain forbidden benchmark {bid}")
print("safe training audit passed: no eval8 target ids in labels or oracle files")
PY

run_one() {
  local name=$1
  local config=$2
  local gpu=$3
  local run_dir
  run_dir=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_dir"])' "$config")
  local log_file="$OUT_ROOT/logs/${name}.log"

  if [[ "$FORCE" != "1" && -s "$run_dir/best_final_horizon.pt" ]]; then
    echo "[$(date -Is)] skip existing name=${name} run_dir=${run_dir}" | tee -a "$log_file"
    return 0
  fi

  echo "[$(date -Is)] start name=${name} gpu=${gpu} config=${config}" | tee "$log_file"
  CUDA_VISIBLE_DEVICES="$gpu" python -m tpi_jepa.train --config "$config" 2>&1 | tee -a "$log_file"
  echo "[$(date -Is)] done name=${name} run_dir=${run_dir}" | tee -a "$log_file"
}

active=0
status=0
i=0
for entry in "${CONFIGS[@]}"; do
  name=${entry%% *}
  config=${entry#* }
  if [[ -n "$MODEL_FILTER" && "$name" != *"$MODEL_FILTER"* ]]; then
    continue
  fi
  gpu=${GPUS[$((i % ${#GPUS[@]}))]}
  run_one "$name" "$config" "$gpu" &
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

python - <<'PY'
import csv
import json
from pathlib import Path

rows = []
for cfg_path in [
    "configs/planner_aligned_q_rank_v2_safe.json",
    "configs/planner_aligned_q_rank_v2_seed2_safe.json",
    "configs/planner_aligned_q_rank_v3_ndcg_safe.json",
    "configs/planner_aligned_q_rank_v4_conservative_safe.json",
    "configs/planner_aligned_q_rank_v5_context_safe.json",
    "configs/planner_aligned_reward_rank_v3_safe.json",
    "configs/planner_aligned_reward_rank_v3_seed2_safe.json",
    "configs/planner_aligned_reward_rank_v4_ndcg_safe.json",
    "configs/planner_aligned_reward_rank_v5_context_safe.json",
    "configs/planner_aligned_guarded_reward_rank_v1_safe.json",
    "configs/planner_aligned_guarded_reward_rank_v1_seed2_safe.json",
    "configs/planner_aligned_hybrid_rank_v1_safe.json",
    "configs/planner_aligned_hybrid_rank_v1_seed2_safe.json",
]:
    cfg = json.loads(Path(cfg_path).read_text())
    run_dir = Path(cfg["run_dir"])
    history = run_dir / "history.csv"
    last = {}
    if history.exists():
        with history.open(newline="") as f:
            hist = list(csv.DictReader(f))
        last = hist[-1] if hist else {}
    rows.append({
        "config": cfg_path,
        "run_dir": str(run_dir),
        "score_field": cfg["oracle_ranking_score_field"],
        "seed": cfg["seed"],
        "epochs": cfg["epochs"],
        "best_final_horizon": str((run_dir / "best_final_horizon.pt").exists()),
        "last_epoch": last.get("epoch", ""),
        "last_horizon": last.get("horizon", ""),
        "last_val_loss": last.get("val_loss", ""),
        "last_train_oracle_loss": last.get("train_oracle_loss", ""),
    })
out = Path("autoresearch/improve-260706-0959/train_safe_parallel/summary.tsv")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
print(f"summary: {out}")
PY

exit "$status"
