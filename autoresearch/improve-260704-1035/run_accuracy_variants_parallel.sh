#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="autoresearch/improve-260704-1035/parallel_accuracy_variants"
mkdir -p "${OUT_DIR}/logs" "${OUT_DIR}/accuracy"

VARIANTS=(
  "v1_balanced|4|configs/mainline_accuracy_improve_v1.json|runs/mainline_accuracy_improve_v1"
  "v2_reward_return|5|configs/mainline_accuracy_improve_v2_reward_return.json|runs/mainline_accuracy_improve_v2_reward_return"
  "v3_hard_precision|6|configs/mainline_accuracy_improve_v3_hard_precision.json|runs/mainline_accuracy_improve_v3_hard_precision"
  "v4_reduction_sign|7|configs/mainline_accuracy_improve_v4_reduction_sign.json|runs/mainline_accuracy_improve_v4_reduction_sign"
)

run_one() {
  local name="$1"
  local gpu="$2"
  local config="$3"
  local run_dir="$4"
  local train_log="${OUT_DIR}/logs/${name}.train.log"
  local eval_log="${OUT_DIR}/logs/${name}.eval.log"
  local acc_dir="${OUT_DIR}/accuracy/${name}"

  {
    echo "[$(date -Is)] ${name} gpu=${gpu} config=${config}"
    CUDA_VISIBLE_DEVICES="${gpu}" python -m tpi_jepa.train --config "${config}"
    echo "[$(date -Is)] ${name} train_done checkpoint=${run_dir}/best.pt"
    CUDA_VISIBLE_DEVICES="${gpu}" python scripts/evaluate_trained_head_accuracy.py \
      --checkpoint "${run_dir}/best.pt" \
      --config "${config}" \
      --max-samples 4096 \
      --device cuda \
      --require-cuda \
      --out-dir "${acc_dir}"
    echo "[$(date -Is)] ${name} eval_done accuracy=${acc_dir}/trained_head_accuracy.tsv"
  } 2>&1 | stdbuf -oL sed -u "s/^/[${name} gpu${gpu}] /" | tee "${train_log}"

  cp "${train_log}" "${eval_log}"
}

pids=()
for spec in "${VARIANTS[@]}"; do
  IFS="|" read -r name gpu config run_dir <<< "${spec}"
  run_one "${name}" "${gpu}" "${config}" "${run_dir}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

python - <<'PY'
import csv
from pathlib import Path

root = Path("autoresearch/improve-260704-1035/parallel_accuracy_variants")
rows = []
for path in sorted(root.glob("accuracy/*/trained_head_accuracy.tsv")):
    variant = path.parent.name
    with path.open(newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            row = dict(row)
            row["variant"] = variant
            rows.append(row)

fields = [
    "variant",
    "metric_type",
    "task",
    "n",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "positive_rate",
    "pred_positive_rate",
]
out = root / "accuracy_summary.tsv"
with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
print(out)
PY

exit "${status}"
