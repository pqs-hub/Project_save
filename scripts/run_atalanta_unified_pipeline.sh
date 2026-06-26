#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/data3/pengqingsong/DFT/TPI-my.2}
cd "$REPO_ROOT"

TRAIN_PRIOR_DIR=${TRAIN_PRIOR_DIR:-autoresearch/atalanta-train-smallmid-priors-v1}
EVAL_PRIOR_DIR=${EVAL_PRIOR_DIR:-autoresearch/atalanta-eval8-priors-v2}
SEQ_DIR=${SEQ_DIR:-autoresearch/atalanta-seq-smallmid-v1/relabel}
OUT_DIR=${OUT_DIR:-autoresearch/atalanta-unified-hard-fault-cone-v1}
DISTILL_DIR=${DISTILL_DIR:-autoresearch/atalanta-hard-fault-cone-distill-smallmid-v1}
ACT_TRAIN=${ACT_TRAIN:-autoresearch/real-fault-logs-sparse075-top8-train-v1/activation_priors_30k.csv}
ACT_EVAL=${ACT_EVAL:-autoresearch/eval8-real-priors-budget5-v1/activation_priors_30k.csv}

TRAIN_PRIOR_PID_FILE=${TRAIN_PRIOR_PID_FILE:-autoresearch/atalanta-train-smallmid-priors-v1/run.pid}
EVAL_PRIOR_PID_FILE=${EVAL_PRIOR_PID_FILE:-autoresearch/atalanta-eval8-priors-v2/run.pid}
RELABEL_PID_FILE=${RELABEL_PID_FILE:-autoresearch/atalanta-seq-smallmid-v1/relabel.pid}
POLL_SEC=${POLL_SEC:-60}

mkdir -p "$OUT_DIR" "$DISTILL_DIR" "$OUT_DIR/combined_priors"

wait_pid_file() {
  local label=$1
  local file=$2
  if [[ ! -f "$file" ]]; then
    echo "missing pid file for $label: $file" >&2
    exit 1
  fi
  local pid
  pid=$(<"$file")
  echo "[$(date -Is)] waiting for $label pid=$pid"
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$POLL_SEC"
  done
  echo "[$(date -Is)] finished $label"
}

wait_pid_file "train-priors" "$TRAIN_PRIOR_PID_FILE"
wait_pid_file "eval-priors" "$EVAL_PRIOR_PID_FILE"
wait_pid_file "sequence-relabel" "$RELABEL_PID_FILE"

test -f "$TRAIN_PRIOR_DIR/real_fault_priors.csv"
test -f "$EVAL_PRIOR_DIR/real_fault_priors.csv"
test -f "$SEQ_DIR/labels.csv"

eval_label_count=$(find "$EVAL_PRIOR_DIR/fault_logs/benchmarks" -name labels.csv 2>/dev/null | wc -l)
if [[ "$eval_label_count" -lt 8 ]]; then
  echo "eval-priors incomplete: expected at least 8 labels.csv files, got $eval_label_count" >&2
  exit 1
fi

REAL_COMBINED="$OUT_DIR/combined_priors/real_fault_priors_train_plus_eval8.csv"
ACT_COMBINED="$OUT_DIR/combined_priors/activation_priors_30k_train_plus_eval8.csv"
{
  head -n 1 "$TRAIN_PRIOR_DIR/real_fault_priors.csv"
  tail -n +2 "$TRAIN_PRIOR_DIR/real_fault_priors.csv"
  tail -n +2 "$EVAL_PRIOR_DIR/real_fault_priors.csv"
} > "$REAL_COMBINED"
{
  head -n 1 "$ACT_TRAIN"
  tail -n +2 "$ACT_TRAIN"
  tail -n +2 "$ACT_EVAL"
} > "$ACT_COMBINED"

echo "[$(date -Is)] collecting Atalanta hard_fault_cone distill labels"
python scripts/collect_candidate_baseline_labels.py \
  --benchmarks iscas89__s420,iscas89__s838a,iscas85__c2670,epfl__random_control__priority__priority,openabcd__mainpla_orig \
  --candidate-strategy hard_fault_cone \
  --budget 5 \
  --patterns 300000 \
  --backend atalanta-bist \
  --timeout-sec 14400 \
  --parallel-jobs 4 \
  --real-fault-priors "$TRAIN_PRIOR_DIR/real_fault_priors.csv" \
  --activation-priors "$ACT_TRAIN" \
  --out-dir "$DISTILL_DIR" \
  --resume --force --cleanup-workdir

echo "[$(date -Is)] mixing Atalanta sequence and distill labels"
python scripts/build_distill_label_mix.py \
  --base-labels "$SEQ_DIR/labels.csv" \
  --distill-labels "$DISTILL_DIR/labels.csv" \
  --distill-repeat 50 \
  --out "$OUT_DIR/labels_mixed.csv" \
  --manifest "$OUT_DIR/mix_manifest.json"

echo "[$(date -Is)] training Atalanta-aligned hard_fault_cone world model"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6} python scripts/overnight_framework_search.py \
  --base-config autoresearch/sequence10k-train-v1/config_tac_10k.json \
  --labels "$OUT_DIR/labels_mixed.csv" \
  --eval-protocol configs/eval_protocol_coverage_only.json \
  --framework-variants hard_fault_cone \
  --head-context true \
  --latent-dims 64 \
  --encoder-layers 3 \
  --dropouts 0.1 \
  --lambda-returns 0.0 \
  --rollout-horizons 5 \
  --rollout-start-epochs 6 \
  --epochs 20 \
  --max-train-samples 13000 \
  --max-train-steps-per-epoch 2000 \
  --max-val-samples 1024 \
  --max-val-steps 256 \
  --device cuda \
  --devices 0 \
  --parallel-jobs 1 \
  --benchmark-id iscas89__s838 \
  --plan-budget 5 \
  --max-candidates 96 \
  --beam-width 4 \
  --lookahead-depth 3 \
  --plan-score-field reward_pred \
  --plan-beam-objective cumulative \
  --patterns 300000 \
  --timeout-sec 14400 \
  --eval-backend atalanta-bist \
  --real-fault-priors "$TRAIN_PRIOR_DIR/real_fault_priors.csv" \
  --activation-priors "$ACT_TRAIN" \
  --out-dir "$OUT_DIR/train"

echo "[$(date -Is)] evaluating trained model on eval8 with Atalanta_BIST"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6} python scripts/run_gmean_sweep.py \
  --checkpoint "$OUT_DIR/train/best.pt" \
  --benchmarks iscas99__b15_1,iscas99__b20,iscas99__b21,iscas99__b22,epfl__random_control__i2c__i2c,epfl__arithmetic__max__max,iscas99__b17,openabcd__mem_ctrl_orig \
  --budget-mode fixed \
  --fixed-budget 5 \
  --planners beam \
  --score-fields reward_pred \
  --beam-objectives cumulative \
  --beam-widths 4 \
  --lookahead-depths 3 \
  --max-candidates 96 \
  --candidate-strategies hard_fault_cone \
  --candidate-diversity-penalties 0.0 \
  --candidate-diversity-depths 4 \
  --patterns 300000 \
  --seed 2026 \
  --timeout-sec 14400 \
  --time-limit-hours 72 \
  --real-fault-priors "$REAL_COMBINED" \
  --activation-priors "$ACT_COMBINED" \
  --eval-backend atalanta-bist \
  --out-dir "$OUT_DIR/eval8"
