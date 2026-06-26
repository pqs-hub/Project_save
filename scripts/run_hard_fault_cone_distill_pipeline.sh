#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/data3/pengqingsong/DFT/TPI-my.2}
cd "$REPO_ROOT"

LABEL_DIR=${LABEL_DIR:-autoresearch/hard-fault-cone-distill-labels-v1}
TRAIN_DIR=${TRAIN_DIR:-autoresearch/hard-fault-cone-distill-train-v1}
BASE_LABELS=${BASE_LABELS:-autoresearch/sequence10k-train-v1/labels.csv}
COLLECT_PID_FILE=${COLLECT_PID_FILE:-$LABEL_DIR/collect.pid}
DISTILL_REPEAT=${DISTILL_REPEAT:-50}
POLL_SEC=${POLL_SEC:-60}

REAL_PRIORS=${REAL_PRIORS:-autoresearch/combined-priors-train-plus-s838-v1/real_fault_priors_train_plus_s838.csv}
ACTIVATION_PRIORS=${ACTIVATION_PRIORS:-autoresearch/combined-priors-train-plus-s838-v1/activation_priors_30k_train_plus_s838.csv}

if [[ ! -f "$COLLECT_PID_FILE" ]]; then
  echo "missing collect pid file: $COLLECT_PID_FILE" >&2
  exit 1
fi

COLLECT_PID=$(<"$COLLECT_PID_FILE")
mkdir -p "$TRAIN_DIR"
echo "[$(date -Is)] waiting for label collection pid=$COLLECT_PID"
while kill -0 "$COLLECT_PID" 2>/dev/null; do
  sleep "$POLL_SEC"
done
echo "[$(date -Is)] label collection finished"

DISTILL_LABELS="$LABEL_DIR/labels.csv"
if [[ ! -f "$DISTILL_LABELS" ]]; then
  echo "distill labels missing: $DISTILL_LABELS" >&2
  exit 1
fi

python scripts/build_distill_label_mix.py \
  --base-labels "$BASE_LABELS" \
  --distill-labels "$DISTILL_LABELS" \
  --distill-repeat "$DISTILL_REPEAT" \
  --out "$TRAIN_DIR/labels_mixed.csv" \
  --manifest "$TRAIN_DIR/mix_manifest.json"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6} python scripts/overnight_framework_search.py \
  --base-config autoresearch/sequence10k-train-v1/config_tac_10k.json \
  --labels "$TRAIN_DIR/labels_mixed.csv" \
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
  --patterns 1000 \
  --timeout-sec 7200 \
  --real-fault-priors "$REAL_PRIORS" \
  --activation-priors "$ACTIVATION_PRIORS" \
  --out-dir "$TRAIN_DIR"
