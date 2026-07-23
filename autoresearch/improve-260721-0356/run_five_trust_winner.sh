#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
SELECT=autoresearch/improve-260721-0356/b15_trust_selection/winner.json
OUT=autoresearch/improve-260721-0356/trust_round4_five
test -f "$SELECT"
mapfile -t selected < <(python -c '
import json,sys
w=json.load(open(sys.argv[1]))["winner"]
for key in ("checkpoint", "score_field", "typed_residual_alpha", "typed_residual_decay_steps", "typed_trust_min_heads", "typed_trust_cp0_min_heads", "typed_trust_head_margin", "typed_trust_advantage_margin"):
    print(w[key])
' "$SELECT")
CHECKPOINT=${selected[0]}
SCORE_FIELD=${selected[1]}
TPI_TYPED_RESIDUAL_ALPHA=${selected[2]}
TPI_TYPED_RESIDUAL_DECAY_STEPS=${selected[3]}
TPI_TYPED_TRUST_MIN_HEADS=${selected[4]}
TPI_TYPED_TRUST_CP0_MIN_HEADS=${selected[5]}
TPI_TYPED_TRUST_HEAD_MARGIN=${selected[6]}
TPI_TYPED_TRUST_ADVANTAGE_MARGIN=${selected[7]}
mkdir -p "$OUT/logs"

export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 TPI_TORCH_DETERMINISTIC=1
export TPI_HARD_CLUSTER_MAX_HARD_NODES=1024 TPI_LATENT_NORM_CLIP_RATIO=4
export TPI_Q_CONTEXT_SUPPORT_ALPHA=0.45 TPI_Q_CONTEXT_DISAGREEMENT_BETA=0.10
export TPI_TYPED_RESIDUAL_ALPHA TPI_TYPED_RESIDUAL_DECAY_STEPS
export TPI_TYPED_TRUST_MIN_HEADS TPI_TYPED_TRUST_CP0_MIN_HEADS
export TPI_TYPED_TRUST_HEAD_MARGIN TPI_TYPED_TRUST_ADVANTAGE_MARGIN
export TPI_TYPED_RESIDUAL_CLIP=1.0 TPI_TYPED_RESIDUAL_DISAGREEMENT_BETA=0.25
export TPI_SCORE_QUANTIZATION=0.001 TPI_PLAN_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

circuits=(b15_C b20_C b21_C b22_C b17_C)
benchmarks=(iscas99__b15_1 iscas99__b20 iscas99__b21 iscas99__b22 iscas99__b17)
gpus=(0 1 2 4 5)

run_one() {
  local circuit=$1 benchmark=$2 gpu=$3
  echo "[trust-five] start circuit=$circuit gpu=$gpu checkpoint=$CHECKPOINT score=$SCORE_FIELD"
  CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/run_gmean_sweep.py \
    --eval-protocol configs/eval_protocol_coverage_only.json --protocol-keep-cli-benchmarks \
    --benchmarks "$benchmark" --checkpoint "$CHECKPOINT" \
    --planners greedy --score-fields "$SCORE_FIELD" \
    --beam-objectives cumulative --beam-widths 1 --lookahead-depths 1 \
    --max-candidates 48 --discount-gammas 0.9 \
    --candidate-strategies hard_fault_cluster --candidate-diversity-penalties 0.0 \
    --candidate-diversity-depths 4 \
    --candidate-real-fault-priors autoresearch/eval8-real-priors-budget5-v1/real_fault_priors.csv \
    --candidate-allowlist "autoresearch/original-netlist-recovery-260712/exact_itc99/$circuit/exact_candidate_nodes.txt" \
    --plan-device cuda --time-limit-hours 72 --stream-logs --out-dir "$OUT/$circuit" \
    2>&1 | sed -u "s|^|[trust-five/$circuit] |" | tee "$OUT/logs/$circuit.log"
  echo "[trust-five] done circuit=$circuit gpu=$gpu"
}

pids=()
for index in "${!circuits[@]}"; do
  run_one "${circuits[$index]}" "${benchmarks[$index]}" "${gpus[$index]}" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
if (( failed )); then exit 1; fi

python scripts/summarize_exact_itc99_eval.py --eval-root "$OUT"
python scripts/verify_uniform_exact_itc99.py "$OUT"
python scripts/check_deeptpi_goal.py "$OUT"

