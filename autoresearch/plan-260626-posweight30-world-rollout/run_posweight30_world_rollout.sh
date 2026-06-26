#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

STAMP="$(date +%y%m%d-%H%M%S)"
OUT_DIR="${OUT_DIR:-autoresearch/posweight30-world-rollout-260626-run-${STAMP}}"

export DFT_ROOT="${DFT_ROOT:-/data4/pengqingsong/DFT}"
export TPI_BENCH_ROOT="${TPI_BENCH_ROOT:-/data4/pengqingsong/DFT/Dataset/deeptpi_official_aig_bench_standard}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER}}"
mkdir -p "$MPLCONFIGDIR"

CHECKPOINT="${CHECKPOINT:-autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt}"
ATALANTA_BIN="${ATALANTA_BIN:-/data4/pengqingsong/DFT/tool/atalanta_bist_with_ufaults/atalanta}"

BENCHMARKS="${BENCHMARKS:-b15_C,b17_C,b20_C,b21_C,b22_C,i2c_aig,max_aig,mem_ctrl_aig}"
PATTERNS="${PATTERNS:-300000}"
SEED="${SEED:-2026}"
TIMEOUT_SEC="${TIMEOUT_SEC:-14400}"
TIME_LIMIT_HOURS="${TIME_LIMIT_HOURS:-24}"
PLAN_DEVICE="${PLAN_DEVICE:-cuda}"
FIXED_BUDGET="${FIXED_BUDGET:-5}"

PLANNER="${PLANNER:-beam}"
SCORE_FIELDS="${SCORE_FIELDS:-reward_pred}"
BEAM_OBJECTIVES="${BEAM_OBJECTIVES:-cumulative}"
BEAM_WIDTHS="${BEAM_WIDTHS:-4}"
LOOKAHEAD_DEPTHS="${LOOKAHEAD_DEPTHS:-3}"
MAX_CANDIDATES="${MAX_CANDIDATES:-96}"
CANDIDATE_STRATEGIES="${CANDIDATE_STRATEGIES:-hard_fault_cone}"
CANDIDATE_DIVERSITY_PENALTIES="${CANDIDATE_DIVERSITY_PENALTIES:-0.0}"
CANDIDATE_DIVERSITY_DEPTHS="${CANDIDATE_DIVERSITY_DEPTHS:-4}"
CANDIDATE_CACHE_DIR="${CANDIDATE_CACHE_DIR:-}"
CANDIDATE_SAMPLE_SEEDS="${CANDIDATE_SAMPLE_SEEDS:-0}"
SAFETY_MIN_DELTA="${SAFETY_MIN_DELTA:--0.005}"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "missing checkpoint: $CHECKPOINT" >&2
  exit 1
fi
if [[ ! -d "$TPI_BENCH_ROOT" ]]; then
  echo "missing TPI_BENCH_ROOT: $TPI_BENCH_ROOT" >&2
  exit 1
fi
if [[ ! -x "$ATALANTA_BIN" ]]; then
  echo "missing executable ATALANTA_BIN: $ATALANTA_BIN" >&2
  exit 1
fi

ARGS=(
  --checkpoint "$CHECKPOINT"
  --benchmarks "$BENCHMARKS"
  --budget-mode fixed
  --fixed-budget "$FIXED_BUDGET"
  --planners "$PLANNER"
  --score-fields "$SCORE_FIELDS"
  --beam-objectives "$BEAM_OBJECTIVES"
  --beam-widths "$BEAM_WIDTHS"
  --lookahead-depths "$LOOKAHEAD_DEPTHS"
  --max-candidates "$MAX_CANDIDATES"
  --candidate-strategies "$CANDIDATE_STRATEGIES"
  --candidate-diversity-penalties "$CANDIDATE_DIVERSITY_PENALTIES"
  --candidate-diversity-depths "$CANDIDATE_DIVERSITY_DEPTHS"
  --candidate-sample-seeds "$CANDIDATE_SAMPLE_SEEDS"
  --plan-device "$PLAN_DEVICE"
  --eval-backend atalanta-bist
  --atalanta-bin "$ATALANTA_BIN"
  --patterns "$PATTERNS"
  --seed "$SEED"
  --timeout-sec "$TIMEOUT_SEC"
  --time-limit-hours "$TIME_LIMIT_HOURS"
  --safety-benchmark b17_C
  --safety-min-delta "$SAFETY_MIN_DELTA"
  --out-dir "$OUT_DIR"
)

if [[ -n "$CANDIDATE_CACHE_DIR" ]]; then
  ARGS+=(--candidate-cache-dir "$CANDIDATE_CACHE_DIR")
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  ARGS+=(--dry-run)
fi

echo "[posweight30-world-rollout] out_dir=$OUT_DIR"
echo "[posweight30-world-rollout] checkpoint=$CHECKPOINT"
echo "[posweight30-world-rollout] benchmarks=$BENCHMARKS"
echo "[posweight30-world-rollout] budget=$FIXED_BUDGET patterns=$PATTERNS backend=atalanta-bist plan_device=$PLAN_DEVICE cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"

python scripts/run_gmean_sweep.py "${ARGS[@]}"

echo "[posweight30-world-rollout] wrote_results=$OUT_DIR/results.tsv"
echo "[posweight30-world-rollout] wrote_grouped=$OUT_DIR/grouped_results.tsv"
