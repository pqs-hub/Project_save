#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt"
SUBCKTS="autoresearch/oracle-negative-rich-260629/sample/pilot_subckts.txt"
CANDIDATE_CACHE_DIR="autoresearch/tp-candidates-negative-rich-260629"

mkdir -p "${CANDIDATE_CACHE_DIR}"
missing="$(python scripts/list_missing_candidate_cache.py \
  --benchmarks-file "${SUBCKTS}" \
  --candidate-cache-dir "${CANDIDATE_CACHE_DIR}")"
if [[ -n "${missing}" ]]; then
  python scripts/build_hard_tp_candidate_cache.py \
    --benchmarks "${missing}" \
    --out-dir "${CANDIDATE_CACHE_DIR}"
fi

python scripts/oracle_action_value_probe.py \
  --checkpoint "${CHECKPOINT}" \
  --benchmarks "$(paste -sd, "${SUBCKTS}")" \
  --candidate-cache-dir "${CANDIDATE_CACHE_DIR}" \
  --candidate-strategies cached_hard_cone,cached_random,cached_stride \
  --max-nets 12 \
  --action-types CP0,CP1,OP \
  --top-ks 1,3,5 \
  --oracle-top-m 1 \
  --patterns 300000 \
  --backend atalanta-bist \
  --candidate-sample-seed 260629 \
  --out-dir autoresearch/oracle-negative-rich-260629/pilot \
  --resume \
  --cleanup-workdir

python scripts/audit_oracle_action_groups.py \
  --oracle-tsv pilot=autoresearch/oracle-negative-rich-260629/pilot/oracle_actions.tsv \
  --out-dir autoresearch/oracle-negative-rich-260629/pilot_audit
