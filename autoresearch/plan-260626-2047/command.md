# Commands

## Static checks

```bash
python -m py_compile tpi_jepa/model.py tpi_jepa/plan.py tpi_jepa/scoap.py scripts/run_gmean_sweep.py scripts/build_tp_candidate_cache.py
python -m tpi_jepa.plan --help
python scripts/run_gmean_sweep.py --help
python scripts/build_tp_candidate_cache.py --help
```

## Build insertable TP candidate cache

```bash
TPI_BENCH_ROOT=/data4/pengqingsong/DFT/Dataset/deeptpi_official_aig_bench_standard \
python scripts/build_tp_candidate_cache.py \
  --benchmarks b15_C,b17_C,i2c_aig,max_aig,b20_C,b21_C,b22_C,mem_ctrl_aig \
  --out-dir autoresearch/tp-candidates-260626-2047
```

## Cache sanity

```bash
python - <<'PY'
import json
from pathlib import Path
root = Path("autoresearch/tp-candidates-260626-2047")
for path in sorted(root.glob("*.json")):
    payload = json.loads(path.read_text())
    assert payload["candidate_count"] > 0, path
    nets = [item["net"] for item in payload["candidates"]]
    assert len(nets) == len(set(nets)), path
    print(path.stem, payload["candidate_count"], nets[:3], nets[-3:])
PY
```

## b17 cached-stride planner probe

```bash
TPI_BENCH_ROOT=/data4/pengqingsong/DFT/Dataset/deeptpi_official_aig_bench_standard \
CUDA_VISIBLE_DEVICES=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m tpi_jepa.plan \
  --checkpoint autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt \
  --benchmark-id b17_C \
  --budget 5 \
  --max-candidates 64 \
  --device cuda \
  --planner beam \
  --beam-width 2 \
  --lookahead-depth 2 \
  --score-field guarded_reward \
  --beam-objective cumulative \
  --candidate-strategy cached_stride \
  --candidate-cache-dir autoresearch/tp-candidates-260626-2047 \
  --candidate-sample-seed 2026 \
  --out autoresearch/plan-260626-2047/probe_b17_cached_stride.csv
```

## Four-benchmark cached sampling sweep

```bash
OUT_DIR=autoresearch/safe-rollout-cached-tp-260626-2047 \
BENCHMARKS=b15_C,b17_C,i2c_aig,max_aig \
PATTERNS=50000 \
MAX_CANDIDATES=32,64,128 \
LOOKAHEAD_DEPTHS=2 \
BEAM_WIDTHS=2 \
SCORE_FIELDS=guarded_reward \
BEAM_OBJECTIVES=cumulative \
CANDIDATE_STRATEGIES=cached_stride,cached_random \
CANDIDATE_CACHE_DIR=autoresearch/tp-candidates-260626-2047 \
CANDIDATE_SAMPLE_SEEDS=2026,2027,2028 \
CUDA_VISIBLE_DEVICES=4 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash autoresearch/plan-260626-posweight30-world-rollout/run_posweight30_world_rollout.sh
```
