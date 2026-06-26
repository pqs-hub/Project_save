# AutoResearch Plan: Cached Insertable-TP Candidate Sampling

generated_at: `2026-06-26 20:47 Asia/Shanghai`

## Goal

```text
Replace the current netlist-order candidate shortcut with a cached set of
backend-valid insertable test-point nodes, then sample from that cache during
planning so model scoring can search the full AIG instead of only the first
legal netlist nodes.
```

## Current Evidence

Previous plan:

```text
autoresearch/plan-260626-2002/
```

b17 candidate-strategy sweep:

```text
autoresearch/safe-rollout-b17-candidates-260626-2002-par/ranked_summary.tsv
```

Key result:

```text
candidate_strategy=netlist: b17_C delta_tc=+0.00003
candidate_strategy=testability: b17_C delta_tc=-0.01023
candidate_strategy=reconvergence: b17_C delta_tc=-0.01060
candidate_strategy=ffr_hier: b17_C delta_tc=-0.01083
candidate_strategy=mixed: b17_C delta_tc=-0.01113
candidate_strategy=ffr: b17_C delta_tc=-0.01158
```

Four-benchmark netlist promotion gate:

```text
autoresearch/safe-rollout-smoke-260626-2002-par/runs/*/results.tsv
```

Observed result:

```text
b15_C delta_tc=+0.00006
b17_C delta_tc=+0.00003
i2c_aig delta_tc=+0.00006
max_aig delta_tc=+0.00021
macro_mean_delta_tc ~= +0.00009
min_delta_tc=+0.00003
safe=True
```

Interpretation:

```text
The netlist strategy is safe but too conservative because it currently takes
the first max_candidates legal graph nodes in parsed netlist order. It is not
sampling across all backend-valid insertable TP nodes. We need to cache the
true insertable set once per benchmark, then sample from that cache with stable
coverage over the whole AIG.
```

## Source of Insertable TP Rules

Use the existing evaluator-side candidate rule:

```text
/data4/pengqingsong/DFT/Dataset/tpi_eval/candidates.py
```

Relevant behavior:

```text
generate_candidates(circuit):
  iterates circuit.assignments
  candidate net = assignment.output
  skips circuit inputs and outputs
  skips special nets containing clock/clk/reset/rst/scan/test
  skips constants
  records kind, driver, fanout, and stable id/order
```

This should be treated as the backend-valid insertable TP source, because
`tpi_eval/insertion.py` applies test points to selected net names.

## Scope

In scope:

```text
scripts/build_tp_candidate_cache.py
tpi_jepa/plan.py
scripts/run_gmean_sweep.py
autoresearch/plan-260626-2047/
autoresearch/tp-candidates-260626-2047/
```

Allowed changes:

```text
1. Add a script that builds per-benchmark insertable TP candidate cache JSON.
2. Add planner options to read candidate caches.
3. Add cached candidate strategies:
   - cached_netlist: deterministic cached order
   - cached_stride: evenly strided sampling over the cached insertable set
   - cached_random: seeded random sampling over the cached insertable set
4. Keep action expansion as control0/control1/observe for each cached net.
5. Continue using model scoring/beam rollout after cache-based sampling.
```

Out of scope:

```text
new model training
new checkpoint selection
Atalanta-BIST backend semantic changes
full 300k rerun before cached 50k gate passes
push/publish/deploy
```

## Metric

Primary:

```text
macro_mean_delta_tc on b15_C,b17_C,i2c_aig,max_aig at 50k patterns
```

Safety gate:

```text
min_delta_tc >= -0.005
b17_C delta_tc >= -0.005
no plan_error rows
```

Secondary:

```text
positive_count
negative_count
candidate_cache coverage
candidate node position spread
elapsed_sec
```

## Verify

Static checks:

```bash
python -m py_compile tpi_jepa/model.py tpi_jepa/plan.py tpi_jepa/scoap.py scripts/run_gmean_sweep.py scripts/build_tp_candidate_cache.py
python -m tpi_jepa.plan --help
python scripts/run_gmean_sweep.py --help
python scripts/build_tp_candidate_cache.py --help
```

Build cache:

```bash
TPI_BENCH_ROOT=/data4/pengqingsong/DFT/Dataset/deeptpi_official_aig_bench_standard \
python scripts/build_tp_candidate_cache.py \
  --benchmarks b15_C,b17_C,i2c_aig,max_aig,b20_C,b21_C,b22_C,mem_ctrl_aig \
  --out-dir autoresearch/tp-candidates-260626-2047
```

Cache sanity:

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

Planner probes:

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

Targeted 50k sweep:

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

## Required Implementation Notes

```text
1. Do not use graph.node_names as the source of insertable points for cached strategies.
2. Cache net names must match graph node names at plan time; missing nets should be counted and skipped with a warning.
3. cached_stride should cover the full cache:
   - choose roughly max_candidates/action_type_count nets spread across candidate order
   - expand each selected net to control0/control1/observe
   - preserve deterministic behavior
4. cached_random should use a stable seed derived from candidate_sample_seed, benchmark_id, and planning step so repeated runs are reproducible.
5. run_gmean_sweep.py must pass candidate cache options through to tpi_jepa.plan.
6. Keep the previous include_aux_heads=False and fast SCOAP changes intact.
```

## Success Criteria

Promote to full 300k only if cached candidate sampling passes:

```text
status ok on b15_C,b17_C,i2c_aig,max_aig
macro_mean_delta_tc > +0.005
min_delta_tc >= -0.005
b17_C delta_tc >= -0.005
negative_count <= 1
no planner OOM or plan_error rows
```

Reject or revise if:

```text
cached strategies are safe but macro_mean_delta_tc remains near zero
b17_C falls below -0.005 again
cache misses remove a large fraction of candidates
candidate sampling collapses to one small netlist region
```

## Proposed Iteration Order

1. Implement and validate candidate cache builder.
2. Add cache-aware planner options and cached strategies.
3. Add run_gmean_sweep pass-through arguments.
4. Build caches for the 8 official AIG benchmarks.
5. Run b17 planner probes and inspect node spread.
6. Run four-benchmark 50k cached sampling sweep.
7. Only if gate passes with meaningful macro gain, plan the 300k rerun.

## Expected Output

```text
autoresearch/tp-candidates-260626-2047/*.json
autoresearch/plan-260626-2047/probe_b17_cached_stride.csv
autoresearch/safe-rollout-cached-tp-260626-2047/results.tsv
autoresearch/safe-rollout-cached-tp-260626-2047/grouped_results.tsv
```

## Next Command

Start by implementing `scripts/build_tp_candidate_cache.py`, planner cache
options, and sweep pass-through flags. Then run the static checks and cache
sanity commands above.
