# AutoResearch Plan: 8-Circuit TC Lift With Per-Circuit Safe Routing

generated_at: `2026-06-28 11:36 Asia/Shanghai`

## Goal

Improve real test-coverage lift on the 8 fixed evaluation circuits while
reducing large negative per-circuit regressions.

The next step should not be another hard-F1-only training sweep. The current
evidence says representation quality is usable, but planner behavior is
unstable across circuits.

## Current Evidence

Best representation checkpoints are already strong:

```text
highseed baseline best:
hard_macro_f1_tuned=0.7980429023392821
predictive_score=0.8218421546846536

posweight-30 best:
hard_macro_f1_tuned=0.7909739745915929
predictive_score=0.8170630360809562
```

Planner results are mixed:

```text
hard_fault_cone 4-circuit smoke:
macro_mean_delta_tc=+0.0192775
min_delta_tc=-0.01153
negative_count=1

cached_hard_cone single-circuit results:
b15_C   +0.03035
b17_C   -0.01137
i2c_aig -0.01134
max_aig +0.04238

hard_fault_recall_union best8 partial run:
b15_C   +0.07069
b20_C   +0.00043
b21_C   +0.00020
i2c_aig -0.01185
max_aig +0.02620
b17_C   plan_error
b22_C   plan_error
mem_ctrl plan_error
partial macro on completed circuits=+0.017134
```

Interpretation:

```text
The model can find high-upside actions on b15_C and max_aig, but the same
global planner/candidate policy is unsafe on i2c_aig and b17_C and can fail on
large circuits. The next improvement should be a circuit-aware planner routing
gate, not a new global model change.
```

## Scope

In scope:

```text
scripts/run_gmean_sweep.py
tpi_jepa/plan.py
scripts/candidate_recall_diagnostics.py
scripts/plan_candidate_baseline.py
autoresearch/plan-260628-1136/
autoresearch/tp-candidates-260626-2047/
```

Allowed work:

```text
1. Diagnose candidate coverage and planner completion per benchmark.
2. Run low-cost 50k-pattern gates on the unstable circuits first.
3. Promote only configurations that complete and meet min-delta safety.
4. Compose a per-circuit route table:
   - aggressive route for circuits where high-upside planner is safe
   - conservative netlist/cached_stride fallback where aggressive route regresses
5. Run the final 8-circuit fixed-budget-5 evaluation with the route table.
```

Out of scope:

```text
new training loss changes
new checkpoint training
Atalanta-BIST backend edits
claiming final paper numbers from 50k-pattern gates
push/publish/deploy
```

## Primary Hypothesis

The framework will improve 8-circuit TC more by selecting the right planner per
circuit than by changing the world model.

Expected route shape:

```text
b15_C      aggressive: hard_fault_recall_union or cached_hard_cone
max_aig    aggressive: cached_hard_cone or hard_fault_recall_union
b20_C      conservative/aggressive both likely near zero; pick safest completed route
b21_C      conservative/aggressive both likely near zero; pick safest completed route
i2c_aig    conservative fallback unless a 50k gate clears min_delta
b17_C      conservative fallback unless plan_error and negative-delta issues are fixed
b22_C      completion-first route; no promotion until plan_error is gone
mem_ctrl   completion-first route; no promotion until plan_error is gone
```

## Metrics

Primary metric:

```text
macro_mean_delta_tc across all 8 circuits at patterns=300000
```

Promotion gate:

```text
status=ok on all 8 circuits
macro_mean_delta_tc > incumbent_macro_mean_delta_tc
min_delta_tc >= -0.005
negative_count <= 1
no plan_error rows
```

Circuit-level gate for 50k smoke:

```text
delta_test_coverage >= -0.005
plan completes within timeout
plan has exactly budget inserted points
```

Secondary diagnostics:

```text
candidate_node_recall_at_32/64/128
candidate_action_recall_at_32/64/128
plan elapsed_sec
plan_score_sum vs real delta_test_coverage
per-circuit selected nets and action types
```

## Verify

Static checks:

```bash
python -m py_compile tpi_jepa/*.py scripts/*.py
python -m tpi_jepa.plan --help
python scripts/run_gmean_sweep.py --help
python scripts/candidate_recall_diagnostics.py --help
```

Candidate recall diagnostics:

```bash
for strategy in netlist testability hard_fault_cone hard_fault_recall_union recall_pool; do
  TPI_BENCH_ROOT=/data4/pengqingsong/DFT/Dataset/deeptpi_official_aig_bench_standard \
  python scripts/candidate_recall_diagnostics.py \
    --candidate-strategy "$strategy" \
    --top-k 32,64,128,256 \
    --max-sequences 1024 \
    --details \
    > autoresearch/plan-260628-1136/candidate_recall_${strategy}.txt
done
```

Current `candidate_recall_diagnostics.py` does not expose
`--candidate-cache-dir`, so diagnose cached strategies only after adding that
flag and passing `benchmark_id` plus `candidate_cache_dir` through to
`enumerate_candidates()`.

50k unstable-circuit gate:

```bash
TPI_BENCH_ROOT=/data4/pengqingsong/DFT/Dataset/deeptpi_official_aig_bench_standard \
python scripts/run_gmean_sweep.py \
  --checkpoint autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt \
  --benchmarks b17_C,b22_C,i2c_aig,mem_ctrl_aig \
  --budget-mode fixed \
  --fixed-budget 5 \
  --planners beam \
  --beam-objectives cumulative,discounted \
  --score-fields guarded_reward,reward_pred \
  --beam-widths 2,4 \
  --lookahead-depths 2,3 \
  --max-candidates 32,64,128 \
  --candidate-strategies netlist,cached_stride,cached_hard_cone,hard_fault_recall_union \
  --candidate-cache-dir autoresearch/tp-candidates-260626-2047 \
  --candidate-sample-seeds 0,2026 \
  --patterns 50000 \
  --seed 2026 \
  --plan-device cuda \
  --eval-backend atalanta-bist \
  --timeout-sec 14400 \
  --time-limit-hours 12 \
  --out-dir autoresearch/tc-lift-unstable-gate-260628-1136
```

50k upside confirmation:

```bash
TPI_BENCH_ROOT=/data4/pengqingsong/DFT/Dataset/deeptpi_official_aig_bench_standard \
python scripts/run_gmean_sweep.py \
  --checkpoint autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt \
  --benchmarks b15_C,max_aig,b20_C,b21_C \
  --budget-mode fixed \
  --fixed-budget 5 \
  --planners beam \
  --beam-objectives cumulative \
  --score-fields reward_pred,guarded_reward \
  --beam-widths 2,4 \
  --lookahead-depths 2,3 \
  --max-candidates 64,128,256 \
  --candidate-strategies cached_hard_cone,hard_fault_recall_union,cached_stride \
  --candidate-cache-dir autoresearch/tp-candidates-260626-2047 \
  --candidate-sample-seeds 0,2026 \
  --patterns 50000 \
  --seed 2026 \
  --plan-device cuda \
  --eval-backend atalanta-bist \
  --timeout-sec 14400 \
  --time-limit-hours 12 \
  --out-dir autoresearch/tc-lift-upside-gate-260628-1136
```

Final 8-circuit route-table evaluation:

```text
Build a route table from the two 50k gates. For each circuit choose the highest
delta route that satisfies delta >= -0.005 and has status=ok. If no aggressive
route passes, use netlist/cached_stride fallback.
```

Then run 300k-pattern evaluation:

```bash
TPI_BENCH_ROOT=/data4/pengqingsong/DFT/Dataset/deeptpi_official_aig_bench_standard \
python scripts/run_gmean_sweep.py \
  --checkpoint autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt \
  --benchmarks b15_C,b20_C,b21_C,b22_C,i2c_aig,max_aig,b17_C,mem_ctrl_aig \
  --budget-mode fixed \
  --fixed-budget 5 \
  --planners beam \
  --beam-objectives cumulative \
  --score-fields guarded_reward \
  --beam-widths 2 \
  --lookahead-depths 2 \
  --max-candidates 64 \
  --candidate-strategies cached_stride \
  --candidate-cache-dir autoresearch/tp-candidates-260626-2047 \
  --candidate-sample-seeds 2026 \
  --patterns 300000 \
  --seed 2026 \
  --plan-device cuda \
  --eval-backend atalanta-bist \
  --timeout-sec 14400 \
  --time-limit-hours 24 \
  --out-dir autoresearch/tc-lift-8ckt-fallback-baseline-260628-1136
```

Use the fallback baseline as the floor. After route-table support is available,
rerun the same 8 circuits with per-circuit routes and compare against it.

## Decision Rules

Keep a route if:

```text
status=ok
delta_test_coverage >= -0.005 at 50k
no repeated plan_error on that circuit
selected plan has exactly budget rows
```

Reject or quarantine a route if:

```text
plan_error appears on any large circuit
delta_test_coverage < -0.005 on i2c_aig or b17_C
macro gain comes only from b15_C/max_aig while more than one circuit regresses
```

## Expected Outcome

The likely near-term improvement is not a new single global best planner. It is
a routed policy:

```text
high-upside planner for b15_C and max_aig
safe fallback for i2c_aig and b17_C
completion-first fallback for b22_C and mem_ctrl_aig
neutral safest route for b20_C and b21_C
```

If route-table evaluation passes, the next implementation step is to add a
first-class per-benchmark planner configuration file so the final command can
run one coherent 8-circuit experiment instead of manually stitching runs.
