# AutoResearch Plan: b17 Safety-Gated Rollout Selection

generated_at: `2026-06-26 20:02 Asia/Shanghai`

## Goal

```text
Revise checkpoint-driven TPI planner selection so the 50k-pattern held-out gate
passes the b17_C safety threshold before any full 300k-pattern rollout is run.
```

## Current Evidence

Previous plan:

```text
autoresearch/plan-260626-1633/
```

Latest low-cost gate:

```text
autoresearch/safe-rollout-smoke-260626-1633/
```

Observed grouped results:

```text
reward_pred:     macro_mean_delta_tc=+0.0166525  min_delta_tc=-0.0116600  b17_C=-0.0116600  safe=False
guarded_reward: macro_mean_delta_tc=+0.0192775  min_delta_tc=-0.0115300  b17_C=-0.0115300  safe=False
```

Planner-only large-graph smoke now passes for `b22_C` and `mem_ctrl_aig`, so the
next blocker is not OOM. The b17_C plans from both variants concentrate actions
in the same local `N790x` region. That suggests a selection safety issue:
high model score inside one local cone is not sufficient evidence that repeated
actions in that region are safe under Atalanta-BIST evaluation.

## Scope

In scope:

```text
tpi_jepa/plan.py
scripts/run_gmean_sweep.py
autoresearch/plan-260626-2002/
autoresearch/safe-rollout-b17-260626-2002/
```

Allowed changes:

```text
1. Add planner-side guardrails that discourage local over-concentration on b17_C.
2. Add or tune conservative score fields that trade a small macro gain for safety.
3. Run targeted low-cost sweeps before broader held-out gates.
4. Keep the OOM/fast-SCOAP fixes from the previous plan intact.
```

Out of scope:

```text
new model training
new checkpoint selection
Atalanta-BIST backend semantic changes
full 8-benchmark 300k rerun before the 50k safety gate passes
push/publish/deploy
```

## Metric

Primary:

```text
b17_C delta_test_coverage at 50k patterns
```

Safety gate:

```text
b17_C delta_tc >= -0.005
min_delta_tc >= -0.005 on b15_C,b17_C,i2c_aig,max_aig
no plan_error rows
```

Secondary:

```text
macro_mean_delta_tc
positive_count
negative_count
elapsed_sec
plan node diversity on b17_C
```

## Verify

Static checks:

```bash
python -m py_compile tpi_jepa/model.py tpi_jepa/plan.py tpi_jepa/scoap.py scripts/run_gmean_sweep.py
python -m tpi_jepa.plan --help
python scripts/run_gmean_sweep.py --help
```

b17 targeted planner probes:

```bash
TPI_BENCH_ROOT=/data4/pengqingsong/DFT/Dataset/deeptpi_official_aig_bench_standard \
CUDA_VISIBLE_DEVICES=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m tpi_jepa.plan \
  --checkpoint autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt \
  --benchmark-id b17_C \
  --budget 5 \
  --max-candidates 32 \
  --device cuda \
  --planner beam \
  --beam-width 2 \
  --lookahead-depth 2 \
  --score-field guarded_reward \
  --beam-objective cumulative \
  --candidate-strategy hard_fault_cone \
  --candidate-diversity-penalty 0.10 \
  --candidate-diversity-depth 8 \
  --out autoresearch/plan-260626-2002/probe_b17_guarded_div0p10_d8.csv
```

Targeted b17 sweep:

```bash
OUT_DIR=autoresearch/safe-rollout-b17-260626-2002 \
BENCHMARKS=b17_C \
PATTERNS=50000 \
MAX_CANDIDATES=32 \
LOOKAHEAD_DEPTHS=1,2 \
BEAM_WIDTHS=1,2 \
SCORE_FIELDS=guarded_reward,return_pred \
BEAM_OBJECTIVES=mean,terminal,cumulative \
CANDIDATE_DIVERSITY_PENALTIES=0.05,0.10,0.20 \
CANDIDATE_DIVERSITY_DEPTHS=6,8,12 \
CUDA_VISIBLE_DEVICES=4 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash autoresearch/plan-260626-posweight30-world-rollout/run_posweight30_world_rollout.sh
```

Promotion gate after a b17-safe candidate is found:

```bash
OUT_DIR=autoresearch/safe-rollout-smoke-260626-2002 \
BENCHMARKS=b15_C,b17_C,i2c_aig,max_aig \
PATTERNS=50000 \
MAX_CANDIDATES=32 \
LOOKAHEAD_DEPTHS=<winning_depth> \
BEAM_WIDTHS=<winning_width> \
SCORE_FIELDS=<winning_score_field> \
BEAM_OBJECTIVES=<winning_objective> \
CANDIDATE_DIVERSITY_PENALTIES=<winning_diversity_penalty> \
CANDIDATE_DIVERSITY_DEPTHS=<winning_diversity_depth> \
CUDA_VISIBLE_DEVICES=4 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash autoresearch/plan-260626-posweight30-world-rollout/run_posweight30_world_rollout.sh
```

## Success Criteria

Promote to a full 300k rerun only if:

```text
50k four-benchmark gate status is ok
macro_mean_delta_tc > 0.0
min_delta_tc >= -0.005
b17_C delta_tc >= -0.005
negative_count <= 1
no planner OOM or plan_error rows
```

Reject or revise if:

```text
b17_C remains below -0.005
the best b17 variant only works by hurting i2c_aig or max_aig below -0.005
diversity penalties collapse macro_mean_delta_tc <= 0.0
planner probes still choose clustered b17 nodes despite penalties
```

## Proposed Iteration Order

1. Preserve and validate the previous OOM/SCOAP fixes.
2. Run b17 planner probes with diversity penalties and inspect selected node spread.
3. Run the targeted b17 50k sweep.
4. Select the best b17-safe candidate by `b17_C delta_tc`, then by macro-safe proxy signals.
5. Run the four-benchmark 50k promotion gate for that candidate.
6. Only if the promotion gate passes, plan the full 8-benchmark 300k rerun.

## Expected Output

```text
autoresearch/safe-rollout-b17-260626-2002/results.tsv
autoresearch/safe-rollout-b17-260626-2002/grouped_results.tsv
autoresearch/safe-rollout-smoke-260626-2002/results.tsv
autoresearch/safe-rollout-smoke-260626-2002/grouped_results.tsv
```

## Next Command

Start with static checks and the b17 targeted planner probe from the Verify
section. Do not start the full 300k evaluation.
