# AutoResearch Plan: Safe Rollout Planner Improvement

generated_at: `2026-06-26 16:33 Asia/Shanghai`

## Goal

```text
Improve the current checkpoint-driven TPI planner so rollout selection is
measurably safer and more stable on held-out benchmarks before another full
300k-pattern Atalanta-BIST run.
```

## Current Evidence

Latest rollout:

```text
autoresearch/posweight30-world-rollout-260626-run-260626-140347/
```

Observed result:

```text
completed benchmarks: 6 / 8
plan errors: b22_C, mem_ctrl_aig
macro_mean_delta_tc: +0.009243
min_delta_tc: -0.031610
safety benchmark b17_C: -0.010880
safe: False
```

Interpretation:

```text
The run did not train. It loaded an existing checkpoint and used beam rollout
to select actions by reward_pred, then evaluated those actions with Atalanta-BIST.
The selection policy is not stable enough: b17_C and i2c_aig regress, while
large benchmarks also hit planner-side CUDA OOM.
```

## Scope

In scope:

```text
tpi_jepa/plan.py
scripts/run_gmean_sweep.py
autoresearch/plan-260626-1633/
```

Allowed changes:

```text
1. Add a memory-safe planner mode or fallback for large graphs.
2. Add an action-selection safety guard that can use conservative scoring
   instead of pure reward_pred.
3. Add a small regression command that evaluates planner variants before the
   expensive 300k-pattern run.
```

Out of scope:

```text
new model architecture training
changing Atalanta-BIST backend semantics
publishing or pushing without explicit approval
full 8-benchmark 300k rerun as the first step
```

## Metric

Primary:

```text
macro_mean_delta_tc across completed held-out benchmarks
```

Safety gate:

```text
min_delta_tc >= -0.005
router/safety benchmark b17_C delta_tc >= -0.005
no planner OOM on b22_C and mem_ctrl_aig smoke planning
```

Secondary:

```text
positive_count
negative_count
plan_error count
elapsed_sec
```

## Verify

Static checks:

```bash
python -m py_compile tpi_jepa/plan.py scripts/run_gmean_sweep.py
python -m tpi_jepa.plan --help
python scripts/run_gmean_sweep.py --help
```

Planner-only OOM smoke:

```bash
CUDA_VISIBLE_DEVICES=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m tpi_jepa.plan \
  --checkpoint autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt \
  --benchmark-id b22_C \
  --budget 5 \
  --max-candidates 32 \
  --device cuda \
  --planner beam \
  --beam-width 2 \
  --lookahead-depth 2 \
  --score-field reward_pred \
  --beam-objective cumulative \
  --candidate-strategy hard_fault_cone \
  --out autoresearch/plan-260626-1633/smoke_b22_plan.csv
```

Low-cost rollout gate:

```bash
OUT_DIR=autoresearch/safe-rollout-smoke-260626-1633 \
BENCHMARKS=b15_C,b17_C,i2c_aig,max_aig \
PATTERNS=50000 \
MAX_CANDIDATES=32 \
LOOKAHEAD_DEPTHS=2 \
BEAM_WIDTHS=2 \
CUDA_VISIBLE_DEVICES=4 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash autoresearch/plan-260626-posweight30-world-rollout/run_posweight30_world_rollout.sh
```

## Success Criteria

Promote to full 300k rerun only if the low-cost gate meets all of:

```text
status is ok or no plan_error rows
macro_mean_delta_tc > 0.0
min_delta_tc >= -0.005
b17_C delta_tc >= -0.005
negative_count <= 1
```

Reject or revise if:

```text
b17_C remains below -0.005
i2c_aig remains below -0.02
large-graph planner smoke still OOMs
the policy only improves b15_C while hurting smaller/generalization circuits
```

## Proposed Iteration Order

1. Reproduce planner-only OOM with current settings on `b22_C`.
2. Add a memory-safe planning path:
   - lower candidate batch pressure for large graphs,
   - avoid computing unused heads when `score_field=reward_pred`,
   - keep CPU fallback available for smoke.
3. Add a conservative planner variant for selection:
   - compare `reward_pred`, `return_pred`, and a guarded score that penalizes negative or unstable return estimates.
4. Run the 50k-pattern gate on `b15_C,b17_C,i2c_aig,max_aig`.
5. Only if gate passes, run the full 8-benchmark 300k-pattern evaluation.

## Expected Output

```text
autoresearch/safe-rollout-smoke-260626-1633/results.tsv
autoresearch/safe-rollout-smoke-260626-1633/grouped_results.tsv
```

## Next Command

Start with the static checks and planner-only OOM smoke from the Verify section.
