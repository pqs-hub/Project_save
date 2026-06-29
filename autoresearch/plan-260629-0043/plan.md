# AutoResearch Plan: Oracle Action-Value Probe for 8-Circuit TC Lift

generated_at: `2026-06-29 00:43 Asia/Shanghai`

## Goal

Replace historical-action candidate recall with an oracle/action-value probe
that measures both:

```text
1. whether the candidate/planner stack can surface actions that actually
   improve test coverage on the 8 evaluation circuits
2. whether the world model's predicted action收益 is aligned with the real
   backend-measured TC收益
```

The previous `candidate_recall_diagnostics.py` answers only:

```text
Does the candidate pool contain the action that appeared in labels.csv?
```

That is not the right target for improving final TC, because those logged
actions are not guaranteed to be optimal. The next plan must label actions by
real backend-measured value.

## Core Claim

Coverage of historical actions is a weak sanity check, not an optimization
metric.

The new diagnostic should answer:

```text
Given one circuit state, which valid candidate actions produce the best real
delta_test_coverage or hard-fault reduction when evaluated by Atalanta/TMAX?
Can the planner's candidate strategy and score function put those actions in
its top-K?
Are reward_pred / guarded_reward / hard_reduction_total_pred / hybrid_pred
monotonic with real delta_test_coverage?
Are the predicted gains calibrated enough to avoid selecting negative-TC
actions as top-1?
```

## Scope

In scope:

```text
new script: scripts/oracle_action_value_probe.py
tpi_jepa/plan.py candidate enumeration/scoring paths
tpi_jepa/evaluate_plan_tmax.py backend evaluation contract
scripts/build_tp_candidate_cache.py candidate cache inputs
autoresearch/plan-260629-0043/
```

Out of scope:

```text
new model training
loss-function changes
claiming final 8-circuit TC numbers from small oracle probes
Atalanta/TMAX backend semantic changes
push/publish/deploy
```

## Oracle Label Definition

For a fixed benchmark and fixed current state:

```text
state = already inserted actions S
candidate action a = (net, CP0/CP1/OP)
oracle_value(a) = backend-measured metric after evaluating S + [a]
```

Primary oracle target:

```text
oracle_delta_tc(a) = delta_test_coverage after inserting S + [a]
```

Secondary oracle targets:

```text
oracle_delta_fault_coverage(a)
oracle_delta_hard_fault_count(a)
oracle_delta_undetected_fault_count(a)
```

The oracle label is not `labels.csv`'s chosen next action. It is:

```text
top-M actions by oracle_delta_tc within a backend-valid candidate pool
```

## Metrics

For candidate quality:

```text
oracle_action_recall@K = |planner_topK ∩ oracle_topM| / M
oracle_node_recall@K   = |planner_topK_nodes ∩ oracle_topM_nodes| / |oracle_topM_nodes|
regret@K               = max(oracle_delta_tc) - max(delta_tc among planner_topK)
best_in_topK_delta_tc  = max(delta_tc among planner_topK)
negative_top1_rate     = fraction where planner top-1 has delta_tc < 0
```

For world-model reward prediction quality:

```text
spearman(pred_score, oracle_delta_tc)
kendall_tau(pred_score, oracle_delta_tc)
pearson(pred_score, oracle_delta_tc)
topK_real_gain_by_pred = mean real delta_tc of predicted top-K
topK_hit_rate          = predicted top-K overlap with oracle top-M
top1_real_delta_tc     = real delta_tc of highest predicted action
top1_regret            = oracle_best_delta_tc - top1_real_delta_tc
sign_accuracy          = accuracy(sign(pred_score), sign(oracle_delta_tc))
negative_top1_rate     = fraction where predicted top-1 has real delta_tc < 0
calibration_slope      = linear fit oracle_delta_tc ~ pred_score
calibration_intercept  = linear fit oracle_delta_tc ~ pred_score
```

Evaluate these separately for each prediction field:

```text
reward_pred
fc_pred
guarded_reward
return_pred
hard_reduction_total_pred
hybrid_pred
```

Promotion signal:

```text
regret@K decreases
best_in_topK_delta_tc increases
negative_top1_rate decreases
rank correlation with oracle_delta_tc increases
top1_real_delta_tc increases
oracle_action_recall@K improves only as a secondary signal
```

Do not promote a strategy only because it covers historical actions.

## Probe Design

Start with small probes because each oracle label requires real backend
simulation.

State selection:

```text
initial state S=[]
optional later states: first 1-2 actions from incumbent planner plan
```

Benchmarks:

```text
Phase 1 smoke: b15_C,i2c_aig
Phase 2 unstable/high-upside set: b15_C,max_aig,b17_C,i2c_aig
Phase 3 full 8: b15_C,b20_C,b21_C,b22_C,i2c_aig,max_aig,b17_C,mem_ctrl_aig
```

Candidate pool:

```text
backend-valid candidates from tpi_eval.candidates.generate_candidates
sample 32-128 candidate actions per state
include all three action types CP0/CP1/OP per selected net when affordable
```

Evaluation:

```text
for each candidate action:
  write a one-step plan CSV for S + [candidate]
  call tpi_jepa.evaluate_plan_tmax.evaluate_plan(...)
  record final row's delta_test_coverage and hard-fault metrics
```

Scoring comparison:

```text
Use tpi_jepa.plan.score_candidate_from_latent() to score the exact same
candidate set evaluated by the backend. Rank each state/action table by:
  reward_pred
  fc_pred
  guarded_reward
  return_pred
  hard_reduction_total_pred
  hybrid_pred

Then compare model-predicted ranks against oracle_delta_tc.
```

## Implementation Plan

Add `scripts/oracle_action_value_probe.py`.

Current validation status:

```text
scripts/oracle_action_value_probe.py does not exist yet.
autoresearch/tp-candidates-260626-2047/ already contains cache JSON files for
all 8 evaluation circuits.
```

Therefore the next executable step is implementation, not launching the smoke
command directly.

Required CLI:

```bash
python scripts/oracle_action_value_probe.py \
  --checkpoint PATH \
  --benchmarks b15_C,i2c_aig \
  --candidate-cache-dir autoresearch/tp-candidates-260626-2047 \
  --candidate-strategies cached_stride,cached_hard_cone,hard_fault_recall_union \
  --score-fields reward_pred,guarded_reward,hard_reduction_total_pred,hybrid_pred \
  --states initial \
  --max-nets 32 \
  --action-types CP0,CP1,OP \
  --patterns 50000 \
  --seed 2026 \
  --backend atalanta-bist \
  --plan-device cuda \
  --timeout-sec 14400 \
  --out-dir autoresearch/oracle-action-probe-260629-smoke
```

Outputs:

```text
oracle_actions.tsv
  one row per benchmark/state/candidate action with real backend metrics

rank_metrics.tsv
  one row per benchmark/state/strategy/score_field/K with recall/regret metrics

prediction_metrics.tsv
  one row per benchmark/state/score_field with correlation, top-K real gain,
  top-1 regret, sign accuracy, and calibration slope/intercept

state_summary.tsv
  oracle best action, model top-1 action per score field, real top-1 delta,
  regret, negative top-1 flag

handoff.json
  best/worst score fields, whether model reward is aligned with true TC, and
  recommended next planner route
```

## Verify

Static checks:

```bash
python -m py_compile tpi_jepa/*.py scripts/*.py
python -m tpi_jepa.plan --help
python -m tpi_jepa.evaluate_plan_tmax --help
python scripts/build_tp_candidate_cache.py --help
```

Existing-interface validation already passes for these commands. The oracle
smoke below is gated on implementing `scripts/oracle_action_value_probe.py`.

Candidate cache:

```bash
TPI_BENCH_ROOT=/data4/pengqingsong/DFT/Dataset/deeptpi_official_aig_bench_standard \
python scripts/build_tp_candidate_cache.py \
  --benchmarks b15_C,i2c_aig,max_aig,b17_C,b20_C,b21_C,b22_C,mem_ctrl_aig \
  --out-dir autoresearch/tp-candidates-260626-2047
```

Oracle smoke:

```bash
TPI_BENCH_ROOT=/data4/pengqingsong/DFT/Dataset/deeptpi_official_aig_bench_standard \
python scripts/oracle_action_value_probe.py \
  --checkpoint autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt \
  --benchmarks b15_C,i2c_aig \
  --candidate-cache-dir autoresearch/tp-candidates-260626-2047 \
  --candidate-strategies cached_stride,cached_hard_cone,hard_fault_recall_union \
  --score-fields reward_pred,guarded_reward,hard_reduction_total_pred,hybrid_pred \
  --states initial \
  --max-nets 16 \
  --action-types CP0,CP1,OP \
  --patterns 10000 \
  --seed 2026 \
  --backend atalanta-bist \
  --plan-device cuda \
  --timeout-sec 14400 \
  --out-dir autoresearch/oracle-action-probe-260629-smoke
```

Promotion gate:

```text
oracle_actions.tsv exists and has rows for every requested benchmark
rank_metrics.tsv includes recall@8/16/32 and regret@8/16/32
prediction_metrics.tsv includes rank correlation and top1_regret per score field
at least one score_field has positive Spearman/Kendall against oracle_delta_tc
at least one score_field/strategy has lower regret than netlist baseline
negative_top1_rate is not worse than netlist baseline
```

## Decision Rules

Keep a candidate/scoring strategy only if it improves oracle value:

```text
lower regret@K
higher best_in_topK_delta_tc
lower or equal negative_top1_rate
```

Reject this argument:

```text
strategy is good because it recalls labels.csv actions
```

Accept this argument:

```text
strategy is good because under the same backend-valid candidate pool it puts
real high-delta actions near the top and avoids negative top-1 choices
```

For world-model reward alignment, accept only if:

```text
the same candidate actions are evaluated both by the backend and by
score_candidate_from_latent()
prediction_metrics.tsv shows positive rank correlation against oracle_delta_tc
predicted top-1 has competitive real delta_tc and low regret
negative_top1_rate is controlled
```

## Next After Probe

If oracle ranking is good but full rollout TC is bad:

```text
problem is multi-step state update / compounding planner error
next plan: one-step greedy oracle-informed route or replanning guardrail
```

If oracle ranking is bad:

```text
problem is model score alignment
next plan: train action-value/ranking head using oracle probe labels, or
recalibrate reward_pred/hard_reduction_total_pred to delta_test_coverage
```

If candidate pool misses oracle actions:

```text
problem is candidate generation
next plan: backend-valid wider candidate sampling and per-circuit route table
```
