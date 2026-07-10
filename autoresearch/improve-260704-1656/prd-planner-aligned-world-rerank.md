# Planner-Aligned World Rerank

Auto-generated from research findings. DECISION NEEDED items and LOW-confidence sections require your judgment.

## Problem

The latest accuracy-improved world models improved static prediction metrics but regressed b21 `world_rerank` insertion from the incumbent 5.838% Delta TC to roughly 0.86-1.15%. The planner needs candidate ranking under long closed-loop rollout, not isolated prediction accuracy.

## User Stories

- As the test-point insertion researcher, I need a model selected by real insertion improvement so that validation metrics do not hide planner regressions.
- As the planner operator, I need score fields that remain finite for 628-step b21 planning so that rerank does not silently degrade into arbitrary ordering.
- As the experiment runner, I need one command that trains variants and gates them on b21 before spending time on the full 8-circuit suite.

## Requirements

### Must

- Train at least one reward-head candidate ranking variant using oracle action groups and `oracle_ranking_score_field=reward_pred`.
- Preserve the incumbent long-horizon rollout recipe with `rollout_max_horizon=5`.
- Gate every candidate on b21 `world_rerank` with the previous best parameters unchanged.
- Reject candidates with NaN planner scores.
- Produce a machine-readable summary TSV.

### Should

- Test a stronger oracle-rank/listwise variant.
- Test a dedicated `q_pred` action-ranking variant.
- Keep hard-head objectives weak unless they improve the b21 gate.

### Could

- Expand oracle action groups to intermediate planner states.
- Add automated 8-circuit evaluation for candidates that beat the b21 incumbent.

## Acceptance Criteria

- `configs/planner_aligned_reward_rank_v1.json`, `configs/planner_aligned_reward_rank_v2_strong.json`, and `configs/planner_aligned_q_rank_v1.json` parse as JSON.
- `autoresearch/improve-260704-1656/run_planner_aligned_variants.sh` passes `bash -n`.
- A completed run writes `autoresearch/improve-260704-1656/planner_aligned_variants/summary.tsv`.
- A candidate is considered worth full 8-circuit testing only if b21 Delta TC is greater than 5.838% and `score_nan_rows=0`.

## Suggested Technical Approach

Start from `configs/rollout_loss_A_reward_only.json`, because it produced the current b21 best. Add oracle ranking loss through the existing `tpi_jepa.train` path:

- `lambda_q_rank`
- `lambda_candidate`
- `lambda_q_value`
- `oracle_ranking_score_field`
- `oracle_pairwise_mode=best_vs_hard_topk`

The first two variants keep planner score as `reward_pred`. The third trains `q_pred` and evaluates with `--score-field q_pred`.

## Risks

- Oracle action groups are mostly initial-state groups, while b21 planning is long closed-loop.
- Strong ranking loss may harm reward calibration.
- b21-only gate can overfit; full 8-circuit confirmation remains required.

## Success Metrics

- Primary: b21 `world_rerank` Delta TC.
- Guardrail: `score_nan_rows=0`.
- Secondary: full 8-circuit macro Delta TC after b21 pass.

## DECISION NEEDED

- Whether to start the full training script now or wait for a smaller smoke run.
- Whether to collect intermediate-state oracle groups if none of these variants beat b21.

## Open Questions

- Should q-head deployment replace reward-head deployment if `q_pred` wins b21?
- Should the b21 gate threshold require beating 5.838% by a margin, such as +0.5pp, before full eval?
