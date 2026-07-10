# Research Findings

Goal: improve the current world model so rerank can beat the previous old result on the strict 8-circuit protocol, without using the 8 target circuits for training, calibration, or model selection.

## Hard Constraint

The 8 target circuits must remain held out. This run therefore does not use:

- `autoresearch/exact-rank-table2-hybrid-k96-realfault-300k-260703-142737/*` for training or calibration.
- `autoresearch/oracle-action-probe-260629-smoke/oracle_actions.tsv`, because it contains target aliases `b15_C` and `i2c_aig`.
- Any Table-II target circuit oracle actions as a gate for model selection.

Final eval8 is still needed to prove the claim, but the script requires `CONFIRM_EVAL8_HELDOUT=1` and is separated from training/dev selection.

## Current Strict Baseline

Source: `autoresearch/improve-260704-1656/eval8_planner_aligned_parallel_oldbudgets/summary.tsv`

- `q_rank_v1`: macro `7.331%`, min `-0.135%`, beats old on `6/8`.
- `reward_rank_v1`: macro `6.432%`, min `0.552%`, beats old on `6/8`.
- Both fail against old on `i2c` and `b15`.
- `q_rank_v1` is macro-best but unsafe on i2c.
- `reward_rank_v1` is all-positive but has a larger b15 gap.

Old C_depth2 strict target:

- max `11.048%`
- i2c `1.196%`
- b15 `10.949%`
- b17 `1.357%`
- b20 `5.351%`
- b21 `6.013%`
- b22 `3.542%`
- mem_ctrl `3.630%`

## Mechanism

The next safe attempt should not chase static prediction accuracy alone. The prior accuracy-improve variants improved some head metrics but hurt rerank. The new variants keep the planner-aligned oracle ranking structure and vary the score field and regularization:

- `q_rank_v2_safe`: lower q-rank/candidate pressure than v1, more hard-reduction preservation.
- `q_rank_v2_seed2_safe`: same family, lower LR/seed change to reduce instability.
- `reward_rank_v3_safe`: stronger reward ranking while preserving hard/fault-reduction heads.
- `reward_rank_v3_seed2_safe`: safer reward variant with lower LR and stronger hard-reduction.
- `guarded_reward_rank_v1_safe` / `guarded_reward_rank_v1_seed2_safe`: train `min(reward_pred, return_pred)` to avoid high reward with poor return.
- `hybrid_rank_v1_safe` / `hybrid_rank_v1_seed2_safe`: train the planner's `hybrid_pred` score using reward, return, and hard reduction.

## Leakage Guard

Added training support for `oracle_forbidden_benchmarks`. New configs include target canonical IDs and aliases. If an oracle TSV contains any forbidden benchmark id, training raises an error before optimization starts.

Also extended `exclude_eval_protocol` handling so protocol aliases and table row circuit names are excluded from labels, not only canonical benchmark IDs.

The training script also audits:

- config has `exclude_eval_protocol=configs/eval_protocol_coverage_only.json`;
- labels contain no target IDs;
- oracle TSV contains no target IDs;
- blocked oracle sources such as `exact-rank-table2` and `oracle-action-probe` are not referenced.
