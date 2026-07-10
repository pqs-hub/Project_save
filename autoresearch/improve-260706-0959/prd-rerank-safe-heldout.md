# PRD: Safe Held-Out Rerank Improvement

Auto-generated from research findings. DECISION NEEDED items and LOW-confidence sections require your judgment.

## Problem Statement

The current planner-aligned world models improve macro rerank performance but do not beat the old C_depth2 result on all 8 held-out circuits. The next iteration must improve model ranking ability without using any of the 8 target circuits as training, calibration, or selection data.

## Requirements

- Must keep Delta SCOAP disabled.
- Must train only from subckt labels and non-target oracle data.
- Must reject oracle files containing target canonical IDs or aliases.
- Must exclude protocol aliases and table circuit names from labels.
- Must keep final eval8 separate from training and non-target model selection.
- Must report per-circuit gap against old, not only macro average.

## Technical Approach

- Extend `tpi_jepa.train` with `oracle_forbidden_benchmarks`.
- Train eight safe variants over score fields `q_pred`, `reward_pred`, `guarded_reward`, and `hybrid_pred`.
- Use `exclude_eval_protocol=configs/eval_protocol_coverage_only.json` in every safe config.
- Run dev rerank on a non-target benchmark before final held-out eval8.
- Require `CONFIRM_EVAL8_HELDOUT=1` before any final 8-circuit evaluation script can run.

## Acceptance Criteria

- Training script audit passes.
- All trained safe variants produce `best_final_horizon.pt`.
- Non-target dev rerank completes and writes `summary.tsv`.
- Final held-out eval8, when explicitly run, writes `comparison_vs_old.tsv`.
- A candidate is accepted only if `beats_old=True` for all 8 target circuits.

## Risks

- Without target-circuit calibration, b15/i2c gains may not improve enough.
- Non-target dev rerank may not predict held-out eval8 behavior.
- `PRIOR_MODE=old` uses the previous candidate-prior setup and is needed for exact old comparison; `PRIOR_MODE=none` changes the benchmark setup.

## Open Questions

- DECISION NEEDED: If no safe variant beats old 8/8, should the next round expand non-target full-circuit oracle data from circuits outside the 8 targets?
