# Autoresearch Reason Summary

Status: CONVERGED

Task: interpret the strict old-budget 8-circuit world-model rerank results and decide what can be claimed.

Final winner: conditional two-track interpretation.

## Conclusion

The new results support a conditional upgrade, not an unconditional replacement of the old best scheme.

- `q_rank_v1` is the strict Table-II budget macro winner: +7.331% macro Delta TC, +1.946 percentage points over old C_depth2.
- `q_rank_v1` is not the safest default because it has one negative circuit, i2c at -0.135%.
- `reward_rank_v1` is the safer all-positive rerank candidate: +6.432% macro Delta TC, +1.046 points over old C_depth2, 8/8 positive.
- Old `C_depth2_rollout` remains the reference baseline: +5.386% macro Delta TC, 8/8 positive, better than both new models on i2c and b15.

## Recommended Claim

Use this wording:

"Under the strict restored DeepTPI Table-II #TP budget protocol, planner-aligned reranking improves macro Delta TC over the old C_depth2 baseline. The q-ranking head gives the best macro result (+7.331% vs +5.386%), while the reward-ranking head gives the safer all-positive result (+6.432%, 8/8 positive)."

Do not claim:

- "q_rank_v1 kills old best on all eight circuits."
- "q_rank_v1 is universally better."
- "The old best is obsolete."

## Next Experiment

Run a fixed-protocol planner-matched isolation:

- q_rank_v1 with greedy depth1 and beam depth2.
- reward_rank_v1 with greedy depth1 and beam depth2.
- Same `configs/eval_protocol_coverage_only.json`.
- Same k=96, heuristic_recall_pool, Atalanta BIST, 300k patterns, seed 2026.
- Validate every output with `scripts/validate_eval_protocol.py`.

Promotion rule:

- Macro Delta TC exceeds old C_depth2 by at least +1 percentage point.
- Minimum Delta TC remains positive, or the negative circuit is explicitly reported.
- No budget/protocol drift.

## Judge Agreement

Rounds: 3

Judge agreement: 9/9 votes for the synthesized conditional interpretation.

Convergence trajectory:

1. Round 1: two-track interpretation selected.
2. Round 2: fixed-protocol planner-matched next experiment added.
3. Round 3: promotion criteria and claim language finalized.
