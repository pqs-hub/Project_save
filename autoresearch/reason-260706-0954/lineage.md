# Autoresearch Reason Lineage

Task: reason about the strict old-budget 8-circuit evaluation results and decide how to interpret q_rank_v1, reward_rank_v1, and the old C_depth2 baseline.

Domain: research hypothesis / experimental decision

Mode: convergent

Judge criteria: testability, evidence support, explanatory power, practical actionability, risk control.

## Evidence Base

Strict protocol validation passed for:

- `autoresearch/improve-260704-1656/eval8_planner_aligned_parallel_oldbudgets`
- `configs/eval_protocol_coverage_only.json`

Core metrics:

| Method | Macro Delta TC | Min Delta TC | Positive Circuits | Beats Old |
|---|---:|---:|---:|---:|
| old C_depth2_rollout | +5.386% | +1.196% | 8/8 | - |
| q_rank_v1 | +7.331% | -0.135% | 7/8 | 6/8 |
| reward_rank_v1 | +6.432% | +0.552% | 8/8 | 6/8 |

Per-circuit deltas versus old C_depth2:

| Circuit | Old | q_rank_v1 | reward_rank_v1 |
|---|---:|---:|---:|
| max | +11.048% | +18.681% | +14.892% |
| i2c | +1.196% | -0.135% | +0.552% |
| b15 | +10.949% | +9.180% | +8.819% |
| b17 | +1.357% | +3.944% | +4.247% |
| b20 | +5.351% | +6.334% | +6.305% |
| b21 | +6.013% | +6.706% | +6.975% |
| b22 | +3.542% | +5.002% | +4.968% |
| mem_ctrl | +3.630% | +8.939% | +4.698% |

## Round 1

### Candidate-A

Use `q_rank_v1` as the new best model. It has the highest strict-budget macro Delta TC: +7.331%, which is +1.946 percentage points over old C_depth2. It beats the old baseline on 6/8 circuits and gives large gains on max and mem_ctrl.

### Critic

1. This overstates the result because q_rank_v1 has a negative i2c result, while the old baseline and reward_rank_v1 are both positive on all eight circuits.
2. It conflates "best macro score" with "best deployable strategy"; research reporting can accept one negative circuit, but a robust DFT flow may not.
3. It does not separate model improvement from planner differences. The old baseline used beam depth2 reward rollout, while current q/reward results use greedy depth1.
4. It ignores that b15 is worse than old for both new models, which matters because b15 was one of the strongest old gains.

A superior candidate would separate macro winner, robust winner, and next validation experiment.

### Candidate-B

Use `reward_rank_v1` as the current best deployable rerank model because it is positive on all eight circuits and still improves macro over old C_depth2 by +1.046 percentage points. Treat q_rank_v1 as a research candidate with higher upside but not yet stable because of i2c.

### Candidate-AB

The correct interpretation is two-track:

- If optimizing macro Delta TC, q_rank_v1 is the strict-budget winner.
- If requiring no circuit regression, reward_rank_v1 is the safer winner.
- Neither should be claimed as a clean replacement for the old system until model/planner effects are isolated under the same fixed protocol.

### Blind Judges

Judge 1: Candidate-AB wins. It best matches the evidence and avoids overclaiming q_rank_v1.

Judge 2: Candidate-AB wins. It preserves the macro result while acknowledging the i2c failure and the safety criterion.

Judge 3: Candidate-AB wins. It gives a practical decision rule instead of forcing one ambiguous "best".

Verdict: Candidate-AB. Convergence count: 1.

## Round 2

### Candidate-A

Incumbent: two-track interpretation. q_rank_v1 is macro-best; reward_rank_v1 is robust all-positive; final claims require fixed-protocol validation.

### Critic

1. The decision rule is still incomplete: it does not say what to do next if the user wants one publishable "best".
2. It risks using per-circuit hindsight. Choosing q on some circuits and reward on others would be oracle selection unless a pre-declared gating rule exists.
3. It does not address whether old C_depth2 might regain advantage if q_rank_v1 is also tested with beam depth2.
4. It lacks a regression gate to prevent future budget/protocol drift.

A superior candidate would define a publishable criterion and a concrete next experiment matrix.

### Candidate-B

Use this rule:

1. For a paper-style aggregate metric, report q_rank_v1 as "macro-best, not all-positive".
2. For a default recommended model, report reward_rank_v1 as "stable all-positive".
3. For a single final replacement, run a fixed-protocol factorial: q/reward model heads crossed with greedy depth1 and beam depth2, all with the Table-II budgets.
4. Any final result must pass `scripts/validate_eval_protocol.py`.

### Candidate-AB

Refined decision:

- Current publishable claim: "Planner-aligned reranking improves strict Table-II macro Delta TC from +5.386% to +7.331%, but the macro-best q_rank_v1 has one circuit regression."
- Current safe recommendation: `reward_rank_v1`, because it is all-positive and still beats old macro.
- Next experiment: isolate model vs planner by running q_rank_v1 and reward_rank_v1 with the old C_depth2 planner shape: beam, cumulative, width 2, depth 2, k=96, heuristic_recall_pool, strict Table-II budgets.
- Permanent guard: every reported run must pass the eval protocol validator.

### Blind Judges

Judge 1: Candidate-AB wins. It is most falsifiable and gives an exact next matrix.

Judge 2: Candidate-AB wins. It avoids oracle per-circuit selection and uses pre-declared criteria.

Judge 3: Candidate-AB wins. It distinguishes report, recommendation, and next experiment.

Verdict: Candidate-AB. Convergence count: 2.

## Round 3

### Candidate-A

Incumbent: report q as macro-best with one regression; recommend reward as robust all-positive; run fixed-protocol beam depth2 isolation next.

### Critic

1. It still could be read as "reward is deployable" despite being tested only on one seed and one pattern budget.
2. It does not explicitly say whether old C_depth2 remains relevant.
3. It should explain why q_rank_v1 can have higher macro but lower minimum, because this is the central tradeoff.
4. It should state what would change the conclusion.

A superior candidate would include conditions for promotion and rejection.

### Candidate-B

Final rule:

- q_rank_v1 is promoted only for macro-optimization claims.
- reward_rank_v1 is promoted only as a safer rerank candidate, not as fully production-proven.
- old C_depth2 remains a strong baseline because it has 8/8 positive gains and better i2c/b15 than both new models.
- The conclusion changes if beam depth2 q/reward removes i2c and b15 regressions, or if multi-seed variance shows the current deltas are unstable.

### Candidate-AB

Final synthesis:

The evidence supports a conditional upgrade, not an unconditional replacement. q_rank_v1 is the best strict-budget macro result, but its i2c regression means it should not be called the safest default. reward_rank_v1 is the safe all-positive rerank candidate, but it has lower macro. Old C_depth2 remains the reference baseline until a fixed-protocol planner-matched test proves the new model improves macro without unacceptable per-circuit regression.

Promotion criteria for the next run:

1. Must use `configs/eval_protocol_coverage_only.json`.
2. Must pass `scripts/validate_eval_protocol.py`.
3. Macro Delta TC should exceed old C_depth2 by at least +1 point.
4. Minimum Delta TC should remain positive, or any negative circuit must be explicitly reported as a tradeoff.
5. Planner comparison must be apples-to-apples: greedy depth1 and beam depth2 should be separated.

### Blind Judges

Judge 1: Candidate-AB wins. It best balances evidence, caution, and next steps.

Judge 2: Candidate-AB wins. It states exact promotion criteria and preserves old C_depth2 as the benchmark.

Judge 3: Candidate-AB wins. It explains the macro/min tradeoff and makes the next experiment falsifiable.

Verdict: Candidate-AB. Convergence count: 3. Status: CONVERGED.
