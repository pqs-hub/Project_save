# Round5 Prefix-Oracle Audit

Source: `autoresearch/loop-260720-0945/model_training_round5/onpolicy_prefix_oracle/oracle_actions.tsv`

Protocol:

- 24 non-target training subcircuits;
- 192 learner-visited states at prefix steps 1, 2, 4, 8, 12, 16, 24, and 31;
- 9 candidates per state, balanced CP0/CP1/OP;
- 1,728 successful candidate evaluations;
- Atalanta-BIST, 300,000 patterns, seed 2026;
- each label is candidate TC minus the TC of the exact same prefix.

## Action-type evidence

| action | rows | mean marginal TC | median | positive | negative |
|---|---:|---:|---:|---:|---:|
| CP0 | 572 | -0.0743 pp | 0.0000 pp | 284 | 220 |
| CP1 | 573 | +0.0585 pp | +0.0040 pp | 304 | 200 |
| OP | 583 | +0.4224 pp | +0.0220 pp | 486 | 22 |

## Teacher-policy mismatch

- The Round4 teacher selected the real best candidate in only 7/192 states (3.65%).
- Mean one-step regret was 0.8208 pp; median 0.2090 pp; maximum 7.4200 pp.
- Teacher choices: CP0 93, CP1 46, OP 53.
- Real best candidates: CP0 25, CP1 47, OP 120.
- At least one positive candidate existed in 167/192 states.

This directly confirms the suspected bottleneck: the candidate recall pool usually contains a much better action, but the learned scorer over-selects CP0 and does not rank same-prefix alternatives correctly. The Round5 training objective therefore increases oracle exposure, retains balanced action types, trains SA0/SA1 heads, and includes a top-heavy listwise variant.

