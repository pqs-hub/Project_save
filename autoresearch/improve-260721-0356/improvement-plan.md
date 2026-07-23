# Improvement Plan

## Must-have

1. Prefix-counterfactual ATPG dataset
   - Regenerate 32-step learner rollouts on the 24 admitted training subcircuits.
   - Label prefixes 1, 2, 4, 8, 12, 16, 24, and 31.
   - Evaluate nine candidates per prefix, balanced across CP0/CP1/OP, at 300,000 patterns and seed 2026.

2. Conservative typed residual
   - Train same-prefix pairwise/listwise losses, typed marginal/return, and SA0/SA1 heads.
   - Preserve the base score unless typed-head evidence supports a deviation; require stricter agreement for CP0.
   - Include an alpha-zero base-policy anchor in b15 selection.

3. Provenance and fairness guards
   - Validate the oracle manifest and row count before training so stale 100k/old-prefix labels cannot be consumed.
   - Keep target aliases forbidden and exact original-netlist allowlists mandatory.
   - Select only on b15 and freeze all settings for the five-circuit run.

## Nice-to-have

4. Independent typed-head ensemble and lower-confidence residual.
5. Active ATPG allocation to prefixes/candidates with the largest head disagreement.
6. DeepGate2/3-inspired functionality-aware and pooled global encoder, evaluated only after the label/policy path is stable.

## Big bets

7. Return-conditioned action-history transformer for the full insertion sequence.
8. Receding-horizon latent MPC with uncertainty-calibrated terminal values.

## Acceptance criteria

- All plan nodes are in the exact original gate-level allowlist.
- One checkpoint, score field, planner, and hyperparameter set is used for all five circuits.
- Budgets are b15 278, b20 616, b21 628, b22 915, b17 994.
- ATPG uses 300,000 patterns and seed 2026.
- Checkpoint/config selection reads b15 only.
- Primary success is 5/5 final TC strictly above 93.20/95.02/94.51/95.59/91.67.
- Before 5/5 is reached, a candidate that turns a Round13 DeepTPI win into a loss is not promoted as the engineering incumbent.

## Execution order

1. Add and unit-test the safe trust-residual score on the existing Round4 checkpoint.
2. Run a b15-only trust sweep, select once, and run the frozen five-circuit validation.
3. In parallel with analysis, collect the Round5 300k prefix counterfactual dataset.
4. Train the three Round5 variants, select on b15, and run the same frozen validation.
5. Only if the policy fixes saturate, prototype the encoder upgrade.

