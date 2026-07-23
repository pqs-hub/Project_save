# PRD: Prefix-Counterfactual Safe Residual

## Problem

Round4's typed heads were trained mainly on chosen actions. At inference the residual changes early decisions, enters new prefixes, and can amplify unsupported CP0 scores. Macro TC improved, but b22 changed from a DeepTPI win to a loss. This is not acceptable as a general framework improvement.

## Product behavior

For every planner prefix, score the exact same legal candidate pool with the stable base score and three independently supervised typed signals: marginal TC, long-return, and SA0/SA1 reduction. The base action remains preferred unless a challenger has sufficient relative typed evidence. CP0 challengers require all three heads; CP1/OP challengers require at least two. All thresholds are global and selected on b15 only.

## Functional requirements

- Expose `q_typed_trust_context` without changing existing score fields.
- Record support count, required count, eligibility, and applied correction in every plan row.
- Alpha zero must reproduce `q_pred_context` ordering exactly.
- Unsupported challengers must not displace the base action solely because of one typed-head spike.
- Validate all trust environment variables and fail on invalid values.
- Validate Round5 oracle provenance: 300k patterns, expected prefixes, nine actions per prefix, current learner plans, complete successful rows.
- Include `lambda_oracle_sa_value` when deciding whether oracle auxiliary training is enabled.

## Non-functional requirements

- Deterministic under the existing CUDA, hash, score-quantization, and single planning-thread settings.
- No validation-circuit information in training, calibration, or selection.
- Existing score fields and checkpoints remain backward compatible.

## Verification

- Unit tests for base preservation, supported override, CP0 unanimity, invalid settings, and oracle manifest drift.
- b15-only selection artifacts must state checkpoint, score field, alpha, decay, support thresholds, and tie-break.
- Frozen five-circuit comparison must report legality ratio, final TC, DeepTPI gap, Round13 gap, and lexicographic goal metric.

