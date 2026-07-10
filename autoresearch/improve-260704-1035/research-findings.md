# Research Findings

## Finding 1: Hard-node threshold behavior is the largest immediate weakness

The current checkpoint has near-perfect hard-node recall at threshold `0.5`,
but very low precision. That means the head is useful as a broad recall prior,
but weak as a calibrated binary classifier.

Proposed mechanism: keep high recall pressure, but add calibration and ranking
pressure with `lambda_hard_brier`, `lambda_hard_soft_f1`, `lambda_hard_rank`,
and stronger top-k hard negative mining.

Confidence: HIGH.

## Finding 2: Delta SCOAP should be removed from this objective

Delta SCOAP targets are mostly zero in the measured validation split, while the
model predicts many nonzero signs. It is not a useful accuracy objective for
the current goal.

Proposed mechanism: set `lambda_delta_scoap=0.0` and exclude it from standard
accuracy reports unless explicitly requested.

Confidence: HIGH.

## Finding 3: Reward and return need more direct weight

The planner works well with `reward_pred` rerank, but validation sign accuracy
is only moderate. The current config gives utility heads less weight than
hard-reduction.

Proposed mechanism: raise `lambda_fc`, raise `lambda_return`, and use
hard-region sample weighting.

Confidence: MEDIUM.

## Finding 4: Architecture changes are not the first move

Existing experiment history shows that changing/unfreezing dynamics in small
oracle-finetune settings can harm transfer. The safest first move is a target
weighting and calibration pass with the same architecture.

Confidence: MEDIUM.
