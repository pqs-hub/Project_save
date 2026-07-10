# PRD: Trained-Head Accuracy Improvement V1

Auto-generated from research findings. DECISION NEEDED items and LOW-confidence
sections require your judgment.

## Problem Statement

The current mainline world model is strong enough for planner reranking, but
its trained auxiliary predictions are not accurate enough to trust directly:
reward/return sign accuracy is moderate, hard-reduction sign accuracy is weak,
and hard-node labels have high recall but low precision.

## Requirements

- Must exclude Delta SCOAP from the improvement objective.
- Must keep the existing model architecture for the first run.
- Must report only accuracy-style metrics, not loss or MAE.
- Must preserve planner compatibility with `heuristic_recall_pool` and
  `reward_pred` reranking.

## Acceptance Criteria

- `reward_pred_sign` improves over `56.35%`.
- `return_pred_sign` improves over `62.48%`.
- `hard_reduction_total_sign` improves over `23.02%`.
- `hard_node_any` F1 improves over `30.15%`.
- `hard_node_top100_recall` stays at or above `69.84%`.
- Follow-up planner eval does not introduce negative circuits on eval8.

## Technical Approach

Start with `configs/mainline_accuracy_improve_v1.json`. It changes loss weights
and hard-label calibration knobs while keeping architecture and planner-facing
settings stable.

Evaluate with `scripts/evaluate_trained_head_accuracy.py` and save metrics to
`trained_head_accuracy.tsv`.

## Risks

- More hard-label precision pressure can reduce hard-node top-k recall.
- Better auxiliary accuracy may not improve planner ΔTC.
- Validation split has only part of the benchmark distribution because samples
  are capped at 4096.

## Open Questions

- DECISION NEEDED: Should promotion optimize auxiliary accuracy, planner ΔTC, or
  a weighted combination?
- DECISION NEEDED: Should threshold-calibrated hard-node accuracy count as model
  improvement, or only raw threshold `0.5` accuracy?
