# PRD: Stability and Calibration Diagnostics

## Problem

The best core route is effective but unstable:

```text
5-seed hard F1 range: about 0.42 to 0.80
std: about 0.16
```

This weakens claims and makes downstream planner behavior unreliable.

## Goal

Make hard F1 less sensitive to seed/split and make thresholded hard predictions more reliable.

## Proposed Design

Extend evaluator outputs:

- per-seed summary
- per-class SA0/SA1 threshold
- per-benchmark F1
- benchmark size/depth/hard-positive-rate buckets
- worst-seed and low-seed diagnostics

Calibration options:

- global threshold
- class-wise threshold
- per-benchmark threshold with shrinkage to global
- temperature scaling for hard logits

## Metrics

Primary:

- 5-seed mean hard F1
- 5-seed std hard F1
- worst-seed hard F1

Secondary:

- SA0 PR-AUC
- SA1 PR-AUC
- hard recall top10
- hard count top10 overlap

## Acceptance Criteria

One of:

```text
hard F1 std <= 0.10
worst-seed hard F1 >= 0.55
mean hard F1 improves by >= 0.03 without increasing std
```

## Implementation Tasks

1. Extend `scripts/evaluate_hard_checkpoints.py` to export per-benchmark rows.
2. Add calibration mode argument.
3. Add `summary_by_seed.tsv` in `scripts/run_predictive_autoresearch.py`.
4. Add a plot script for seed/error bars.
5. Update report generation to include worst-seed diagnostics.

## Risks

- Per-benchmark calibration can overfit.
- More metrics can obscure the main conclusion.

## Mitigation

- Tune thresholds only on validation.
- Report global and calibrated results side by side.
- Keep hard F1 and predictive score as headline metrics.
