# PRD: Action-Level Ranking Alignment

## Problem

The current model optimizes node-level `hard_macro_f1_tuned`, but test point insertion is an action selection problem. A model can classify hard nodes well and still rank candidate test point actions poorly.

## Goal

Train and evaluate the world model so that hard-fault-aware representations improve candidate action selection.

## Non-Goals

- Do not replace the current hard node head.
- Do not implement a full DRL planner in this phase.
- Do not add multi-view netlist support.

## Proposed Design

Add an action scorer:

```text
score(action | state) = MLP([graph_summary, action_node_latent, action_type_embedding, hard_summary])
```

Training signal:

```text
For actions from the same circuit/state:
score(a_i) > score(a_j) if hard_reduction_target_i > hard_reduction_target_j + margin
```

Loss options:

- pairwise margin ranking loss
- listwise softmax over action gains
- hybrid with existing hard node loss

## Metrics

Primary:

- `NDCG@10`
- `top1_best_action_hit`
- `MRR`
- `pairwise_action_acc`

Guardrails:

- `hard_macro_f1_tuned`
- `predictive_score`
- `hard_recall_at_top_10pct`

## Acceptance Criteria

On at least 3 seeds:

```text
NDCG@10 improves by >= 0.05
or top1_best_action_hit improves by >= 0.05
while mean hard F1 drops by <= 0.03
```

## Implementation Tasks

1. Group samples by benchmark/state before ranking.
2. Add action scorer head to `tpi_jepa/model.py`.
3. Add action ranking loss to `tpi_jepa/train.py`.
4. Add ranking metrics to `scripts/evaluate_hard_checkpoints.py`.
5. Add autoresearch switches:
   - `lambda_action_rank`
   - `action_rank_margin`
   - `action_rank_mode`

## Risks

- `hard_reduction_target` may be noisy.
- Candidate sets may not contain enough comparable pairs.
- Optimizing action ranking can reduce node hard F1 if loss weights are too high.

## Mitigation

- Use margin filtering.
- Only compare actions from the same circuit/state.
- Start with low `lambda_action_rank`, for example `0.02`.
