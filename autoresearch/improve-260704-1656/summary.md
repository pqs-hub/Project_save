# Summary

Mode: `$autoresearch improve`

Goal: recover and improve `world_rerank` insertion performance after static-accuracy training regressed b21.

## Key Result

The improvement path should move away from hard-head accuracy optimization and back toward planner-aligned candidate ranking. The current best b21 reference remains `runs/rollout_loss_A_reward_only/epoch_009.pt` with 5.838% Delta TC.

## Generated Variants

- `planner_aligned_reward_rank_v1`: incumbent recipe plus moderate oracle ranking on `reward_pred`.
- `planner_aligned_reward_rank_v2_strong`: stronger oracle rank/listwise pressure on `reward_pred`.
- `planner_aligned_q_rank_v1`: dedicated `q_pred` action-ranking head, evaluated with `score_field=q_pred`.

## Generated Scripts

- `run_planner_aligned_variants.sh`: trains all three variants and gates them on b21.
- `run_b21_gate_only.sh`: runs a single checkpoint through the same b21 gate.

## Recommended Gate

Promote a model to 8-circuit evaluation only if:

- b21 Delta TC > 5.838%
- planner score has no NaN rows

## Status

Complete. Long training not started automatically.
