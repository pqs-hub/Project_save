# Planner-Aligned World Model Improvement Plan

## Must-Have

1. Reward-head candidate ranking training

   Keep the incumbent `rollout_loss_A_reward_only` recipe as the base and add oracle pairwise/listwise ranking directly on `reward_pred`. This is the closest match to the current `world_rerank` deployment path.

2. Long-horizon rollout preservation

   Keep `rollout_max_horizon=5`; do not collapse back to horizon 2 for planner models. The b21 gate is 628 closed-loop decisions, and the prior regression shows short-horizon prediction accuracy does not transfer.

3. Real insertion gate

   Every candidate must pass the same b21 gate: `greedy + world_rerank + heuristic_recall_pool + max_candidates=96 + budget=628 + 300k patterns`. Static accuracy is only diagnostic.

4. NaN and score-stability guard

   Any candidate with non-finite planner score, adjusted score, or sequence score during b21 planning is rejected before 8-circuit evaluation.

## Nice-To-Have

1. Dedicated `q_pred` ranking head

   Train `q_pred` on oracle action groups and evaluate planner with `--score-field q_pred`. This decouples action ranking from reward/fc calibration.

2. Stronger oracle-rank variant

   Increase pairwise/listwise loss weight while reducing unrelated hard-head pressure. This tests whether direct candidate ranking can beat incumbent reward-only behavior.

## Moonshot

1. In-loop oracle data from non-initial states

   Current oracle groups mostly score initial-state actions. The final target is closed-loop states after many inserted points, so collecting oracle groups from intermediate planner states should reduce distribution drift.

## Selected PRDs

- `prd-planner-aligned-world-rerank.md`

## Output Artifacts

- Configs:
  - `configs/planner_aligned_reward_rank_v1.json`
  - `configs/planner_aligned_reward_rank_v2_strong.json`
  - `configs/planner_aligned_q_rank_v1.json`
- Scripts:
  - `autoresearch/improve-260704-1656/run_planner_aligned_variants.sh`
  - `autoresearch/improve-260704-1656/run_b21_gate_only.sh`
