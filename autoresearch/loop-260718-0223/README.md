# Uniform exact-legal bottleneck search

This run explores one shared planner configuration across all five recovered
ITC99 circuits. Every plan is restricted to the exact original-netlist
allowlist and uses the Table-IV TP budgets, Atalanta-BIST 300K patterns, and
seed 2026.

Hard constraints:

- no per-circuit score-head or hyperparameter selection;
- no ATPG suffix enumeration, splicing, truncation, or action rewriting;
- all five circuits must use the same checkpoint, score, candidate strategy,
  candidate count, hard-seed count, and planner;
- the final success predicate requires every circuit, and the macro average,
  to exceed the DeepTPI final TC values.

The first parallel batch compares three unified variants of the held-out
`planner_aligned_q_rank_v5_context_safe` checkpoint. It launches 15 independent
plan/evaluation processes (three variants times five circuits) on the six idle
GPUs.

