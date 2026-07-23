# Unified b15-best reward configuration

All five circuits use one fixed planner configuration:

- checkpoint: `runs/rollout_loss_A_reward_only/epoch_009.pt`
- score: `reward_pred`
- planner: greedy
- candidate strategy: `hard_fault_cluster`
- candidate pool: 48
- maximum hard-fault seeds: 512
- exact original-netlist candidate allowlist
- Table-IV TP budgets, Atalanta-BIST 300K patterns, seed 2026

The b15 and b17 runs reuse matching existing exact-legal experiments. The
b20, b21, and b22 runs were generated together by `run_missing.sh`.
