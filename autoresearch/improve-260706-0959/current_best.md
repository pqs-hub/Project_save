# Current Best Result

Recorded: 2026-07-07

Current best method: `q_lcb_ensemble_safe`

Selection basis:

- Metric: macro final test coverage, higher is better.
- Final TC formula: measured local baseline TC + model delta TC.
- Comparison target: DeepTPI Table-II final TC.
- Held-out rule: the 8 target circuits are not used for training, calibration, or model selection.
- Budget rule: strict restored DeepTPI Table-II #TP budgets.

Protocol:

- Eval protocol: `configs/eval_protocol_coverage_only.json`
- BENCH root: `autoresearch/deeptpi_table2_restored_bench`
- Backend: `atalanta-bist`
- Patterns: 300000
- Seed: 2026
- Eval mode: final-only

Method:

- Planner: greedy cumulative rerank
- Score field: `q_pred_lcb`
- Ensemble checkpoints:
  - `runs/planner_aligned_q_rank_v1/best_final_horizon.pt`
  - `runs/planner_aligned_q_rank_v2_safe/best_final_horizon.pt`
  - `runs/planner_aligned_q_rank_v2_seed2_safe/best_final_horizon.pt`
- LCB alpha: 0.75
- Candidate strategy: `heuristic_recall_pool`
- Max candidates per step: 96
- Beam width: 1
- Lookahead depth: 1

Aggregate result:

| Method | Macro Final TC | Min Final TC | Beats DeepTPI | Worst Gap |
|---|---:|---:|---:|---:|
| `q_lcb_ensemble_safe` | 90.357 | 74.852 | 8/8 | +0.046pp |

Per-circuit result:

| ID | Circuit | Budget | Local Baseline TC | q_lcb Final TC | DeepTPI TC | Gap |
|---|---:|---:|---:|---:|---:|---:|
| D1 | b15_C | 278 | 81.764 | 90.656 | 90.610 | +0.046 |
| D2 | b20_C | 616 | 90.757 | 97.087 | 90.600 | +6.487 |
| D3 | b21_C | 628 | 89.682 | 96.379 | 89.730 | +6.649 |
| D4 | b22_C | 915 | 92.130 | 97.187 | 91.700 | +5.487 |
| D5 | i2c | 34 | 94.900 | 96.907 | 86.110 | +10.797 |
| D6 | max | 94 | 59.796 | 79.916 | 52.650 | +27.266 |
| D7 | b17_C | 994 | 85.623 | 89.869 | 86.840 | +3.029 |
| D8 | mem_ctrl | 1273 | 65.823 | 74.852 | 69.950 | +4.902 |

Result files:

- `autoresearch/improve-260706-0959/eval8_final_heldout_oldbudget/summary.tsv`
- `autoresearch/improve-260706-0959/eval8_final_heldout_oldbudget/comparison_final_tc_vs_deeptpi.tsv`

Reproduce:

```bash
cd /data4/pengqingsong/DFT/TPI-my.3 && CONFIRM_EVAL8_HELDOUT=1 PRIOR_MODE=old MODEL_FILTER=q_lcb GPUS_CSV=0,1,2,3 MAX_PARALLEL=2 FORCE=1 bash autoresearch/improve-260706-0959/run_eval8_final_heldout_parallel.sh
```
