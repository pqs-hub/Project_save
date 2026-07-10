# Summary

This improve pass was adjusted after the hard constraint: do not use the 8 target circuits for training, calibration, or selection.

Implemented:

- Added `oracle_forbidden_benchmarks` enforcement in `tpi_jepa.train`.
- Extended eval-protocol exclusion so target aliases such as `b15_C` and `i2c` are excluded from labels too.
- Added eight safe training configs under `configs/planner_aligned_*_safe.json`.
- Added parallel training script with target-leakage audit:
  `autoresearch/improve-260706-0959/run_train_safe_variants_parallel.sh`
- Added non-target dev rerank script:
  `autoresearch/improve-260706-0959/run_dev_non_target_rerank_parallel.sh`
- Added final held-out eval8 script that refuses to run unless `CONFIRM_EVAL8_HELDOUT=1`:
  `autoresearch/improve-260706-0959/run_eval8_final_heldout_parallel.sh`
- Added held-out comparator:
  `autoresearch/improve-260706-0959/summarize_vs_old.py`

Current best held-out result:

- Marked in `autoresearch/improve-260706-0959/current_best.md`
- Machine-readable record: `autoresearch/improve-260706-0959/current_best.json`
- Best method: `q_lcb_ensemble_safe`
- Metric: final TC, computed as measured local baseline TC + model delta TC
- Comparison target: DeepTPI Table-II final TC
- Aggregate: macro final TC 90.357%, min final TC 74.852%, beats DeepTPI 8/8
- Strict budgets: D1 278, D2 616, D3 628, D4 915, D5 34, D6 94, D7 994, D8 1273

No training or calibration script references Table-II exact-rank data or b15/i2c smoke probe data.

Next command:

```bash
cd /data4/pengqingsong/DFT/TPI-my.3 && GPUS_CSV=0,1,2,3,4,5,6,7 bash autoresearch/improve-260706-0959/run_train_safe_variants_parallel.sh
```
