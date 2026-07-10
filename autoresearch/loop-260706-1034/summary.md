# Autoresearch Loop Summary

Mode: classic

Goal: improve world-model rerank with NDCG/listwise, conservative/uncertainty, candidate-context, or DAG-aware methods without using the 8 target circuits for training, calibration, or model selection.

## Iteration 1

Implemented the first focused experiment unit: top-heavy listwise/NDCG oracle ranking.

Changes:

- Added `lambda_ndcg_rank` support in `tpi_jepa.train`.
- Added `_oracle_topk_ndcg_loss`, which concentrates supervision on oracle top-k actions.
- Added training metrics/logging for `oracle_ndcg_loss`.
- Added two safe configs:
  - `configs/planner_aligned_reward_rank_v4_ndcg_safe.json`
  - `configs/planner_aligned_q_rank_v3_ndcg_safe.json`
- Added the two NDCG models to training, non-target dev rerank, and final held-out scripts.
- Added `MODEL_FILTER` and `MAX_PARALLEL` to scripts so runs can target only NDCG variants without overloading GPUs.
- Added `tests/test_oracle_ranking_losses.py`.

## Guard Result

Passed:

- `python -m py_compile tpi_jepa/protocol.py tpi_jepa/train.py tpi_jepa/model.py tpi_jepa/plan.py autoresearch/improve-260706-0959/summarize_vs_old.py`
- `python -m pytest tests/test_eval_protocol_contract.py tests/test_oracle_ranking_losses.py`
- JSON validation for new configs
- bash syntax validation for train/dev/final scripts
- safe config leakage audit across 10 safe configs

## Not Run

The classic Verify command launches long GPU rerank jobs, so it was not started automatically. Use the commands below.

## Iteration 2

Implemented the next focused experiment unit: conservative q ranking.

Changes:

- Added `lambda_conservative_q` support in `tpi_jepa.train`.
- Added `_oracle_conservative_loss`, a CQL-style penalty that discourages high scores on non-oracle-top candidates.
- Added training metrics/logging for `oracle_conservative_loss`.
- Added safe config:
  - `configs/planner_aligned_q_rank_v4_conservative_safe.json`
- Added the conservative model to training, non-target dev rerank, and final held-out scripts.
- Extended `tests/test_oracle_ranking_losses.py`.

Guard after iteration 2:

- `python -m py_compile ...` passed.
- `python -m pytest tests/test_eval_protocol_contract.py tests/test_oracle_ranking_losses.py` passed: 5 tests.
- JSON and bash syntax checks passed.
- Safe config leakage audit passed across 11 configs.

Train only the new NDCG variants:

```bash
cd /data4/pengqingsong/DFT/TPI-my.3 && MODEL_FILTER=ndcg GPUS_CSV=0,1 MAX_PARALLEL=2 bash autoresearch/improve-260706-0959/run_train_safe_variants_parallel.sh
```

Train only the conservative q variant:

```bash
cd /data4/pengqingsong/DFT/TPI-my.3 && MODEL_FILTER=conservative GPUS_CSV=2 MAX_PARALLEL=1 bash autoresearch/improve-260706-0959/run_train_safe_variants_parallel.sh
```

Run non-target dev Verify only for the conservative q variant:

```bash
cd /data4/pengqingsong/DFT/TPI-my.3 && MODEL_FILTER=conservative GPUS_CSV=2 MAX_PARALLEL=1 bash autoresearch/improve-260706-0959/run_dev_non_target_rerank_parallel.sh
```

## Iteration 3

Implemented the uncertainty/lower-confidence rerank variant.

Changes:

- Added greedy ensemble scoring to `tpi_jepa.plan`.
- Added `--ensemble-checkpoints` and `--ensemble-lcb-alpha` to `tpi_jepa.plan`.
- Added ensemble argument pass-through to `scripts/run_gmean_sweep.py`.
- Added LCB score fields such as `q_pred_lcb` and `reward_pred_lcb`.
- Added two inference-only script variants:
  - `q_lcb_ensemble_safe`
  - `reward_lcb_ensemble_safe`

Guard after iteration 3:

- `python -m py_compile tpi_jepa/plan.py scripts/run_gmean_sweep.py tpi_jepa/train.py` passed.
- `python -m pytest tests/test_eval_protocol_contract.py tests/test_oracle_ranking_losses.py` passed: 5 tests.
- Script syntax checks passed.
- CPU smoke test passed with repeated q checkpoint and `score_field=q_pred_lcb`.

Run LCB variants after their member checkpoints exist:

```bash
cd /data4/pengqingsong/DFT/TPI-my.3 && MODEL_FILTER=lcb GPUS_CSV=3,4 MAX_PARALLEL=2 bash autoresearch/improve-260706-0959/run_dev_non_target_rerank_parallel.sh
```

## Dev Script Fix

The non-target dev script now defaults to the protocol-safe local benchmark `subckt_0001` and preflights BENCH lookup before launching GPU jobs. This fixed the stale `iscas89__s838` default/env failure without touching target circuits.

## Variant 4

Implemented the candidate-context rerank variant.

Changes:

- Added `lambda_context_rank` support in `tpi_jepa.train`.
- Added `_oracle_context_loss`, which aligns the normalized candidate-pool score shape to normalized oracle delta-TC.
- Added training metrics/logging for `oracle_context_loss`.
- Added planner score fields:
  - `q_pred_context`
  - `reward_pred_context`
  - `guarded_reward_context`
  - `hybrid_pred_context`
  - `bounded_residual_hybrid_pred_context`
- Added two trained safe configs:
  - `configs/planner_aligned_q_rank_v5_context_safe.json`
  - `configs/planner_aligned_reward_rank_v5_context_safe.json`
- Added two inference-only fast context variants using old checkpoints:
  - `q_context_v1`
  - `reward_context_v1`
- Added context entries to training, non-target dev rerank, and final held-out scripts.
- Added `tests/test_candidate_context_scores.py` and extended `tests/test_oracle_ranking_losses.py`.

Guard after variant 4:

- `python -m py_compile tpi_jepa/train.py tpi_jepa/plan.py scripts/run_gmean_sweep.py` passed.
- `python -m pytest tests/test_oracle_ranking_losses.py tests/test_candidate_context_scores.py tests/test_eval_protocol_contract.py` passed: 7 tests.
- Script syntax checks passed.
- JSON validation for new configs passed.
- Safe training audit passed with no eval8 target ids in labels or oracle files.
- CPU planner smoke passed with `score_field=q_pred_context`.

Train only the new candidate-context configs:

```bash
cd /data4/pengqingsong/DFT/TPI-my.3 && MODEL_FILTER=context GPUS_CSV=5,6 MAX_PARALLEL=2 bash autoresearch/improve-260706-0959/run_train_safe_variants_parallel.sh
```

Run non-target dev Verify for candidate-context variants:

```bash
cd /data4/pengqingsong/DFT/TPI-my.3 && DEV_BENCHMARK=subckt_0001 MODEL_FILTER=context GPUS_CSV=5,6 MAX_PARALLEL=2 FORCE=1 bash autoresearch/improve-260706-0959/run_dev_non_target_rerank_parallel.sh
```

## Oracle Filter Fix

The shared oracle TSV contains 36 `subckt_0001` rows. Because current safe configs exclude the protocol auxiliary dev benchmark from label training, oracle loading now skips forbidden benchmark rows and reports the count instead of crashing. A real config preflight confirmed `subckt_0001` is forbidden, 36 rows are skipped, 178 oracle groups remain, and no forbidden group is loaded.

Run non-target dev Verify only for NDCG variants:

```bash
cd /data4/pengqingsong/DFT/TPI-my.3 && MODEL_FILTER=ndcg GPUS_CSV=0,1 MAX_PARALLEL=2 bash autoresearch/improve-260706-0959/run_dev_non_target_rerank_parallel.sh
```

Run all safe variants if you want the complete proxy comparison:

```bash
cd /data4/pengqingsong/DFT/TPI-my.3 && GPUS_CSV=0,1,2,3,4,5,6,7 MAX_PARALLEL=8 bash autoresearch/improve-260706-0959/run_train_safe_variants_parallel.sh
cd /data4/pengqingsong/DFT/TPI-my.3 && GPUS_CSV=0,1,2,3,4,5,6,7 MAX_PARALLEL=8 bash autoresearch/improve-260706-0959/run_dev_non_target_rerank_parallel.sh
```

Final held-out eval8 remains acceptance-only:

```bash
cd /data4/pengqingsong/DFT/TPI-my.3 && CONFIRM_EVAL8_HELDOUT=1 PRIOR_MODE=old MODEL_FILTER=ndcg GPUS_CSV=0,1,2,3,4,5,6,7 MAX_PARALLEL=8 bash autoresearch/improve-260706-0959/run_eval8_final_heldout_parallel.sh
```
