# Improvement Plan

## Acceptance Criteria

Primary claim requires final held-out eval8:

- strict protocol: `configs/eval_protocol_coverage_only.json`;
- restored Table-II budgets: max 94, i2c 34, b15 278, b17 994, b20 616, b21 628, b22 915, mem_ctrl 1273;
- 8/8 per-circuit `delta_test_coverage` greater than old C_depth2;
- no target circuit oracle/training/calibration data used before final held-out evaluation.

## Ranked Variants

Must-have:

1. `reward_rank_v3_safe`
   - Best first attempt for safety. It extends the all-positive `reward_rank_v1` but strengthens ranking and hard-reduction preservation.

2. `q_rank_v2_safe`
   - Best first attempt for macro. It keeps the q-score path but reduces the aggressive q/listwise weights that likely caused i2c instability.

3. `guarded_reward_rank_v1_safe`
   - Conservative fallback. It uses `guarded_reward` to penalize actions that look good only in one horizon head.

Nice-to-have:

4. `hybrid_rank_v1_safe`
   - Tests whether hard-reduction information can carry transfer without target calibration.

5. `reward_rank_v3_seed2_safe`
   - Seed/LR replicate for the safer reward route.

6. `q_rank_v2_seed2_safe`
   - Seed/LR replicate for the macro route.

7. `guarded_reward_rank_v1_seed2_safe`
   - Second guarded-reward run with lower LR and stronger hard-negative top-k.

8. `hybrid_rank_v1_seed2_safe`
   - Second hybrid run with lower LR and stronger hard-reduction preservation.

## Run Order

1. Train all safe variants:

```bash
cd /data4/pengqingsong/DFT/TPI-my.3 && GPUS_CSV=0,1,2,3,4,5,6,7 bash autoresearch/improve-260706-0959/run_train_safe_variants_parallel.sh
```

2. Run non-target rerank sanity check:

```bash
cd /data4/pengqingsong/DFT/TPI-my.3 && GPUS_CSV=0,1,2,3,4,5,6,7 bash autoresearch/improve-260706-0959/run_dev_non_target_rerank_parallel.sh
```

3. Only after selecting candidates without target data, run final held-out eval8:

```bash
cd /data4/pengqingsong/DFT/TPI-my.3 && CONFIRM_EVAL8_HELDOUT=1 PRIOR_MODE=old GPUS_CSV=0,1,2,3,4,5,6,7 bash autoresearch/improve-260706-0959/run_eval8_final_heldout_parallel.sh
```

`PRIOR_MODE=old` keeps the previous old candidate-prior setup for apples-to-apples comparison. Use `PRIOR_MODE=none` only for an additional no-prior sensitivity run; it is not the same old baseline.
