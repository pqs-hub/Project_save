# Accuracy Improvement Plan

Goal: improve trained prediction-head accuracy for the current mainline world
model, excluding Delta SCOAP.

## Baseline Accuracy Snapshot

Checkpoint: `runs/mainline_world_model_simplified/best.pt`

- `reward_pred` sign accuracy: `56.35%`
- `return_pred` sign accuracy: `62.48%`
- `hard_reduction_total` sign accuracy: `23.02%`
- `hard_reduction_sa0` sign accuracy: `14.75%`
- `hard_reduction_sa1` sign accuracy: `27.78%`
- `hard_node_any` binary accuracy: `21.12%`, balanced accuracy: `52.47%`, F1: `30.15%`
- `hard_node_sa0` binary accuracy: `32.47%`, balanced accuracy: `65.73%`, F1: `3.33%`
- `hard_node_sa1` binary accuracy: `23.49%`, balanced accuracy: `54.28%`, F1: `29.92%`
- hard-node top-100 recall: `69.84%`

## Must-Have Improvements

1. Make trained-head accuracy reproducible.

   Mechanism: add `scripts/evaluate_trained_head_accuracy.py`, which reports
   accuracy-style metrics only: sign accuracy, binary accuracy, balanced
   accuracy, precision/recall/F1, and hard-node top-k recall.

2. Remove Delta SCOAP from the improvement objective.

   Mechanism: set `lambda_delta_scoap=0.0` in the candidate config and omit
   Delta SCOAP unless explicitly requested by `--include-delta-scoap`.

3. Improve hard-node precision without losing top-k recall.

   Mechanism: increase node hard supervision from `lambda_hard=0.03` to `0.08`,
   enable `lambda_hard_brier=0.02`, enable `lambda_hard_soft_f1=0.06`, enable
   `lambda_hard_rank=0.03`, lower ASL negative focusing from `4.0` to `2.0`,
   and increase hard negative mining ratio to `10`.

4. Improve reward/return direction accuracy.

   Mechanism: raise `lambda_fc` from `0.25` to `0.45`, raise `lambda_return`
   from `0.03` to `0.10`, and use `hard_sample_weight=1.0` so action-utility
   heads pay more attention to hard-fault-heavy regions.

5. Preserve planner utility while improving auxiliary accuracy.

   Mechanism: keep architecture, candidate policy, relation mode, rollout
   horizon, and planner-facing score fields unchanged for the first run.

## Nice-To-Have Follow-Ups

- Add explicit sign-classification auxiliary losses for `reward_pred` and
  `hard_reduction_pred` if v1 does not improve sign accuracy.
- Tune hard-label decision thresholds on validation and report calibrated
  accuracy separately from raw `0.5` threshold accuracy.
- Add per-benchmark accuracy reporting to avoid one validation benchmark
  dominating the aggregate.

## Candidate Config

`configs/mainline_accuracy_improve_v1.json`

Key changes versus `configs/mainline_world_model_simplified.json`:

- `lambda_delta_scoap: 0.0`
- `lambda_hard: 0.08`
- `lambda_hard_rank: 0.03`
- `lambda_hard_brier: 0.02`
- `lambda_hard_soft_f1: 0.06`
- `lambda_hard_reduction: 1.0`
- `lambda_fc: 0.45`
- `lambda_return: 0.10`
- `hard_asl_gamma_neg: 2.0`
- `hard_asl_clip: 0.02`
- `hard_pos_weight_max: 10.0`
- `hard_negative_sample_ratio: 10`
- `hard_sample_weight: 1.0`

## Verification

Run:

```bash
CUDA_VISIBLE_DEVICES=0 bash autoresearch/improve-260704-1035/run_accuracy_improvement_v1.sh
```

Primary acceptance criteria:

- `reward_pred_sign` > `56.35%`
- `return_pred_sign` > `62.48%`
- `hard_reduction_total_sign` > `23.02%`
- `hard_node_any` F1 > `30.15%`
- hard-node top-100 recall >= `69.84%`

Secondary gate:

- Re-run the current 8-circuit planner eval before promotion; auxiliary
  accuracy improvements should not reduce `B_world_rerank` macro ΔTC or create
  negative circuits.
