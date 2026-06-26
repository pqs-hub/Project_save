# AutoResearch Improve Report: Hard-Fault-Aware TPI-JEPA

generated_at: 2026-06-26 00:05 Asia/Shanghai

## 1. Current Diagnosis

The current project thesis is strong and now has empirical support:

```text
Use hard fault node prediction as an auxiliary/self-supervised signal
to force the JEPA world model to learn fault-testability representations.
```

The latest 5-seed A/B run shows the core route is real:

| Core Route | mean hard F1 | std | min | max | mean predictive |
|---|---:|---:|---:|---:|---:|
| `lambda_hard_count=0.10` | `0.6468` | `0.1636` | `0.4245` | `0.7963` | `0.7110` |
| `lambda_hard_count=0.12` | `0.6453` | `0.1554` | `0.4401` | `0.7949` | `0.7131` |

Main conclusion:

```text
0.10 vs 0.12 is no longer the key question.
Seed/split variance and node-to-action objective mismatch are now the main bottlenecks.
```

## 2. External Research Signals

Relevant external directions, used only to constrain the next project moves:

| Source | Relevant Signal | Implication for This Project |
|---|---|---|
| DeepTPI, GNN + DQN action-value TPI | TPI is fundamentally action selection, not pure node classification. | Add action-level ranking and planner-level metrics. |
| DeepGate2 | Function-aware gate embeddings improve circuit representation. | Keep gate/function-aware ideas, but test them with clean ablations. |
| DeepGate3 | Long-range and subcircuit relations need stronger than local GNNs. | Consider sparse cone/global modules only after cheap ablations. |
| DeepCell / MCM | Masked circuit modeling helps representation learning when labels are expensive. | Add masked testability pretraining later, not before action alignment. |
| ASL | Imbalanced positive/negative multi-label tasks benefit from asymmetric loss. | Keep ASL as default; tune gamma/clip only locally. |

Primary references:

- DeepTPI: https://arxiv.org/abs/2206.06975
- DeepGate2: https://arxiv.org/abs/2305.16373
- DeepGate3: https://arxiv.org/abs/2407.11095
- DeepCell: https://arxiv.org/abs/2502.06816
- NetTAG: https://arxiv.org/abs/2504.09260
- ASL: https://arxiv.org/abs/2009.14119

## 3. ICP / User Challenges

For this project, the effective user is an EDA/DFT researcher who needs a model that can:

1. Improve hard fault coverage under limited test point budget.
2. Reduce expensive ATPG / fault simulation calls.
3. Generalize across circuits, not just win one seed.
4. Produce reproducible evidence strong enough for a paper or internal technical review.
5. Give planner-useful action scores, not just high node-level F1.

Current gaps against that profile:

| Gap | Evidence | Impact |
|---|---|---|
| Seed/split instability | Same core route ranges from about `0.42` to `0.80` hard F1. | Single-run claims are weak. |
| Node/action mismatch | Current objective is node hard F1; final task is test point action selection. | A better node classifier may not improve planner decisions. |
| New GNN route is under-explained | `gate_dir + cone + rank` underperformed as a bundle. | Cannot tell which idea is useful. |
| Calibration weakness | High top10 recall can coexist with poor F1. | False positives hurt action scoring and thresholded hard labels. |
| No planner-level acceptance gate | Metrics stop at node/reduction proxies. | Hard to prove real TPI value. |

## 4. Fifteen Improvement Candidates

| # | Candidate | Effort | Expected Gain | Risk | Priority |
|---:|---|---:|---:|---:|---:|
| 1 | Action-level pairwise/listwise ranking head | M | H | M | P0 |
| 2 | Planner-level evaluation at budgets `{5,10,20}` | M | H | M | P0 |
| 3 | Threshold calibration with held-out-only tuning and smoothing | S | H | L | P0 |
| 4 | Gate/cone/rank factorial ablation | S | M | L | P0 |
| 5 | Seed/split diagnostics by hard-positive rate, circuit size, depth | S | M | L | P0 |
| 6 | Hard-count ordinal/bin calibration head | M | M | M | P1 |
| 7 | Pairwise hard-rank local tuning `{0.01,0.02,0.03,0.05}` | S | M | M | P1 |
| 8 | Hard-negative curriculum from random/mixed to topk | M | M | M | P1 |
| 9 | Class-wise SA0/SA1 threshold smoothing | S | M | L | P1 |
| 10 | Two-stage action scorer: hard-region prior then rerank | M | H | M | P1 |
| 11 | Sparse cone/global attention for long-range dependencies | L | H | H | P2 |
| 12 | Masked testability pretraining | L | M | M | P2 |
| 13 | Ensemble/MC-dropout uncertainty for selective simulation | M | M | M | P2 |
| 14 | ASL gamma/clip local sweep | S | S | L | P2 |
| 15 | Label noise audit for low-seed failures | M | M | M | P2 |

## 5. Recommended Productized Improvements

### PRD-A: Action Ranking Alignment

Goal:

```text
Convert hard-fault-aware node representation into planner-useful action ranking.
```

Why now:

- DeepTPI frames TPI as action-value estimation.
- This project already has useful node hard representations.
- Current risk is optimizing a proxy that may not improve action selection.

Scope:

- Add action scorer using graph/action summary and current hard heads.
- Train pairwise or listwise action ranking from `hard_reduction_target`.
- Add action metrics: `NDCG@10`, `MRR`, `top1_best_action_hit`, `pairwise_action_acc`.
- Keep hard F1 as guardrail, not sole objective.

Acceptance:

```text
On 3 seeds, action ranking improves NDCG@10 or top1 hit
without reducing mean hard F1 by more than 0.03.
```

### PRD-B: Stability and Calibration

Goal:

```text
Reduce seed/split sensitivity and make hard F1 less threshold-fragile.
```

Scope:

- Add evaluator output by benchmark/circuit bucket.
- Save threshold diagnostics per seed and per SA0/SA1 class.
- Add optional threshold smoothing:
  - global threshold
  - class-wise threshold
  - per-benchmark threshold with shrinkage to global
- Add summary stats: mean/std/min/max across seeds.

Acceptance:

```text
Reduce 5-seed hard F1 std from ~0.16 to <=0.10,
or lift worst-seed hard F1 from ~0.42 to >=0.55.
```

### PRD-C: Clean GNN Technology Ablation

Goal:

```text
Identify whether gate_dir, cone summary, or hard_rank is useful independently.
```

Scope:

Run a factorial ablation:

```text
encoder_type in {mean, gate_dir}
summary_mode in {global, cone}
lambda_hard_rank in {0.0, 0.03, 0.05}
```

Keep everything else fixed at the 5-seed baseline center.

Acceptance:

```text
At least one component gives >=0.03 hard F1 gain or >=0.03 predictive gain
on a matched seed, without increasing low-seed failure rate.
```

## 6. Immediate Next Experiments

### Experiment 1: Clean Tech Ablation

Purpose: answer whether `gate_dir`, `cone`, or `rank` is useful.

```bash
python scripts/run_predictive_autoresearch.py \
  --base-config configs/aig_lowtc_100k_hard_pretrain.json \
  --objective hard_f1 \
  --max-variants 12 \
  --out-dir autoresearch/predictive-tech-ablation-260626 \
  --lambda-hards 0.5 \
  --lambda-hard-counts 0.1 \
  --lambda-hard-reductions 0.5 \
  --lambda-hard-ranks 0.0,0.03,0.05 \
  --encoder-types mean,gate_dir \
  --summary-modes global,cone \
  --hard-losses asl \
  --hard-head-types residual_context \
  --hard-pos-weight-maxes 20 \
  --hard-negative-sample-ratios 5 \
  --hard-negative-minings topk \
  --train-sample-strategies hard_weighted \
  --feature-modes testability \
  --edge-weight-modes fault_path \
  --edge-keep-ratios 0.6 \
  --lambda-fcs 0.0 \
  --center-lambda-hard 0.5 \
  --center-lambda-hard-count 0.1 \
  --center-lambda-hard-reduction 0.5 \
  --center-lambda-hard-rank 0.03 \
  --center-edge-keep-ratio 0.6 \
  --stream-logs
```

### Experiment 2: Ranking Weight Local Sweep on Baseline Encoder

Purpose: isolate whether `lambda_hard_rank` helps when not bundled with `gate_dir`.

```bash
python scripts/run_predictive_autoresearch.py \
  --base-config configs/aig_lowtc_100k_hard_pretrain.json \
  --objective predictive \
  --max-variants 8 \
  --out-dir autoresearch/predictive-rank-local-260626 \
  --seeds 2027,2028 \
  --lambda-hards 0.5 \
  --lambda-hard-counts 0.1 \
  --lambda-hard-reductions 0.5 \
  --lambda-hard-ranks 0.0,0.01,0.03,0.05 \
  --encoder-types mean \
  --summary-modes global \
  --hard-losses asl \
  --hard-head-types residual_context \
  --hard-pos-weight-maxes 20 \
  --hard-negative-sample-ratios 5 \
  --hard-negative-minings topk \
  --train-sample-strategies hard_weighted \
  --feature-modes testability \
  --edge-weight-modes fault_path \
  --edge-keep-ratios 0.6 \
  --lambda-fcs 0.0 \
  --center-lambda-hard 0.5 \
  --center-lambda-hard-count 0.1 \
  --center-lambda-hard-reduction 0.5 \
  --center-lambda-hard-rank 0.03 \
  --center-edge-keep-ratio 0.6 \
  --stream-logs
```

### Experiment 3: Stability Confirmation After Choosing a Component

Purpose: any winning component must survive multi-seed validation.

Template:

```bash
python scripts/run_predictive_autoresearch.py \
  --base-config configs/aig_lowtc_100k_hard_pretrain.json \
  --objective hard_f1 \
  --max-variants 5 \
  --out-dir autoresearch/stability-next-component-260626 \
  --seeds 2026,2027,2028,2029,2030 \
  --lambda-hards 0.5 \
  --lambda-hard-counts 0.1 \
  --lambda-hard-reductions 0.5 \
  --lambda-hard-ranks <WINNING_RANK_WEIGHT> \
  --encoder-types <WINNING_ENCODER> \
  --summary-modes <WINNING_SUMMARY> \
  --hard-losses asl \
  --hard-head-types residual_context \
  --hard-pos-weight-maxes 20 \
  --hard-negative-sample-ratios 5 \
  --hard-negative-minings topk \
  --train-sample-strategies hard_weighted \
  --feature-modes testability \
  --edge-weight-modes fault_path \
  --edge-keep-ratios 0.6 \
  --lambda-fcs 0.0 \
  --center-lambda-hard 0.5 \
  --center-lambda-hard-count 0.1 \
  --center-lambda-hard-reduction 0.5 \
  --center-lambda-hard-rank <WINNING_RANK_WEIGHT> \
  --center-edge-keep-ratio 0.6 \
  --stream-logs
```

## 7. Implementation Backlog

### Backlog 1: Evaluator Calibration Diagnostics

Files:

```text
scripts/evaluate_hard_checkpoints.py
scripts/run_predictive_autoresearch.py
```

Add:

- per-class threshold table
- per-benchmark or per-circuit aggregated F1
- seed summary writer when `--seeds` is used
- worst-seed tracking in `summary.md`

### Backlog 2: Action Ranking Metrics

Files:

```text
tpi_jepa/dataset.py
tpi_jepa/train.py
tpi_jepa/model.py
scripts/evaluate_hard_checkpoints.py
```

Add:

- candidate action groups by circuit/state
- pairwise action comparisons from `hard_reduction_target`
- action scorer head
- `NDCG@10`, `MRR`, `top1_best_action_hit`

### Backlog 3: Hard Count Calibration

Files:

```text
tpi_jepa/model.py
tpi_jepa/train.py
scripts/evaluate_hard_checkpoints.py
```

Add:

- hard count ordinal bins
- optional rank + ordinal combined loss
- top-k hard-count overlap by benchmark

## 8. Decision Rule for the Next Two Days

Run order:

1. Run Experiment 1.
2. If no component improves hard F1/predictive by at least `0.03`, do not expand GNN work.
3. Run Experiment 2 only if ranking shows promise or if Experiment 1 is inconclusive.
4. Implement evaluator calibration diagnostics before claiming a new best.
5. Start action-ranking PRD implementation after diagnostics, because that is the highest-value project direction.

## 9. Handoff Summary

Recommended next command:

```text
Experiment 1: Clean Tech Ablation
```

Recommended default baseline:

```text
ASL + residual_context + topk + hard_weighted
lambda_hard=0.5
lambda_hard_count=0.10
lambda_hard_reduction=0.5
lambda_hard_rank=0.0
encoder_type=mean
summary_mode=global
edge_weight_mode=fault_path
edge_keep_ratio=0.6
```

Do not prioritize:

```text
More 0.10 vs 0.12 count-loss sweeps.
Large multi-view implementation.
Large transformer implementation before clean ablation.
```
