# AutoResearch Reason Report: Next-Step Decision for TPI-JEPA

generated_at: 2026-06-26 00:20 Asia/Shanghai

Subcommand:

```text
$autoresearch reason
Iterations: 8
```

## 0. Question

What should this project do next to most reliably improve hard-fault-aware TPI-JEPA?

The competing options:

1. **Action Alignment**: implement action-level ranking / planner-aligned objective.
2. **Calibration First**: improve evaluator diagnostics and threshold stability.
3. **GNN Ablation**: run the clean `gate_dir / cone / hard_rank` factorial ablation.
4. **More Hard-F1 Sweeps**: continue searching around `lambda_hard_count`, ASL, top-k mining.
5. **Large Representation Upgrade**: sparse attention, masked pretraining, or bigger circuit encoder.

## 1. Evidence Base

Current best-supported route:

```text
ASL + residual_context hard head + top-k hard negative mining + hard-weighted sampling
lambda_hard=0.5
lambda_hard_count=0.10 or 0.12
lambda_hard_reduction=0.5
edge_weight_mode=fault_path
edge_keep_ratio=0.6
```

Latest 5-seed stability result:

| `lambda_hard_count` | mean hard F1 | std | min | max | mean predictive | mean top10 recall |
|---:|---:|---:|---:|---:|---:|---:|
| `0.10` | `0.6468` | `0.1636` | `0.4245` | `0.7963` | `0.7110` | `0.8938` |
| `0.12` | `0.6453` | `0.1554` | `0.4401` | `0.7949` | `0.7131` | `0.8878` |

Paired seed differences for `0.10 - 0.12`:

```text
2026: -0.0156
2027: -0.0092
2028: +0.0353
2029: -0.0043
2030: +0.0013
```

Inference:

```text
lambda_hard_count=0.10 vs 0.12 is not the bottleneck.
Seed/split instability and node-to-action mismatch are larger risks.
```

## 2. Adversarial Debate

### Iteration 1: Should we continue hard-F1 hyperparameter sweeps?

**Pro-Hard-F1 Sweep**

The objective is hard F1. The current best single point is `0.7963`, so further local sweeps might push it higher.

**Against**

The paired 5-seed result shows `hc=0.10` and `0.12` differ by only about `0.0015` mean F1. The standard deviation is around `0.16`, two orders larger than the mean difference. Additional narrow sweeps will likely optimize seed noise.

**Judge**

Reject as top priority. Keep hard-F1 sweeps only for local validation after a structural change.

### Iteration 2: Should we implement action ranking immediately?

**Pro-Action Ranking**

TPI is action selection. DeepTPI-style framing is action value estimation. The current hard node representation is useful but may not translate to better candidate actions.

**Against**

Action ranking requires code changes in dataset grouping, model head, loss, and evaluator. If the baseline is not calibrated, ranking gains may be hard to interpret.

**Judge**

Action ranking is the highest-value research direction, but not the first execution step if we need immediate low-risk progress. Implement it after evaluator diagnostics or after a short ablation clarifies whether existing ranking mechanisms help.

### Iteration 3: Should calibration come before action ranking?

**Pro-Calibration**

Current hard F1 ranges from about `0.42` to `0.80`. Without seed/bucket diagnostics, any action-ranking result may be confounded by split effects. Threshold sensitivity already appears in the gap between top10 recall and F1.

**Against**

Calibration can improve metrics without improving actual planner decisions. It may produce a cleaner proxy while delaying the real action objective.

**Judge**

Calibration diagnostics should be implemented before claiming a new best, but they do not need to block running the already-supported clean ablation. They should block paper-level claims and action-ranking acceptance.

### Iteration 4: Should clean GNN ablation be the next run?

**Pro-GNN Ablation**

`gate_dir + cone + rank` was tested as a bundle and underperformed, but `rank=0.05` improved over `rank=0.0` inside that bundle. A 12-variant factorial run is cheap relative to implementing a new action-ranking pipeline and will answer whether to keep these existing switches.

**Against**

The current baseline `mean + global` is already strong. More GNN ablation may distract from action alignment.

**Judge**

Run clean ablation next because it is already implemented, bounded, and directly informs whether to keep or drop the new GNN switches. It must be treated as a triage run, not the main research arc.

### Iteration 5: Should large sparse attention / transformer work start now?

**Pro-Large Representation**

Hard faults depend on long-range propagation and reconvergence. DeepGate3/4 style long-range modules are conceptually relevant.

**Against**

Current evidence says the simple baseline is strong and the first gate/cone/rank bundle did not beat it. Large encoders add training cost, variance, and confounding before the cheap ablation is resolved.

**Judge**

Reject for now. Reconsider only if clean ablation shows `gate_dir` or `cone` helps but saturates.

### Iteration 6: Should masked testability pretraining start now?

**Pro-Masked Pretraining**

It may reduce seed variance and label dependence. This matches DeepCell/MCM-style representation learning.

**Against**

It is a larger implementation and may optimize proxy reconstruction rather than action utility. Current immediate bottleneck is not lack of a generic encoder but lack of action-level closure and stability diagnostics.

**Judge**

Defer. Keep as P2 after action ranking and calibration.

### Iteration 7: What is the right acceptance metric?

**Hard-F1 Judge**

Keep `hard_macro_f1_tuned` as the main metric because it directly tests whether the latent learned fault-testability.

**Planner Judge**

Hard F1 is a representation metric, not the final value metric. Future acceptance must include action ranking or planner budget metrics.

**Reproducibility Judge**

Any claimed improvement must be multi-seed or paired-seed validated.

**Final Metric Ruling**

Use a three-tier acceptance rule:

```text
Representation: hard_macro_f1_tuned, hard PR-AUC, hard_count_top10_overlap
Action: NDCG@10, top1 best-action hit, pairwise action accuracy
Planner: budgeted hard reduction / coverage gain at B in {5,10,20}
```

Until action metrics exist:

```text
Use hard F1 as primary and predictive_score as guardrail.
Require paired-seed or multi-seed validation.
```

### Iteration 8: Final ordering decision

**Candidate Order A**

Run clean ablation -> implement diagnostics -> implement action ranking.

**Candidate Order B**

Implement action ranking immediately -> then ablation.

**Candidate Order C**

Implement diagnostics immediately -> then action ranking -> then ablation.

**Judge Decision**

Choose A with a strict stop rule:

1. Run clean ablation because it is already implemented and bounded.
2. If no component improves by `>=0.03`, stop GNN work.
3. Implement calibration/seed diagnostics next.
4. Implement action-ranking metrics and loss after diagnostics.
5. Do not run more `hc=0.10/0.12` sweeps.

Reason:

```text
Clean ablation is the fastest way to retire or validate existing technical switches.
Calibration makes future claims trustworthy.
Action ranking is the main research direction after these blockers.
```

## 3. Converged Decision

### Immediate Next Run

Run clean gate/cone/rank ablation:

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

### Stop Rule

After the ablation:

```text
Keep a component only if matched-seed hard F1 or predictive_score improves by >=0.03.
Otherwise freeze architecture and move to calibration + action ranking.
```

### Next Coding Task

Implement evaluator diagnostics before major new claims:

```text
scripts/evaluate_hard_checkpoints.py
scripts/run_predictive_autoresearch.py
```

Required outputs:

```text
summary_by_seed.tsv
thresholds_by_class.tsv
per_benchmark_metrics.tsv
worst_seed section in summary.md
```

### Main Research Task After That

Implement action-level ranking:

```text
action scorer head
pairwise/listwise loss from hard_reduction_target
NDCG@10 / MRR / top1 hit evaluator
```

## 4. Claims Rejected

| Claim | Rejection Reason |
|---|---|
| Continue `lambda_hard_count=0.10/0.12` tuning | 5-seed paired difference is negligible relative to std. |
| Switch now to large transformer/sparse attention | Too much complexity before cheap ablation resolves existing switches. |
| Treat `0.7963` as final project result | It is a single-seed maximum; use it as a checkpoint, not a paper claim. |
| Optimize only node hard F1 | Final TPI value is action selection and budgeted fault reduction. |
| Start multi-view work now | User explicitly excluded multi-view in the previous plan, and current bottlenecks are elsewhere. |

## 5. Final Recommendation

The most defensible plan is:

```text
1. Run clean gate/cone/rank ablation with --stream-logs.
2. Use a strict >=0.03 matched-seed gain threshold.
3. Implement calibration/seed diagnostics.
4. Implement action ranking and planner-level metrics.
5. Validate any winner on 5 seeds before treating it as a new baseline.
```

This keeps momentum while preventing the project from drifting into ungrounded architecture complexity.
