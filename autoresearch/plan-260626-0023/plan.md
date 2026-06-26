# AutoResearch Plan: Clean Gate/Cone/Rank Ablation

generated_at: 2026-06-26 00:23 Asia/Shanghai

Subcommand:

```text
$autoresearch plan
```

## 1. Goal

Convert the current research decision into a validated experiment plan.

Primary question:

```text
Do gate-direction encoding, cone-aware summary, or pairwise hard ranking
improve hard-fault-aware TPI-JEPA beyond the current strong baseline?
```

## 2. Scope

### In Scope

Run a clean factorial ablation over the already implemented switches:

```text
encoder_type in {mean, gate_dir}
summary_mode in {global, cone}
lambda_hard_rank in {0.0, 0.03, 0.05}
```

Fixed baseline:

```text
ASL loss
residual_context hard head
top-k hard negative mining
hard-weighted sample strategy
lambda_hard=0.5
lambda_hard_count=0.10
lambda_hard_reduction=0.5
edge_weight_mode=fault_path
edge_keep_ratio=0.6
lambda_fc=0.0
```

### Out of Scope

Do not include:

```text
more lambda_hard_count 0.10 vs 0.12 sweeps
multi-view netlist work
large transformer / sparse attention implementation
action-level ranking implementation
masked testability pretraining
```

Reason:

```text
This plan is a bounded triage run.
It decides whether current implemented GNN/ranking switches deserve more work.
```

## 3. Metric

Primary metric:

```text
hard_macro_f1_tuned
```

Selection objective:

```text
--objective hard_f1
```

Guardrail metrics:

```text
predictive_score
hard_recall_at_top_10pct
hard_count_top10_overlap
hard_reduction_score
hard_sa0_pr_auc
hard_sa1_pr_auc
```

Current baseline evidence:

| Config | mean hard F1 | std | min | max | mean predictive |
|---|---:|---:|---:|---:|---:|
| core baseline, `hc=0.10` | `0.6468` | `0.1636` | `0.4245` | `0.7963` | `0.7110` |
| core baseline, `hc=0.12` | `0.6453` | `0.1554` | `0.4401` | `0.7949` | `0.7131` |

## 4. Verify Config

Expected command:

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

Expected outputs:

```text
autoresearch/predictive-tech-ablation-260626/results.tsv
autoresearch/predictive-tech-ablation-260626/summary.md
autoresearch/predictive-tech-ablation-260626/best.pt
autoresearch/predictive-tech-ablation-260626/logs/*.train.log
autoresearch/predictive-tech-ablation-260626/logs/*.eval.log
```

## 5. Acceptance Rules

### Component Keep Rule

Keep a component only if matched-seed comparison shows:

```text
hard_macro_f1_tuned improves by >= 0.03
or predictive_score improves by >= 0.03
```

and it does not cause:

```text
hard_recall_at_top_10pct drop > 0.05
or hard_reduction_score collapse
```

### Architecture Freeze Rule

If none of `gate_dir`, `cone`, or `lambda_hard_rank` clears the keep rule:

```text
freeze encoder_type=mean
freeze summary_mode=global
freeze lambda_hard_rank=0.0
move to calibration diagnostics and action-level ranking implementation
```

### Follow-up Validation Rule

If a component clears the keep rule:

```text
rerun the winning component on seeds 2026,2027,2028,2029,2030
before treating it as a new baseline
```

## 6. Interpretation Matrix

| Result Pattern | Interpretation | Next Step |
|---|---|---|
| `mean + global + rank` improves | hard ranking helps independently | Run rank local sweep, then 5-seed validation |
| `mean + cone + rank/no-rank` improves | cone summary is useful and cheap | Validate cone on 5 seeds |
| `gate_dir + global` improves | gate direction helps; cone interaction may hurt | Validate gate_dir without cone |
| only `gate_dir + cone + rank` improves | components interact | Validate exact bundle before code expansion |
| none improve | current baseline is sufficient | Freeze architecture; implement diagnostics/action ranking |

## 7. Post-Run Analysis Checklist

After the run completes:

1. Sort `results.tsv` by `hard_macro_f1_tuned`.
2. Compare each variant against `mean + global + rank=0.0`.
3. Check `predictive_score` guardrail.
4. Check whether improvements are due to top10 recall only or actual F1.
5. Update `docs/hard_f1_autoresearch_report.md`.
6. If there is a winner, generate a 5-seed validation command.
7. If there is no winner, start calibration diagnostics implementation.

## 8. Next Phase After This Plan

If architecture freezes:

```text
Phase 2: evaluator calibration / seed diagnostics
Phase 3: action-level ranking and planner metrics
```

If a component wins:

```text
Phase 2: 5-seed validation of the winning component
Phase 3: evaluator calibration / action-level ranking
```
