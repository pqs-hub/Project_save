# AutoResearch Improve PRD: Hard-Fault-Aware TPI-JEPA

generated_at: `2026-06-26 03:55 Asia/Shanghai`

## 1. Executive Decision

当前项目已经证明：

```text
hard fault node supervision 能显著增强 world model 的可测性表征。
```

下一阶段不要继续盲扫 encoder / cone / rank。已有 clean ablation 表明：

- `mean + global + no-rank` 应冻结为主线。
- `gate_dir`、`cone`、`hard_rank` 没达到 keep rule。
- post-hoc benchmark threshold policy 没有超过 class-tuned threshold。
- 固定 `0.5` 阈值明显失败，说明 hard head probability calibration 是真实瓶颈。

本轮 improve 的主结论：

```text
优先推进“训练期校准 + 边界稳定性 + action-ranking 数据重构”。
暂缓复杂 encoder、多视图和直接 action-ranking loss。
```

## 2. Current Evidence

### 2.1 已经成立的技术点

| 技术点 | 当前判断 | 证据 |
|---|---|---|
| ASL hard loss | 保留 | 当前高分主线均使用 ASL；适合稀疏 multi-label hard fault。 |
| residual_context hard head | 保留 | 当前高分主线均依赖 action relation context。 |
| top-k negative mining | 保留 | 比 random/mixed 更符合 hard F1 目标。 |
| hard_weighted sampling | 保留 | 对稀疏 hard label 有帮助。 |
| fault_path edge weighting | 保留 | 已在当前主线中稳定使用。 |
| class-tuned validation threshold | 保留 | 固定 `0.5` 阈值大幅降低 hard F1。 |

### 2.2 已经否定或暂缓的技术点

| 技术点 | 处理 | 原因 |
|---|---|---|
| `gate_dir` encoder | 暂缓 | clean ablation 中显著弱于 mean encoder。 |
| `cone` summary | 暂缓 | 小幅波动，未达到 `+0.03` keep rule。 |
| `lambda_hard_rank > 0` | 暂缓 | 对 F1/predictive 没有稳定正收益。 |
| benchmark-tuned / shrinkage threshold | 暂缓 | 未超过 class-tuned threshold。 |
| 直接 action-ranking loss | 暂缓 | 当前 comparable action groups 太少，ranking diagnostics 信号弱。 |
| 多视图 AIG/PM netlist | 排除本阶段 | 用户明确要求当前不要考虑多视图。 |

## 3. External Research Signals

这些文献给出的启发用于约束下一步，而不是直接照搬架构。

| 方向 | 相关工作 | 对本项目的启发 |
|---|---|---|
| TPI 是 action selection 问题 | DeepTPI: Test Point Insertion with Deep Reinforcement Learning, https://arxiv.org/abs/2206.06975 | 最终必须闭环到 action value / ranking；但需要先构造可靠 action group 标签。 |
| 电路功能表征 | DeepGate2, https://arxiv.org/abs/2305.16373 | 功能监督有效，但当前项目已有 hard fault 监督，应先稳定 hard head 边界。 |
| 长程/大图表示 | DeepGate3, https://arxiv.org/abs/2407.11095; DeepGate4, https://arxiv.org/abs/2502.01681 | 长程依赖是潜在瓶颈，但当前 clean ablation 说明先做训练目标更划算。 |
| 稀疏 multi-label loss | ASL, https://arxiv.org/abs/2009.14119 | 当前 ASL 路线合理；下一步应调 ASL 的 calibration 副作用，而不是换掉 ASL。 |
| GNN calibration | GETS, https://arxiv.org/abs/2410.09570 | GNN 概率校准可以独立成为优化目标；本项目应先做 lightweight calibration，不上复杂 ensemble。 |

## 4. ICP / User Pain Points

这里的 ICP 是后续真正使用这个系统的人：需要在 DFT/TPI 流程里减少 ATPG / fault simulation 调用，同时仍能选出有效 test points。

| Pain Point | 当前项目症状 | 影响 |
|---|---|---|
| hard node probability 不可信 | ECE 高，固定 `0.5` 阈值失败 | 无法稳定把 node-level 表征交给 planner。 |
| seed / split 波动大 | 5-seed F1 从约 `0.42` 到 `0.80` | 单次 autoresearch 容易误判技术路线。 |
| node F1 与 action value 错位 | action-ranking diagnostics 中 pairwise acc 低 | hard F1 提升不一定转化为 TPI 选择收益。 |
| action group 覆盖不足 | comparable groups 很少 | 不能可靠训练 listwise/pairwise action objective。 |
| 长程 hard propagation 未显式验证 | gate/cone 未提升，但 top10 recall 高 | 可能是数据/目标问题掩盖了架构问题。 |

## 5. Candidate Improvements: 15 Iterations

| Iter | Improvement | Expected Gain | Cost | Risk | Decision |
|---:|---|---|---|---|---|
| 1 | Add validation-wide calibration diagnostics across all benchmarks, no training | 提高结论可信度 | Low | Low | Do now |
| 2 | Add logit temperature scaling evaluator for SA0/SA1 | 降 ECE，验证概率校准空间 | Low | Low | Do now |
| 3 | Add train-time Brier auxiliary loss for hard logits | 改善概率边界 | Medium | Medium | Do next |
| 4 | Add soft-F1 / Dice auxiliary loss for hard labels | 直接优化 F1-like 边界 | Medium | Medium | Do next |
| 5 | Tune ASL `gamma_neg` and `clip` around current best | 保留 ASL 优势，降低过度压制 easy negatives | Low | Medium | Do next |
| 6 | Add per-class hard loss weights for SA0/SA1 imbalance | 缓解 SA1 calibration 异常 | Low | Medium | Do next |
| 7 | Build action candidate grouping dataset from same state_key | 为 action-ranking loss 提供有效样本 | Medium | Low | Do before ranking loss |
| 8 | Add offline action-ranking evaluator over candidate groups | 判断 action score 是否有真实排序信号 | Medium | Low | Do before ranking loss |
| 9 | Add pairwise action-ranking loss only on high-margin pairs | 提升 planner 对齐 | Medium | Medium | Later |
| 10 | Add planner-level top-k hit / NDCG metrics to autoresearch objective | 防止 F1-only 目标错位 | Medium | Low | Later |
| 11 | Add seed-aware objective aggregation in runner | 降低偶然高点 | Medium | Low | Later |
| 12 | Add hard-positive-rate stratified validation report | 找出低 F1 的数据桶 | Low | Low | Do now |
| 13 | Add label audit report for zero/near-zero hard-positive samples | 排查标签噪声和无效样本 | Low | Low | Do now |
| 14 | Revisit sparse global attention only after calibration stabilizes | 长程建模潜力 | High | Medium | Defer |
| 15 | Multi-view AIG/PM representation | 潜在泛化收益 | High | High | Excluded this phase |

## 6. Prioritized PRDs

### PRD-A: Full Calibration Diagnostic Pass

Objective:

```text
把当前有限 calibration 诊断扩展到更多 validation benchmarks，
判断 ECE / threshold / bucket failure 是否稳定存在。
```

Scope:

- Use `scripts/evaluate_hard_checkpoints.py`.
- No training.
- Evaluate best seed checkpoints across at least seeds `2026-2030`.
- Increase `--max-val-samples` and `--max-steps` if runtime allows.
- Output:
  - per-seed calibration summary
  - per-benchmark ECE/F1
  - hard-positive-rate bucket failure table

Metric:

- `ece_sa0`, `ece_sa1`
- `hard_macro_f1_tuned`
- `hard_macro_f1@0.5`
- per-benchmark worst-case F1

Promote rule:

```text
If high ECE and 0.5-threshold failure reproduce across seeds,
implement train-time calibration loss.
```

### PRD-B: Train-Time Calibration Loss

Objective:

```text
在保持 ASL 主损失的同时，引入轻量 calibration auxiliary loss，
减少 hard head 对 validation threshold 的过度依赖。
```

Implementation options:

1. Add `lambda_hard_brier`:
   - `Brier = mean((sigmoid(logits) - targets)^2)`
   - cheap, differentiable, no new labels.
2. Add `lambda_hard_soft_f1`:
   - directly targets F1-like precision/recall tradeoff.
   - must keep small weight to avoid instability.
3. Add `hard_logit_temperature` evaluator only:
   - post-hoc validation calibration before training change.

Suggested first grid:

```text
lambda_hard_brier in {0.02, 0.05, 0.10}
lambda_hard_soft_f1 in {0.00, 0.02}
hard_asl_gamma_neg in {2.0, 3.0, 4.0}
hard_asl_clip in {0.02, 0.05}
```

Fixed mainline:

```text
encoder_type=mean
summary_mode=global
lambda_hard_rank=0.0
hard_loss=asl
hard_head_type=residual_context
hard_negative_mining=topk
train_sample_strategy=hard_weighted
```

Primary metric:

```text
hard_macro_f1_tuned
```

Guardrails:

```text
predictive_score
hard_recall_at_top_10pct
ece_sa0/ece_sa1
hard_macro_f1@0.5
```

Promote rule:

```text
mean hard F1 improves by >= 0.03 over frozen mainline,
or ECE drops by >= 20% without hard F1 loss > 0.01.
```

### PRD-C: Action Candidate Group Reconstruction

Objective:

```text
先让 evaluator 拥有足够 action groups，再考虑 action-ranking loss。
```

Problem:

Current action-ranking diagnostics found too few comparable action groups and weak pairwise signal. This may be a data grouping issue, not necessarily a model issue.

Scope:

- Audit `state_key`, `sequence_id`, `pre_action_count`, `action_node_name`, `action_type`.
- Build `candidate_group_id = benchmark_id + state_key + pre_action_count`.
- Require group size >= 5.
- Filter pairs with target gain margin above a configurable threshold.
- Report:
  - group count
  - mean group size
  - target gain spread
  - pairwise label quality
  - oracle upper bound NDCG

Promote rule:

```text
Only implement action-ranking loss if there are enough groups:
>= 100 comparable groups,
mean target_gain_spread > 0.01,
oracle NDCG@10 materially above random.
```

### PRD-D: Seed-Aware Runner Upgrade

Objective:

```text
把 autoresearch 从 single-seed best-pick 升级为 seed-aware comparison。
```

Why:

Current 5-seed result shows variance is large enough to swamp small route gains.

Scope:

- Runner accepts `--seeds 2026,2027,2028`.
- Aggregates mean/std/min for objective.
- Writes `seed_summary.tsv`.
- Keeps `--stream-logs` progress output.

Promote rule:

```text
Any new route must beat frozen mainline by >= 0.03 mean hard F1,
or improve predictive_score by >= 0.03 with no hard F1 regression.
```

## 7. Recommended Next Autoresearch Plan

Goal:

```text
实现并验证 hard-head calibration 训练目标：
先补 evaluator 的 temperature/Brier 诊断，
再加入 lambda_hard_brier 和可选 soft-F1 loss，
围绕 frozen mean/global/no-rank 主线做小网格。
```

Scope:

```text
tpi_jepa/train.py
scripts/evaluate_hard_checkpoints.py
scripts/run_predictive_autoresearch.py
docs/hard_f1_autoresearch_report.md
```

Metric:

```text
Primary: hard_macro_f1_tuned
Guardrails: predictive_score, hard_recall_at_top_10pct, ece_sa0, ece_sa1, hard_macro_f1@0.5
```

Verify:

```bash
python -m py_compile tpi_jepa/train.py scripts/evaluate_hard_checkpoints.py scripts/run_predictive_autoresearch.py
python scripts/evaluate_hard_checkpoints.py --help
python scripts/run_predictive_autoresearch.py --help
```

Training command shape, with progress:

```bash
python scripts/run_predictive_autoresearch.py \
  --base-config configs/aig_lowtc_100k_hard_pretrain.json \
  --objective hard_f1 \
  --max-variants 12 \
  --seeds 2026 \
  --lambda-hards 0.5 \
  --lambda-hard-counts 0.1 \
  --lambda-hard-reductions 0.5 \
  --lambda-hard-ranks 0.0 \
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
  --stream-logs
```

Note:

```text
The command above is the current runner shape. It must be extended with
calibration-loss arguments before running PRD-B as a real training sweep.
```

## 8. Stop Rules

Stop calibration-loss route if:

- hard F1 improves less than `0.01` in single-seed smoke tests.
- ECE improves but predictive_score drops by more than `0.03`.
- tuned F1 improves only because thresholds become more extreme and `hard_macro_f1@0.5` remains unusable.

Stop action-ranking route if:

- comparable action groups remain fewer than `100`.
- target gain spread is near zero.
- oracle ranking upper bound is weak.

Stop architecture route if:

- calibration-loss route has not yet been evaluated.
- proposed encoder change cannot beat frozen mean/global/no-rank by `>= 0.03`.

## 9. Next Decision

Recommended next command:

```text
$autoresearch plan
```

Recommended plan goal:

```text
实现 hard-head calibration loss 与 evaluator temperature/Brier 诊断，
围绕 frozen mean/global/no-rank 主线设计下一轮可输出进度的训练命令。
```
