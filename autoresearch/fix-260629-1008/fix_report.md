# AutoResearch Fix Report: Pairwise-Only Oracle Ranking

generated_at: `2026-06-29 10:08 Asia/Shanghai`

## Objective

Test the cleanest objective first:

```text
pairwise ranking loss only
```

This followed:

```text
autoresearch/plan-260629-1004/plan.md
```

## Method

No code changes were required.

The existing finetune script already supports pairwise-only training:

```text
--lambda-oracle-value 0.0
--lambda-oracle-rank 1.0
```

Training data:

```text
autoresearch/oracle-action-probe-260629-labeled-subckt-train/oracle_actions.tsv
```

Held-out subckt gate:

```text
autoresearch/oracle-action-probe-260629-labeled-subckt-val/oracle_actions.tsv
```

Transfer gate:

```text
autoresearch/oracle-action-probe-260629-smoke/oracle_actions.tsv
```

## Training Result

Output:

```text
autoresearch/oracle-action-value-finetune-260629-pairwise-only/
```

History:

| epoch | rank loss | pairs |
|---:|---:|---:|
| 1 | 0.587324 | 5032 |
| 2 | 0.587263 | 5032 |
| 3 | 0.586839 | 5032 |
| 4 | 0.586169 | 5032 |
| 5 | 0.585994 | 5032 |

Interpretation:

```text
The pairwise training objective decreased only slightly.
```

## Held-Out Subckt Gate

Output:

```text
autoresearch/oracle-action-value-gate-260629-pairwise-only-val/
```

Verdict:

```text
candidate: INCONCLUSIVE
```

Key comparison:

| checkpoint | score_field | Spearman | negative top1 | top1 regret |
|---|---|---:|---:|---:|
| incumbent | reward_pred | -0.137870 | 0.416667 | 0.023290 |
| candidate | reward_pred | -0.160885 | 0.541667 | 0.035315 |
| incumbent | hybrid_pred | 0.005693 | 0.458333 | 0.043452 |
| candidate | hybrid_pred | -0.001395 | 0.458333 | 0.036230 |

Interpretation:

```text
Pairwise-only training did not improve held-out subckt ranking.
reward_pred became less safe by negative_top1.
hybrid top1 regret improved somewhat, but ranking got worse and negative_top1
did not improve.
```

## Transfer Gate

Output:

```text
autoresearch/oracle-action-value-gate-260629-pairwise-only-transfer/
```

Verdict:

```text
candidate: INCONCLUSIVE
```

Key comparison:

| checkpoint | score_field | Spearman | negative top1 | top1 regret |
|---|---|---:|---:|---:|
| incumbent | reward_pred | 0.294742 | 0.500000 | 0.020223 |
| candidate | reward_pred | 0.280641 | 0.500000 | 0.026983 |
| incumbent | hybrid_pred | 0.327398 | 0.166667 | 0.012552 |
| candidate | hybrid_pred | 0.323155 | 0.166667 | 0.012552 |

Interpretation:

```text
Pairwise-only training did not transfer.
Both reward_pred and hybrid_pred Spearman decreased.
Top1 safety did not improve.
```

## Conclusion

The hypothesis:

```text
pairwise ranking loss alone is enough
```

is not supported by this experiment.

This is a useful negative result. It means the next fix should not simply tune
pairwise rank weight or run more epochs. The model needs either:

```text
1. explicit top1/sign safety objective
2. a cleaner score carrier such as action_value_head
3. different trainable scope, e.g. include action_encoder/dynamics or hard_reduction_head
```

Given the current evidence, the next most targeted step is:

```text
add explicit sign/top1 safety loss, but keep it isolated from pairwise-only
results.
```

## Next Recommendation

Run the Stage A safety-loss implementation from:

```text
autoresearch/plan-260629-0944/plan.md
```

But interpret success by:

```text
negative_top1_rate
top1_regret
Spearman as guardrail
```

not by Spearman alone.
