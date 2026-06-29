# AutoResearch Fix Report: Head-Only Hybrid Ranking

generated_at: `2026-06-29 11:00 Asia/Shanghai`

## Objective

Run:

```text
train reward_head and return_head only, but rank hybrid_pred
```

This tests whether reward/return can act as a residual correction to the stable
hard-reduction signal, without changing:

```text
action_encoder
dynamics
hard_reduction_head
online_encoder
target_encoder
```

## Configuration

Command mode:

```text
--train-scope heads
--train-heads reward,return
--ranking-score-field hybrid_pred
--lambda-oracle-value 0.0
--lambda-oracle-rank 1.0
```

Train oracle:

```text
autoresearch/oracle-action-probe-260629-labeled-subckt-train/oracle_actions.tsv
```

Candidate:

```text
autoresearch/oracle-action-value-finetune-260629-heads-hybrid/candidate.pt
```

## Verification

Syntax:

```text
python -m py_compile scripts/finetune_oracle_action_values.py
PASS
```

Smoke:

```text
autoresearch/oracle-action-value-finetune-260629-heads-hybrid-smoke/
```

Smoke confirmed trainable prefixes:

```text
reward_head
return_head
```

## Training

Output:

```text
autoresearch/oracle-action-value-finetune-260629-heads-hybrid/
```

History:

| epoch | train rank loss | train value loss | pairs |
|---:|---:|---:|---:|
| 1 | 0.804766 | 0.479482 | 5032 |
| 2 | 0.778338 | 0.495561 | 5032 |
| 3 | 0.752365 | 0.525522 | 5032 |
| 4 | 0.724025 | 0.589888 | 5032 |
| 5 | 0.745346 | 0.641121 | 5032 |

Interpretation:

```text
The rank loss improves through epoch 4, then regresses at epoch 5.
Head-only hybrid ranking is weaker than joint-hybrid training.
```

## Held-Out Labeled-Subckt Gate

Output:

```text
autoresearch/oracle-action-value-gate-260629-heads-hybrid-val/
```

Hybrid comparison:

| checkpoint | hybrid Spearman | negative top1 | top1 regret |
|---|---:|---:|---:|
| incumbent | 0.005693 | 0.458333 | 0.043452 |
| joint_hybrid | 0.025509 | 0.458333 | 0.031754 |
| freezehard | -0.010977 | 0.416667 | 0.028037 |
| heads_hybrid | 0.001006 | 0.500000 | 0.043200 |

Hard-reduction comparison:

| checkpoint | hard Spearman | negative top1 | top1 regret |
|---|---:|---:|---:|
| incumbent | 0.020105 | 0.458333 | 0.043452 |
| heads_hybrid | 0.020105 | 0.458333 | 0.043452 |

Interpretation:

```text
heads_hybrid preserves hard_reduction exactly, but does not improve held-out
hybrid ranking. It slightly worsens negative top1.
```

## Full-Circuit Transfer Gate

Output:

```text
autoresearch/oracle-action-value-gate-260629-heads-hybrid-transfer/
```

Hybrid comparison:

| checkpoint | hybrid Spearman | negative top1 | top1 regret |
|---|---:|---:|---:|
| incumbent | 0.327398 | 0.166667 | 0.012552 |
| joint_hybrid | 0.256683 | 0.166667 | 0.012558 |
| freezehard | 0.182085 | 0.166667 | 0.017645 |
| heads_hybrid | 0.334122 | 0.166667 | 0.012552 |

Hard-reduction comparison:

| checkpoint | hard Spearman | negative top1 | top1 regret |
|---|---:|---:|---:|
| incumbent | 0.324443 | 0.166667 | 0.012552 |
| heads_hybrid | 0.324443 | 0.166667 | 0.012552 |

Interpretation:

```text
heads_hybrid does not damage transfer. It slightly improves transfer hybrid
Spearman over incumbent because reward/return changed without moving the hard
score.
```

## Verdict

Candidate:

```text
DO NOT PROMOTE
```

Reason:

```text
Transfer is safe, but held-out labeled-subckt ranking does not improve and
negative top1 worsens.
```

## Diagnosis

The experiment answered the safety question:

```text
Keeping action_encoder/dynamics/hard_reduction fixed prevents full-circuit
transfer collapse.
```

But it also shows:

```text
reward_head/return_head alone are too weak to improve held-out subckt ranking
from the current 72 oracle train groups.
```

So the current bottleneck is likely not just trainable scope. It is:

```text
insufficient oracle action-value coverage
```

or:

```text
hybrid score composition is dominated by hard_reduction and reward/return cannot
move ranking enough without destabilizing safety.
```

## Recommended Next Step

Do not keep tuning losses on the same 72 train groups.

Next useful step:

```text
export more oracle action groups from the 131 labeled subckt distribution
```

Then rerun the safest training variant:

```text
heads_hybrid with early stopping on held-out hybrid negative_top1 and Spearman
```

Use epoch 4 or add validation-based checkpoint selection, because this run
showed train rank loss can rebound by epoch 5.

