# AutoResearch Reason Report: What Should We Fix Next?

generated_at: `2026-06-29 09:47 Asia/Shanghai`

## Question

After oracle action-value finetuning on the 131 labeled subgraphs, should the
next move be:

```text
A. add sign/top1 safety loss
B. add a dedicated action_value_head
C. collect more oracle data / do not change model yet
```

## Facts

Current labeled-subckt oracle experiment:

```text
train oracle: 1296 actions, 72 groups
val oracle: 432 actions, 24 groups
finetune: reward_head + return_head
```

Held-out subckt gate:

| checkpoint | score_field | Spearman | negative top1 | top1 regret |
|---|---|---:|---:|---:|
| incumbent | hybrid_pred | 0.005693 | 0.458333 | 0.043452 |
| candidate | hybrid_pred | 0.004533 | 0.416667 | 0.043300 |
| incumbent | reward_pred | -0.137870 | 0.416667 | 0.023290 |
| candidate | reward_pred | -0.205044 | 0.541667 | 0.031650 |

Full-circuit transfer gate:

| checkpoint | score_field | Spearman | negative top1 | top1 regret |
|---|---|---:|---:|---:|
| incumbent | hybrid_pred | 0.327398 | 0.166667 | 0.012552 |
| candidate | hybrid_pred | 0.325323 | 0.166667 | 0.012557 |
| incumbent | reward_pred | 0.294742 | 0.500000 | 0.020223 |
| candidate | reward_pred | 0.348502 | 0.333333 | 0.022373 |

Gate result:

```text
held-out subckt: INCONCLUSIVE
full-circuit transfer: REJECT
```

## Position A: Add Sign/Top1 Safety Loss

Argument:

```text
The failure is specifically top1 safety and sign calibration, not just ranking.
```

Evidence:

```text
candidate reward_pred transfer Spearman improved from 0.294742 to 0.348502,
but the candidate was rejected because negative_top1 remained worse than the
incumbent best field.
```

Strength:

```text
Minimal code change.
Does not alter checkpoint architecture.
Directly targets the gate failure.
Can reuse existing oracle train/val data.
```

Weakness:

```text
If reward_pred is semantically overloaded by historical delta_fault_coverage,
more loss terms may create conflicting gradients instead of a clean action
value score.
```

Counterargument:

```text
That conflict is real, but Stage A is still cheaper and more diagnostic than
adding a new head. If sign/top1 loss fails, it gives stronger evidence that the
score representation itself needs separation.
```

Verdict:

```text
Best next move.
```

## Position B: Add Dedicated action_value_head Now

Argument:

```text
reward_pred is the wrong semantic container. It was trained on historical
coverage deltas, not oracle candidate action values. A separate head avoids
destroying reward_pred while learning oracle action value.
```

Strength:

```text
Cleaner modeling.
Can expose action_value_pred as a planner score.
Avoids mixing old reward target with oracle target.
```

Weakness:

```text
Requires model.py, plan.py, gate script, checkpoint compatibility, and planner
score-field changes.
More moving parts.
If data is still insufficient, the new head can overfit just as reward_head did.
```

Counterargument:

```text
The current evidence does not yet prove the head is the bottleneck. It proves
the current loss does not optimize top1 safety. Try explicit safety loss first.
```

Verdict:

```text
Second-stage move if safety loss fails.
```

## Position C: Collect More Oracle Data First

Argument:

```text
The oracle data is still small relative to circuit/action diversity. More data
may solve the problem without loss changes.
```

Strength:

```text
Always helpful for reliability.
Reduces variance across subckt samples.
May expose more negative actions.
```

Weakness:

```text
We already have 1296 train actions and 432 val actions from the correct
training distribution. The observed failure is not simply "no data"; it is that
the objective does not punish unsafe top1 choices enough.
```

Counterargument:

```text
More data is useful, but collecting it before fixing the loss can be wasteful.
The loss currently has no direct term for negative top1.
```

Verdict:

```text
Do later if safety loss has high variance or underfits.
```

## Blind Judge Decision

Decision:

```text
Proceed with Stage A from autoresearch/plan-260629-0944.
```

Required discipline:

```text
Do not judge success by Spearman alone.
```

Primary comparison:

```text
negative_top1_rate
top1_regret
Spearman as a guardrail, not sole target
```

If Stage A yields:

```text
negative_top1 down, top1_regret flat/down, Spearman slightly down
```

Then it is still potentially useful for planner safety.

If Stage A yields:

```text
negative_top1 up
```

Reject immediately.

If Stage A yields:

```text
all metrics flat
```

Then move to Stage B dedicated `action_value_head`.

## Important Caveat

The current checkpoint gate verdict logic chooses a candidate's best score
field by Spearman. This can make a candidate look tempting through
`reward_pred` even when `hybrid_pred` is the actual safer planner score.

For the next fix, report both:

```text
best-by-Spearman verdict
planner-field verdict for hybrid_pred
planner-field verdict for reward_pred
```

This prevents hiding a safety regression behind a different score field.

## Next Command

```text
$autoresearch fix autoresearch/plan-260629-0944/plan.md
```
