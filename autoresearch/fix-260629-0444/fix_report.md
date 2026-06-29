# AutoResearch Fix Report: Oracle Action-Value Finetune

generated_at: `2026-06-29 04:44 Asia/Shanghai`

## Objective

Implement:

```text
autoresearch/plan-260629-0437/plan.md
```

Goal:

```text
Use backend-labeled oracle action groups to finetune action-value scoring,
then validate the candidate checkpoint with the fixed oracle checkpoint gate.
```

## Code Added

Added:

```text
scripts/finetune_oracle_action_values.py
```

Behavior:

```text
1. reads oracle_actions.tsv
2. groups actions by benchmark_id + state_id + candidate_strategy
3. supports initial state only
4. loads incumbent checkpoint
5. freezes encoder/dynamics/hard heads by default
6. trains reward_head and return_head
7. optimizes SmoothL1 pointwise value loss and pairwise logistic rank loss
8. writes candidate.pt, history.tsv, handoff.json
```

No changes were made to:

```text
tpi_jepa/model.py
tpi_jepa/train.py
TPIWorldModel architecture
backend evaluation code
```

## Loss

For each oracle action group:

```text
target_i = coverage_scale * oracle_delta_tc_i
pred_i   = reward_pred_i
```

Pointwise:

```text
SmoothL1(pred_i, target_i)
```

Pairwise:

```text
softplus(-sign(target_i - target_j) * (pred_i - pred_j) / temperature)
```

Default trained parameters:

```text
reward_head
return_head
```

## Validation

Passed:

```bash
python -m py_compile scripts/finetune_oracle_action_values.py scripts/evaluate_oracle_action_values.py scripts/oracle_action_value_probe.py
python scripts/finetune_oracle_action_values.py --help
```

Tiny smoke:

```text
oracle_actions: autoresearch/oracle-action-probe-260629-resume-smoke/oracle_actions.tsv
out_dir: autoresearch/oracle-action-value-finetune-260629-tiny
candidate.pt: written
history.tsv: written
handoff.json: written
```

Tiny smoke note:

```text
The tiny oracle TSV has one finite action, so it validates pointwise training
and checkpoint output only. It has no pairwise rank pairs.
```

## Real Smoke

Training oracle data:

```text
autoresearch/oracle-action-probe-260629-smallckt/oracle_actions.tsv
```

Output:

```text
autoresearch/oracle-action-value-finetune-260629-smallckt/
```

Training summary:

| epoch | loss | value loss | rank loss | groups | pairs |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.407496 | 3.062430 | 0.690133 | 12 | 2257 |
| 2 | 3.305246 | 2.959852 | 0.690789 | 12 | 2257 |
| 3 | 3.230899 | 2.884395 | 0.693009 | 12 | 2257 |

Interpretation:

```text
Pointwise value loss decreased.
Pairwise rank loss did not improve materially.
```

## Gate Result

Held-out oracle gate:

```text
oracle_actions: autoresearch/oracle-action-probe-260629-smoke/oracle_actions.tsv
out_dir: autoresearch/oracle-action-value-gate-260629-candidate
baseline: incumbent
candidate: autoresearch/oracle-action-value-finetune-260629-smallckt/candidate.pt
```

Verdict:

```text
candidate: INCONCLUSIVE
incumbent: INCONCLUSIVE
```

Summary:

| checkpoint | score_field | mean Spearman | negative top1 rate | mean top1 regret |
|---|---|---:|---:|---:|
| incumbent | hybrid_pred | 0.327398 | 0.166667 | 0.012552 |
| candidate | hybrid_pred | 0.318279 | 0.166667 | 0.012552 |
| incumbent | reward_pred | 0.294742 | 0.500000 | 0.020223 |
| candidate | reward_pred | 0.278469 | 0.500000 | 0.020223 |
| incumbent | hard_reduction_total_pred | 0.324443 | 0.166667 | 0.012552 |
| candidate | hard_reduction_total_pred | 0.324443 | 0.166667 | 0.012552 |

Interpretation:

```text
The candidate did not improve held-out oracle ranking.
reward_pred and guarded_reward got worse on mean Spearman.
hybrid_pred got slightly worse.
hard_reduction_total_pred stayed unchanged because hard_reduction_head was frozen.
```

This is a useful negative result:

```text
small-subckt oracle action-value supervision on reward/return heads alone did
not transfer to b15_C/i2c_aig oracle action ranking.
```

## Next Recommended Iteration

Do not promote this candidate.

Next plausible variants:

```text
1. Train on eval-like oracle groups, not only tiny/small subcircuits.
2. Increase rank loss strength and train longer only after adding more oracle groups.
3. Add a dedicated action_value_head if reward_pred should remain tied to historical reward.
4. Try unfreezing dynamics/action_encoder with a lower lr, but only with larger oracle data.
```

The most defensible next step:

```text
Generate more oracle action groups for eval-like circuits, then rerun this
finetune/gate loop.
```

Reason:

```text
The current training oracle set has only 12 groups from small subcircuits.
The held-out failures are on larger eval circuits with different negative-action
distribution.
```
