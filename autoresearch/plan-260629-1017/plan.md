# AutoResearch Plan: Joint Planner-Score Oracle Ranking Finetune

generated_at: `2026-06-29 10:17 Asia/Shanghai`

## Goal

Implement and validate a finetuning mode that directly trains the model to rank
candidate actions by backend-labeled true `delta_tc`.

The important change is:

```text
train the score used by the planner, not only reward_pred
```

This plan is specifically for:

```text
Joint-train action_encoder/dynamics/reward/hard_reduction on oracle pairwise
planner-score ranking instead of head-only reward finetune.
```

## Background

Previous experiments failed:

```text
1. reward/return head-only value+rank finetune
2. reward/return head-only pairwise-only finetune
```

The useful negative result is:

```text
pairwise ranking is the right objective family, but the current trainable scope
and score field are wrong.
```

Current training script limitation:

```text
scripts/finetune_oracle_action_values.py
```

only predicts:

```text
reward_pred
return_pred
```

and `set_trainable_parts()` only supports:

```text
reward
return
hard_reduction
all
```

Current planner scoring formula in:

```text
tpi_jepa/plan.py
```

is:

```python
hybrid_pred = return_pred + reward_pred + hard_reduction_total * coverage_scale
```

So the next training objective must align to `hybrid_pred` or an equivalent
differentiable planner score.

## Scope

### In Scope

Modify:

```text
scripts/finetune_oracle_action_values.py
```

Add:

```text
--train-scope planner_joint
--ranking-score-field hybrid_pred
```

or equivalent names if the implementation is cleaner.

The new mode should train:

```text
action_encoder
dynamics
reward_head
return_head
hard_reduction_head
```

and keep frozen initially:

```text
online_encoder
target_encoder
```

Primary loss:

```text
pairwise logistic ranking loss on hybrid_pred
```

Use oracle target:

```text
coverage_scale * oracle_delta_tc
```

### Out Of Scope

Do not add a dedicated `action_value_head` in this step.

Do not add multiple safety losses in this step.

Do not change candidate generation.

Do not regenerate oracle labels unless the existing TSV is missing or corrupt.

## Design

### Trainable Scope

Add a trainable scope helper that supports both legacy and joint modes.

Required behavior:

```text
--train-heads reward,return
```

must keep current behavior for existing experiments.

New behavior:

```text
--train-scope planner_joint
```

sets `requires_grad=True` for parameter names starting with:

```text
action_encoder
dynamics
reward_head
return_head
hard_reduction_head
```

and `False` for:

```text
online_encoder
target_encoder
```

Reason:

```text
The latent encoder was trained on only 131 labeled subgraphs and should not be
allowed to drift from 1296 oracle actions in the first joint experiment.
```

### Differentiable Score Fields

Extend `predict_group_scores()` to return a dict of tensors:

```text
reward_pred
return_pred
guarded_reward
hard_reduction_total_pred
hybrid_pred
```

For training, `hybrid_pred` must be tensor-valued and keep gradients:

```python
hard_reduction_total = hard_reduction_pred.view(-1)[0]
hybrid_pred = return_pred + reward_pred + hard_reduction_total * coverage_scale
```

Do not call `.detach()`, `.item()`, or `float()` inside the training score
path.

### Ranking Loss

Use the existing pairwise logistic form:

```python
softplus(-sign(target_i - target_j) * (score_i - score_j) / temperature)
```

but apply it to:

```text
scores[args.ranking_score_field]
```

not hardcoded `reward_preds`.

Recommended defaults for this experiment:

```text
--lambda-oracle-value 0.0
--lambda-oracle-rank 1.0
--ranking-score-field hybrid_pred
--pairwise-min-delta 0.001
--pairwise-temperature 1.0
--lr 1e-5
--epochs 5
```

Use a smaller LR than head-only finetune because dynamics/action encoder are
now trainable.

### Optional Value Loss Compatibility

Keep the existing value loss for backwards compatibility.

If `--lambda-oracle-value > 0`, apply value loss to:

```text
args.value-score-field
```

defaulting to:

```text
reward_pred
```

But the first joint experiment should use:

```text
--lambda-oracle-value 0.0
```

to isolate ranking.

## Implementation Steps

1. Edit `scripts/finetune_oracle_action_values.py`.
2. Add CLI args:

```text
--train-scope
--ranking-score-field
--value-score-field
```

3. Preserve existing `--train-heads` behavior when `--train-scope heads`.
4. Add `planner_joint` trainable scope.
5. Change `predict_group_scores()` to return score tensor dict instead of only
   `(reward_preds, return_preds)`.
6. Compute rank loss using `scores[args.ranking_score_field]`.
7. Save selected score field and train scope in `handoff.json`.
8. Run a syntax check.
9. Run a one-epoch smoke finetune on a small action subset.
10. Run the full joint finetune on labeled-subckt train oracle data.
11. Gate candidate on held-out labeled-subckt val oracle data.
12. Gate candidate on full-circuit smoke oracle data for transfer risk.

## Commands

### Syntax Check

```bash
python -m py_compile scripts/finetune_oracle_action_values.py
```

### Smoke Finetune

```bash
python scripts/finetune_oracle_action_values.py \
  --checkpoint autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt \
  --oracle-actions autoresearch/oracle-action-probe-260629-labeled-subckt-train/oracle_actions.tsv \
  --out-dir autoresearch/oracle-action-value-finetune-260629-joint-smoke \
  --epochs 1 \
  --lr 1e-5 \
  --lambda-oracle-value 0.0 \
  --lambda-oracle-rank 1.0 \
  --ranking-score-field hybrid_pred \
  --train-scope planner_joint \
  --max-actions-per-group 8
```

### Full Joint Finetune

```bash
python scripts/finetune_oracle_action_values.py \
  --checkpoint autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt \
  --oracle-actions autoresearch/oracle-action-probe-260629-labeled-subckt-train/oracle_actions.tsv \
  --out-dir autoresearch/oracle-action-value-finetune-260629-joint-hybrid \
  --epochs 5 \
  --lr 1e-5 \
  --lambda-oracle-value 0.0 \
  --lambda-oracle-rank 1.0 \
  --ranking-score-field hybrid_pred \
  --train-scope planner_joint
```

### Held-Out Labeled-Subckt Gate

```bash
python scripts/evaluate_oracle_action_values.py \
  --oracle-actions autoresearch/oracle-action-probe-260629-labeled-subckt-val/oracle_actions.tsv \
  --checkpoint incumbent=autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt \
  --checkpoint candidate=autoresearch/oracle-action-value-finetune-260629-joint-hybrid/candidate.pt \
  --baseline incumbent \
  --score-fields reward_pred,guarded_reward,hard_reduction_total_pred,hybrid_pred \
  --out-dir autoresearch/oracle-action-value-gate-260629-joint-hybrid-val
```

### Transfer Gate

```bash
python scripts/evaluate_oracle_action_values.py \
  --oracle-actions autoresearch/oracle-action-probe-260629-smoke/oracle_actions.tsv \
  --checkpoint incumbent=autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt \
  --checkpoint candidate=autoresearch/oracle-action-value-finetune-260629-joint-hybrid/candidate.pt \
  --baseline incumbent \
  --score-fields reward_pred,guarded_reward,hard_reduction_total_pred,hybrid_pred \
  --out-dir autoresearch/oracle-action-value-gate-260629-joint-hybrid-transfer
```

## Metrics

Primary score field:

```text
hybrid_pred
```

Primary distribution:

```text
held-out labeled subckt val
```

Primary metrics:

```text
mean_spearman
negative_top1_rate
mean_top1_regret
```

Secondary metrics:

```text
reward_pred Spearman
hard_reduction_total_pred Spearman
transfer hybrid_pred Spearman
transfer negative_top1_rate
```

## Acceptance Criteria

Promote candidate only if held-out labeled-subckt `hybrid_pred` satisfies:

```text
mean_spearman >= incumbent + 0.05
negative_top1_rate <= incumbent
mean_top1_regret <= incumbent
```

Reject candidate if held-out labeled-subckt `hybrid_pred` satisfies any:

```text
negative_top1_rate > incumbent
mean_spearman < incumbent - 0.02
mean_top1_regret > incumbent + 0.005
```

If held-out improves but full-circuit transfer regresses heavily, mark:

```text
INCONCLUSIVE / needs more oracle data
```

not immediate promote.

## Expected Outcomes

### Success Case

If this works, the result should show:

```text
hybrid_pred ranking improves on held-out labeled subckts
negative top1 does not worsen
top1 regret does not worsen
```

This would validate that the problem was mainly:

```text
head-only scope and reward-only score mismatch
```

### Failure Case

If rank loss decreases but held-out `hybrid_pred` does not improve, the likely
causes are:

```text
1. 1296 oracle train actions are still too small
2. frozen online_encoder cannot represent oracle action value
3. hybrid_pred mixes incompatible heads and needs a dedicated action_value_pred
```

Then the next branch should be:

```text
collect more oracle groups from the 131 labeled subckts
```

or:

```text
add dedicated action_value_head only after this joint planner-score test fails
```

## Risks

Risk:

```text
planner_joint overfits 72 train groups.
```

Mitigation:

```text
use held-out 24 groups as primary gate, small LR, frozen encoders.
```

Risk:

```text
hybrid_pred scale is dominated by hard_reduction_total * coverage_scale.
```

Mitigation:

```text
report component score fields separately; if hard_reduction dominates, add a
later normalized planner score experiment.
```

Risk:

```text
evaluate_oracle_action_values.py gate verdict is best-by-Spearman, while this
experiment should judge hybrid_pred specifically.
```

Mitigation:

```text
read oracle_action_value_summary.tsv directly and compare hybrid_pred rows.
```

## Next Step

Run:

```text
$autoresearch fix autoresearch/plan-260629-1017/plan.md
```

