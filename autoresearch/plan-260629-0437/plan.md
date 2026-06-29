# AutoResearch Plan: Action-Value Training Loss with Oracle Groups

generated_at: `2026-06-29 04:37 Asia/Shanghai`

## Goal

Design and implement a minimal action-value training path that uses backend
labeled oracle action groups:

```text
fixed state + multiple candidate actions + backend-measured oracle_delta_tc
```

Then validate the candidate checkpoint with:

```text
scripts/evaluate_oracle_action_values.py
```

against the incumbent checkpoint.

## Current State

Already implemented:

```text
scripts/oracle_action_value_probe.py
  --resume
  oracle_groups.tsv
  manifest.json

scripts/evaluate_oracle_action_values.py
  fixed oracle_actions.tsv checkpoint gate
  PROMOTE / REJECT / INCONCLUSIVE
```

Current incumbent oracle-gate baseline:

| score_field | mean Spearman | negative top1 rate | mean top1 regret |
|---|---:|---:|---:|
| `hybrid_pred` | `0.327398` | `0.166667` | `0.012552` |
| `hard_reduction_total_pred` | `0.324443` | `0.166667` | `0.012552` |
| `guarded_reward` | `0.310535` | `0.500000` | `0.020223` |
| `reward_pred` | `0.294742` | `0.500000` | `0.020223` |

The next missing artifact is a candidate checkpoint trained to improve these
oracle action-value metrics.

## Why Existing Training Is Not Enough

Current `tpi_jepa/train.py` trains one historical transition at a time:

```text
one sample = one logged action
target = delta_fault_coverage / hard_reduction / latent transition
```

This does not tell the model:

```text
within the same state, action A is better than action B
```

But the planner needs exactly that:

```text
rank many candidate actions for the same circuit state
```

Therefore the action-value training path must use grouped labels:

```text
group = benchmark_id + state_id + candidate_strategy
items = candidate actions in that state
label = oracle_delta_tc
```

## Design Decision

Do not start by adding a new model head.

Use the existing heads first:

```text
reward_pred
return_pred
hard_reduction_pred
```

Train primarily through:

```text
reward_pred -> oracle_delta_tc * coverage_scale
```

Reason:

```text
1. reward_pred is already used by planner scoring
2. score_candidate_from_latent already exposes it
3. this is the smallest model-compatible change
4. if this improves the oracle gate, a dedicated action_value_head becomes a justified P1 change
```

## Proposed New Script

Add:

```text
scripts/finetune_oracle_action_values.py
```

Purpose:

```text
Load an incumbent checkpoint, train on oracle_actions.tsv groups, and save a
candidate checkpoint that can be compared with the fixed oracle gate.
```

This should be separate from the main `tpi_jepa/train.py` first.

Reason:

```text
The oracle dataset is grouped by fixed states and candidate actions. It does
not match the current TPIDataset/TPIRolloutDataset sample shape cleanly.
Keeping it separate avoids destabilizing the existing training pipeline.
```

## Oracle Dataset Contract

Input:

```text
oracle_actions.tsv
```

Required fields:

```text
benchmark_id
state_id
candidate_strategy
node
type
action_key
oracle_delta_tc
```

V1 restriction:

```text
state_id must be initial
```

This matches current oracle probe support and avoids state reconstruction
ambiguity.

Filtering:

```text
keep rows with finite oracle_delta_tc
drop groups with fewer than 2 finite actions
optionally cap actions per group with --max-actions-per-group
preserve negative actions; do not train only on positives
```

## Training Objective

For each group:

```text
pred_i = reward_pred(action_i)
target_i = coverage_scale * oracle_delta_tc_i
```

Pointwise loss:

```text
L_value = mean_i SmoothL1(pred_i, target_i)
```

Pairwise ranking loss:

For pairs where target values differ by at least `min_delta`:

```text
target_order_ij = sign(target_i - target_j)
score_diff_ij = pred_i - pred_j
L_rank = softplus(-target_order_ij * score_diff_ij / temperature)
```

Total:

```text
L = lambda_oracle_value * L_value
  + lambda_oracle_rank  * L_rank
  + lambda_weight_decay_guard * parameter_regularization_optional
```

Initial defaults:

```text
lambda_oracle_value = 1.0
lambda_oracle_rank = 0.5
coverage_scale = checkpoint config coverage_scale, default 100.0
pairwise_min_delta = 0.001
pairwise_temperature = 1.0
lr = 1e-4
epochs = 3
```

Why pairwise:

```text
Planner needs ranking more than exact numeric calibration.
```

Why pointwise too:

```text
negative_top1 requires sign/calibration. A pure pairwise loss can rank actions
but still put every score positive.
```

## Parameters to Train

Default v1:

```text
freeze encoder and dynamics
train reward_head and return_head only
```

Config:

```text
--train-heads reward,return
--unfreeze-dynamics false
```

Reason:

```text
The oracle dataset is small. Updating the whole model risks destroying
hard-fault representation and latent transition quality.
```

Optional v2 if v1 underfits:

```text
unfreeze dynamics and hard_reduction_head with low lr
```

## Candidate Scoring During Finetune

Use the same scoring path as planner/gate:

```text
make_base_node_features
make_state_features
model.online_encoder
score_candidate_from_latent or model.predict_from_latent
```

For efficiency:

```text
encode z_state once per group
score all actions in that group
```

Important:

```text
Do not call backend during finetune.
```

## Data Split

Use existing oracle probes:

Train candidate v0 on small oracle groups:

```text
autoresearch/oracle-action-probe-260629-smallckt/oracle_actions.tsv
```

Validation / gate:

```text
autoresearch/oracle-action-probe-260629-smoke/oracle_actions.tsv
```

This intentionally tests whether small-subckt action-value supervision transfers
to eval-circuit oracle groups.

If this is too weak, next dataset expansion should generate more oracle groups
before changing architecture.

## CLI

Required command:

```bash
python scripts/finetune_oracle_action_values.py \
  --checkpoint <incumbent.pt> \
  --oracle-actions autoresearch/oracle-action-probe-260629-smallckt/oracle_actions.tsv \
  --out-dir autoresearch/oracle-action-value-finetune-260629-smoke \
  --epochs 3 \
  --lr 1e-4 \
  --lambda-oracle-value 1.0 \
  --lambda-oracle-rank 0.5 \
  --pairwise-min-delta 0.001 \
  --pairwise-temperature 1.0 \
  --train-heads reward,return \
  --plan-device cpu
```

Outputs:

```text
candidate.pt
history.tsv
handoff.json
```

History fields:

```text
epoch
train_loss
train_value_loss
train_rank_loss
train_groups
train_pairs
```

## Validation Gate

After finetune:

```bash
python scripts/evaluate_oracle_action_values.py \
  --oracle-actions autoresearch/oracle-action-probe-260629-smoke/oracle_actions.tsv \
  --checkpoints incumbent=<incumbent.pt>,candidate=autoresearch/oracle-action-value-finetune-260629-smoke/candidate.pt \
  --score-fields reward_pred,guarded_reward,hard_reduction_total_pred,hybrid_pred \
  --top-ks 8,16,32 \
  --oracle-top-m 5 \
  --plan-device cpu \
  --out-dir autoresearch/oracle-action-value-gate-260629-candidate \
  --baseline incumbent
```

Promotion threshold:

```text
mean_spearman >= incumbent + 0.10
negative_top1_rate <= incumbent - 0.10
mean_top1_regret <= incumbent - 0.01
```

Guardrail:

```text
REJECT if Spearman improves but negative_top1_rate increases
```

## Implementation Scope

In scope:

```text
scripts/finetune_oracle_action_values.py
minor helper reuse from scripts/evaluate_oracle_action_values.py
minor helper reuse from scripts/oracle_action_value_probe.py
```

Out of scope:

```text
modifying TPIWorldModel architecture
modifying tpi_jepa/train.py main training loop
full 8-circuit oracle dataset generation
non-initial state support
```

## Verification Commands

Static:

```bash
python -m py_compile \
  scripts/finetune_oracle_action_values.py \
  scripts/evaluate_oracle_action_values.py \
  scripts/oracle_action_value_probe.py
```

Help:

```bash
python scripts/finetune_oracle_action_values.py --help
```

Tiny overfit smoke:

```bash
python scripts/finetune_oracle_action_values.py \
  --checkpoint autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt \
  --oracle-actions autoresearch/oracle-action-probe-260629-resume-smoke/oracle_actions.tsv \
  --out-dir autoresearch/oracle-action-value-finetune-260629-tiny \
  --epochs 1 \
  --lr 1e-4 \
  --lambda-oracle-value 1.0 \
  --lambda-oracle-rank 0.5 \
  --plan-device cpu
```

Expected tiny smoke:

```text
script exits 0
candidate.pt exists
history.tsv exists
handoff.json exists
```

Real smoke:

```bash
python scripts/finetune_oracle_action_values.py \
  --checkpoint autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt \
  --oracle-actions autoresearch/oracle-action-probe-260629-smallckt/oracle_actions.tsv \
  --out-dir autoresearch/oracle-action-value-finetune-260629-smallckt \
  --epochs 3 \
  --lr 1e-4 \
  --lambda-oracle-value 1.0 \
  --lambda-oracle-rank 0.5 \
  --pairwise-min-delta 0.001 \
  --pairwise-temperature 1.0 \
  --train-heads reward,return \
  --plan-device cpu
```

Candidate gate:

```bash
python scripts/evaluate_oracle_action_values.py \
  --oracle-actions autoresearch/oracle-action-probe-260629-smoke/oracle_actions.tsv \
  --checkpoints incumbent=autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt,candidate=autoresearch/oracle-action-value-finetune-260629-smallckt/candidate.pt \
  --score-fields reward_pred,guarded_reward,hard_reduction_total_pred,hybrid_pred \
  --top-ks 8,16,32 \
  --oracle-top-m 5 \
  --plan-device cpu \
  --out-dir autoresearch/oracle-action-value-gate-260629-candidate \
  --baseline incumbent
```

## Acceptance Criteria

Implementation is acceptable if:

```text
finetune script compiles
finetune --help works
tiny smoke writes candidate checkpoint
real smoke writes candidate checkpoint
candidate gate runs without backend calls
gate outputs candidate verdict
```

Research success is stronger:

```text
candidate gets PROMOTE
```

If candidate is `INCONCLUSIVE`, inspect:

```text
whether train oracle loss decreased
whether reward_pred improved but hybrid did not
whether small-subckt oracle supervision transfers poorly to eval circuits
```

If candidate is `REJECT`, likely causes:

```text
oracle train set too small or distribution-shifted
pointwise loss overfit sign/calibration but hurt ranking
rank loss too weak/strong
unfreezing policy too broad
```

## Stop Criteria

Stop this route after two reasonable variants if:

```text
no improvement in held-out oracle mean Spearman
or negative_top1_rate increases consistently
```

Then switch to:

```text
expand oracle train dataset across eval-like circuits
or add dedicated action_value_head
or analyze candidate-pool/action-type distribution mismatch
```

Do not switch before running at least one oracle-supervised candidate checkpoint.
