# AutoResearch Fix Report: Oracle Action-Value Dataset Gate

generated_at: `2026-06-29 03:39 Asia/Shanghai`

## Objective

Implement:

```text
autoresearch/plan-260629-0311/plan.md
```

Scope:

```text
scripts/oracle_action_value_probe.py
scripts/evaluate_oracle_action_values.py
```

## Implemented

### Dataset export / resume

Updated:

```text
scripts/oracle_action_value_probe.py
```

Added:

```text
--resume
--manifest
oracle_groups.tsv
manifest.json
handoff resume metadata
```

Resume behavior:

```text
reuse backend oracle columns keyed by:
  benchmark_id, state_id, candidate_strategy, action_key

always recompute model score fields for the current checkpoint
```

### Fixed oracle checkpoint gate

Added:

```text
scripts/evaluate_oracle_action_values.py
```

Behavior:

```text
read fixed oracle_actions.tsv
rescore exact actions for one or more checkpoints
compute ranking/calibration metrics
aggregate summaries
emit PROMOTE / REJECT / INCONCLUSIVE
```

The gate does not call backend evaluation.

## Validation

### Static / help

Passed:

```bash
python -m py_compile scripts/oracle_action_value_probe.py scripts/evaluate_oracle_action_values.py
python scripts/oracle_action_value_probe.py --help
python scripts/evaluate_oracle_action_values.py --help
```

### Gate smoke

Command used existing oracle TSV:

```text
autoresearch/oracle-action-probe-260629-smoke/oracle_actions.tsv
```

Output:

```text
autoresearch/oracle-action-value-gate-260629-smoke/
```

Records:

```text
rescored_oracle_actions: 288
oracle_action_value_metrics: 24
oracle_action_value_summary: 4
verdict incumbent: INCONCLUSIVE
```

Summary matched prior oracle evals:

```text
best field: hybrid_pred
mean Spearman: 0.327398
negative top1 rate: 0.166667
mean top1 regret: 0.012552
```

### Resume smoke

Output:

```text
autoresearch/oracle-action-probe-260629-resume-smoke/
```

First run:

```text
evaluated_oracle_actions: 1
reused_oracle_actions: 0
```

Second `--resume` run:

```text
evaluated_oracle_actions: 0
reused_oracle_actions: 1
```

Generated:

```text
oracle_groups.tsv
manifest.json
```

## Notes

Current implementation intentionally supports fixed-gate rescoring only for:

```text
state_id=initial
```

This matches the current oracle probe state support. Non-initial state
reconstruction should be added only after state action serialization is defined.

## Status

```text
completed
```
