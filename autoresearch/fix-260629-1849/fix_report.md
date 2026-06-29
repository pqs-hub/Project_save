# Fix Report: Oracle Split Audit And Balanced Groups

## Objective

Execute:

```text
$autoresearch fix autoresearch/plan-260629-1839/plan.md
```

Goal:

```text
Audit train / expanded val / transfer oracle split mismatch and build
negative-action-balanced train/val subsets before training another fixed ranker.
```

## Code Changes

Added:

```text
scripts/audit_oracle_action_groups.py
scripts/build_balanced_oracle_action_subset.py
```

No model, training loop, planner, or checkpoint code was changed.

## Audit Result

Audit output:

```text
autoresearch/oracle-split-audit-260629/oracle_group_audit_report.md
```

Key split mismatch:

| split | rows | groups | negative rate | all-positive groups |
|---|---:|---:|---:|---:|
| train | 5184 | 288 | 13.81% | 168 |
| expanded val | 864 | 48 | 41.55% | 11 |
| transfer | 288 | 6 | 34.03% | 3 |

Key action-type mismatch:

| split | control0 negative | control1 negative | observe negative |
|---|---:|---:|---:|
| train | 20.89% | 19.44% | 1.10% |
| expanded val | 63.89% | 57.99% | 2.78% |
| transfer | 50.00% | 50.00% | 2.08% |

Interpretation:

```text
The current train oracle data under-represents negative control0/control1 actions.
That explains why the fixed ranker can improve expanded val but fails transfer safety.
```

## Balanced Subset Result

Balanced output:

```text
autoresearch/oracle-balanced-groups-260629/
```

Policy:

```text
min_negatives_per_group = 3
min_positives_per_group = 3
prefer_negative_types = control0,control1
max_actions_per_group = 18
```

Result:

| split | kept groups | kept rows | negative rate | group target met |
|---|---:|---:|---:|---:|
| train | 71 | 1278 | 50.08% | no |
| expanded val | 34 | 612 | 58.01% | yes |
| transfer | 0 | 0 | eval only | eval only |

The transfer split is explicitly eval-only and was not written into train/val balanced subsets.

## Verdict

```text
completed_needs_more_oracle
```

Reason:

```text
Existing labels are enough to build a balanced validation set, but not enough to
build the minimum useful balanced train set.
```

The target was:

```text
balanced_train groups >= 80
```

Actual:

```text
balanced_train groups = 71
```

## Verification

Passed:

```text
python -m py_compile scripts/audit_oracle_action_groups.py scripts/build_balanced_oracle_action_subset.py
python scripts/audit_oracle_action_groups.py ...
python scripts/build_balanced_oracle_action_subset.py ...
test -s autoresearch/oracle-split-audit-260629/oracle_group_audit_report.md
test -s autoresearch/oracle-balanced-groups-260629/balance_report.md
python -m json.tool autoresearch/oracle-split-audit-260629/handoff.json
python -m json.tool autoresearch/oracle-balanced-groups-260629/handoff.json
```

## Artifacts

```text
autoresearch/oracle-split-audit-260629/oracle_group_audit_report.md
autoresearch/oracle-split-audit-260629/oracle_group_audit_summary.tsv
autoresearch/oracle-split-audit-260629/oracle_group_audit_by_group.tsv
autoresearch/oracle-balanced-groups-260629/balanced_train_oracle_actions.tsv
autoresearch/oracle-balanced-groups-260629/balanced_val_oracle_actions.tsv
autoresearch/oracle-balanced-groups-260629/balance_report.md
autoresearch/oracle-balanced-groups-260629/balanced_manifest.json
```

## Recommended Next Step

Collect more negative-rich oracle groups before rerunning the fixed ranker.

Recommended next command:

```text
$autoresearch plan Collect negative-rich oracle action groups from sampled subckts with control-heavy candidate pools, then rebuild balanced train data.
```

Do not train on transfer rows.

