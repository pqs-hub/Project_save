# Version A Result: Disable hard_count / FC / return

## Verdict

Version A is **not promoted for planner/oracle ranking**.

But Version A shows `hard_count` loss is probably not required for the basic node-level hard classifier: hard F1 improves versus incumbent.

## Hybrid Score Summary

| checkpoint | expanded Spearman | expanded neg top1 | transfer Spearman | transfer neg top1 | hard F1 tuned | hard reduction score | predictive score |
|---|---:|---:|---:|---:|---:|---:|---:|
| `incumbent` | 0.031476 | 0.375000 | 0.327398 | 0.166667 | 0.794805 | 0.798620 | 0.820147 |
| `control` | 0.014643 | 0.416667 | 0.276475 | 0.500000 | 0.782191 | 0.833744 | 0.822154 |
| `version_A` | -0.040485 | 0.562500 | -0.084822 | 0.500000 | 0.810817 | 0.830606 | 0.778059 |

## Interpretation

- Version A hard F1: `0.810817`, incumbent hard F1: `0.794805`. So disabling `hard_count` loss did not hurt the hard classifier.
- Version A hard-reduction score: `0.830606`, incumbent: `0.798620`. The direct hard-reduction head also did not collapse.
- Version A oracle ranking is bad: expanded hybrid Spearman `-0.040485`, transfer hybrid Spearman `-0.084822`. This fails the planner-ranking gate.
- Control retrain also has weaker oracle transfer than incumbent, so the retrain setup itself still differs from incumbent behavior even when weights match.
- The useful conclusion is narrow: `hard_count` loss looks unnecessary for hard classification, but removing it may change action-score ordering.

## Decision

- Do not use Version A as planner checkpoint.
- Keep Version A as evidence that `lambda_hard_count` can be reduced or removed if the only target is hard-label quality.
- Do not add oracle ranking on top of Version A yet.
- Before Version B, implement derived hard-count/reduction metrics, because old oracle fields will not represent Version B correctly.

## Artifacts

- `autoresearch/autoresearch-260629-1550/configs/control_incumbent_like.json`
- `autoresearch/autoresearch-260629-1550/configs/version_A_no_hard_count.json`
- `autoresearch/autoresearch-260629-1550/runs/control_incumbent_like/history.csv`
- `autoresearch/autoresearch-260629-1550/runs/version_A_no_hard_count/history.csv`
- `autoresearch/autoresearch-260629-1550/gates/hard`
- `autoresearch/autoresearch-260629-1550/gates/expanded_val/oracle_action_value_summary.tsv`
- `autoresearch/autoresearch-260629-1550/gates/transfer/oracle_action_value_summary.tsv`
- `autoresearch/autoresearch-260629-1550/version_A_summary.tsv`
