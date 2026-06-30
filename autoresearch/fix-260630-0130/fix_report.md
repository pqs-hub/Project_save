# Fix Report: Q Calibration Fixed-Candidate Ablation

source plan: `autoresearch/plan-260630-0124/plan.md`

generated at: `2026-06-30 01:30`

## What Was Changed

Added a dedicated post-hoc Q calibration evaluator:

- `scripts/evaluate_q_calibration.py`

The evaluator reuses already rescored oracle action TSVs and does not regenerate candidates, does not rerun backend labeling, and does not add training loss.

Inputs used:

- `autoresearch/q-oracle-260629/gates/expanded_val/rescored_oracle_actions.tsv`
- `autoresearch/q-oracle-260629/gates/transfer/rescored_oracle_actions.tsv`

Score field:

- `q_pred`

Checkpoints:

- `Q_v0_rank0p5`
- `Q_v0_rank1p0`
- `Q_v0_rank2p0`
- `Q_v0_value1_rank1`

Calibration methods:

- `raw`
- `group_center`
- `group_zscore`
- `group_rank_pct`
- `circuit_zscore`
- `global_zscore`
- `platt`

## Command

```bash
python scripts/evaluate_q_calibration.py \
  --expanded-rescored autoresearch/q-oracle-260629/gates/expanded_val/rescored_oracle_actions.tsv \
  --transfer-rescored autoresearch/q-oracle-260629/gates/transfer/rescored_oracle_actions.tsv \
  --checkpoints Q_v0_rank0p5,Q_v0_rank1p0,Q_v0_rank2p0,Q_v0_value1_rank1 \
  --score-field q_pred \
  --methods raw,group_center,group_zscore,group_rank_pct,circuit_zscore,global_zscore,platt \
  --out-dir autoresearch/q-calibration-260630
```

## Outputs

- `autoresearch/q-calibration-260630/q_calibrated_actions.tsv`
- `autoresearch/q-calibration-260630/q_calibration_metrics.tsv`
- `autoresearch/q-calibration-260630/q_calibration_summary.tsv`
- `autoresearch/q-calibration-260630/q_calibration_promotion.tsv`
- `autoresearch/q-calibration-260630/q_calibration_report.md`
- `autoresearch/q-calibration-260630/handoff.json`

## Result

No calibration variant promoted.

| Best View | Checkpoint | Method | Value | Verdict |
|---|---|---:|---:|---|
| best expanded Spearman | `Q_v0_rank1p0` | `raw` | 0.576224 | REJECT |
| best expanded negative top1 | `Q_v0_rank1p0` | `raw` | 0.135135 | REJECT |
| best transfer Spearman | `Q_v0_value1_rank1` | `raw` | 0.300224 | REJECT |
| best transfer regret | `Q_v0_value1_rank1` | `raw` | 0.012273 | REJECT |
| best transfer sign accuracy | `Q_v0_value1_rank1` | `global_zscore` | 0.833333 | REJECT |

Promotion rows:

- total rows: `28`
- promoted rows: `0`

## Important Finding

Most calibration methods are monotonic transforms of `q_pred`.

That means they can change score scale and sign interpretation, but they do not change which action is ranked first inside a fixed candidate group.

Observed rank changes:

- `raw`: `0/0`
- `group_center`: `0/0`
- `group_zscore`: `0/0`
- `circuit_zscore`: `0/0`
- `global_zscore`: `0/0`
- `platt`: `0/0`
- `group_rank_pct`: only minor tie-related changes in 3 rows

Therefore, fixed-candidate top1 and top1 regret stay almost exactly unchanged.

## Conclusion

Q calibration is useful for interpreting whether a score looks positive or negative, but it is not enough to fix action selection.

The current failure is mainly an ordering problem inside each candidate group:

- some checkpoints can rank actions reasonably on expanded validation,
- some checkpoints transfer better,
- but calibration does not repair cases where the model puts a bad action at top1.

Under the constraint of not adding loss, not changing candidates, and not using hybrid scoring, this route is exhausted for promotion.

## Verification

Passed:

```bash
python -m py_compile scripts/evaluate_q_calibration.py
git diff --check -- scripts/evaluate_q_calibration.py
python -m json.tool autoresearch/q-calibration-260630/handoff.json
```

