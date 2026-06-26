# Predictive Auto-Research Summary

generated_at: 2026-06-25T12:06:38

## Best Variant

- variant_id: `lh0p7__lhc0p0__lhr0p5__pw20__ns5__fmtestability__ewfault_path__ek0p6__fc0p0`
- predictive_score: `0.37034302246914347`
- best_epoch: `1`
- hard_macro_f1_tuned: `0.011673151750972763`
- hard_recall_at_top_10pct: `0.6041666666666666`
- hard_reduction_score: `0.8731813649646938`

## Compared With Base Defaults

- lambda_hard: base=`0.7` best=`0.7`
- lambda_hard_count: base=`0.1` best=`0.0`
- lambda_hard_reduction: base=`0.5` best=`0.5`
- hard_pos_weight_max: base=`20.0` best=`20`
- hard_negative_sample_ratio: base=`` best=`5`
- feature_mode: base=`testability` best=`testability`
- edge_weight_mode: base=`fault_path` best=`fault_path`
- edge_keep_ratio: base=`0.6` best=`0.6`
- lambda_fc: base=`0.0` best=`0.0`

## Trend Notes

- Prefer variants with higher `predictive_score`, then `hard_recall_at_top_10pct`, then `hard_macro_f1_tuned`.
- If train loss improves but predictive score stalls, widen hard negative sampling and threshold search before changing architecture.
- If hard count overlap stays low, move to the second-stage hard-count calibration change.

## Suggested Next Round

- Center the next grid around the best `lambda_hard`, `lambda_hard_reduction`, and `hard_negative_sample_ratio`.
- Keep `lambda_fc=0.0` until hard-fault predictive metrics plateau.
- Add code-level focal/ranking loss only after two config-only rounds fail to improve.

## Completed Variants

- `lh0p7__lhc0p1__lhr0p5__pw20__ns5__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok predictive_score=0.3441549585897178
- `lh0p7__lhc0p05__lhr0p5__pw20__ns5__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok predictive_score=0.3489555745687898
- `lh0p7__lhc0p0__lhr0p5__pw20__ns5__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok predictive_score=0.37034302246914347
