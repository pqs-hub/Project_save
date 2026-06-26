# Predictive Auto-Research Summary

generated_at: 2026-06-25T13:19:01

## Best Variant

- variant_id: `lh0p7__lhc0p1__lhr0p5__pw20__ns5__fmtestability__ewfault_path__ek0p8__fc0p0`
- predictive_score: `0.6125736615004326`
- best_epoch: `2`
- hard_macro_f1_tuned: `0.0707937810070706`
- hard_recall_at_top_10pct: `0.9720052083333338`
- hard_reduction_score: `0.9274469599840813`

## Compared With Base Defaults

- lambda_hard: base=`0.7` best=`0.7`
- lambda_hard_count: base=`0.1` best=`0.1`
- lambda_hard_reduction: base=`0.5` best=`0.5`
- hard_pos_weight_max: base=`20.0` best=`20`
- hard_negative_sample_ratio: base=`` best=`5`
- feature_mode: base=`testability` best=`testability`
- edge_weight_mode: base=`fault_path` best=`fault_path`
- edge_keep_ratio: base=`0.6` best=`0.8`
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

- `lh0p7__lhc0p1__lhr0p5__pw20__ns5__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok predictive_score=0.5655515901688104
- `lh0p7__lhc0p05__lhr0p5__pw20__ns5__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok predictive_score=0.5738786904305038
- `lh0p7__lhc0p0__lhr0p5__pw20__ns5__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok predictive_score=0.598195593495728
- `lh0p7__lhc0p2__lhr0p5__pw20__ns5__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok predictive_score=0.5808127280977897
- `lh0p5__lhc0p1__lhr0p5__pw20__ns5__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok predictive_score=0.5791018943046808
- `lh0p7__lhc0p1__lhr0p5__pw20__ns5__fmtestability__ewfault_path__ek0p8__fc0p0` status=ok predictive_score=0.6125736615004326
- `lh0p5__lhc0p05__lhr0p5__pw20__ns5__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok predictive_score=0.5793517648907063
- `lh0p7__lhc0p05__lhr0p5__pw20__ns5__fmtestability__ewfault_path__ek0p8__fc0p0` status=ok predictive_score=0.591654264928461
- `lh0p5__lhc0p0__lhr0p5__pw20__ns5__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok predictive_score=0.561292138593577
- `lh0p5__lhc0p2__lhr0p5__pw20__ns5__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok predictive_score=0.5880351571158109
- `lh0p7__lhc0p1__lhr0p2__pw20__ns5__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok predictive_score=0.5853471762258553
- `lh0p7__lhc0p1__lhr0p5__pw20__ns3__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok predictive_score=0.5771721294749916
