# Predictive Auto-Research Summary

generated_at: 2026-06-25T17:00:08
objective: `hard_f1` (`hard_macro_f1_tuned`)

## Best Variant

- variant_id: `lh0p5__lhc0p1__lhr0p5__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0`
- objective_value: `0.5788207260664393`
- predictive_score: `0.6029605889234299`
- best_epoch: `3`
- hard_macro_f1_tuned: `0.5788207260664393`
- hard_recall_at_top_10pct: `0.6710611979166687`
- hard_reduction_score: `0.9480104100002791`

## Compared With Base Defaults

- lambda_hard: base=`0.7` best=`0.5`
- lambda_hard_count: base=`0.1` best=`0.1`
- lambda_hard_reduction: base=`0.5` best=`0.5`
- hard_pos_weight_max: base=`20.0` best=`20`
- hard_negative_sample_ratio: base=`5` best=`5`
- hard_loss: base=`asl` best=`asl`
- hard_head_type: base=`residual_context` best=`residual_context`
- hard_negative_mining: base=`topk` best=`topk`
- train_sample_strategy: base=`hard_weighted` best=`hard_weighted`
- feature_mode: base=`testability` best=`testability`
- edge_weight_mode: base=`fault_path` best=`fault_path`
- edge_keep_ratio: base=`0.6` best=`0.6`
- lambda_fc: base=`0.0` best=`0.0`

## Trend Notes

- This run selected the best variant by `hard_macro_f1_tuned`.
- Keep `predictive_score` as a secondary guardrail so F1 improvements do not destroy ranking and reduction quality.
- If train loss improves but hard F1 stalls, compare ASL against focal and switch negative mining between top-k and mixed.
- If hard count overlap stays low, move to the second-stage hard-count calibration change.

## Suggested Next Round

- Center the next grid around the best hard loss, head type, negative mining mode, `lambda_hard`, and `hard_negative_sample_ratio`.
- Keep `lambda_fc=0.0` until hard-fault predictive metrics plateau.
- Add ranking or pairwise calibration loss only after ASL/focal and top-k mining plateau.

## Completed Variants

- `lh0p7__lhc0p1__lhr0p5__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.36080888009671397 predictive_score=0.5869074080849154
- `lh0p7__lhc0p05__lhr0p5__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.3186344238975818 predictive_score=0.4721973653957995
- `lh0p7__lhc0p0__lhr0p5__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.408646850966535 predictive_score=0.5903542143878917
- `lh0p7__lhc0p2__lhr0p5__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.37253556456402775 predictive_score=0.5992015834184715
- `lh0p7__lhc0p1__lhr0p5__hlasl__hhresidual_context__pw20__ns5__nmmixed__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.3718462464528334 predictive_score=0.5748061131246165
- `lh0p5__lhc0p1__lhr0p5__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.5788207260664393 predictive_score=0.6029605889234299
- `lh0p7__lhc0p05__lhr0p5__hlasl__hhresidual_context__pw20__ns5__nmmixed__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.3706528765352295 predictive_score=0.588854071304044
- `lh0p7__lhc0p1__lhr0p5__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p8__fc0p0` status=ok hard_macro_f1_tuned=0.3993346893804559 predictive_score=0.531400718429491
- `lh0p5__lhc0p05__lhr0p5__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.36780366933783215 predictive_score=0.5519805457141851
- `lh0p7__lhc0p0__lhr0p5__hlasl__hhresidual_context__pw20__ns5__nmmixed__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.3973917473255222 predictive_score=0.6073267976550835
- `lh0p7__lhc0p1__lhr0p5__hlfocal__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.38625872249060655 predictive_score=0.5871191121230906
- `lh0p7__lhc0p2__lhr0p5__hlasl__hhresidual_context__pw20__ns5__nmmixed__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.4058742033804378 predictive_score=0.6063425333610989
