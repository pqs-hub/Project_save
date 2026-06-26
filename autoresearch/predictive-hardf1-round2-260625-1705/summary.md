# Predictive Auto-Research Summary

generated_at: 2026-06-25T18:49:20
objective: `hard_f1` (`hard_macro_f1_tuned`)

## Best Variant

- variant_id: `lh0p5__lhc0p12__lhr0p5__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0`
- objective_value: `0.48224195338512765`
- predictive_score: `0.6293502983643712`
- best_epoch: `2`
- hard_macro_f1_tuned: `0.48224195338512765`
- hard_recall_at_top_10pct: `0.982096354166667`
- hard_reduction_score: `0.8639040967682377`

## Compared With Base Defaults

- lambda_hard: base=`0.7` best=`0.5`
- lambda_hard_count: base=`0.1` best=`0.12`
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

- `lh0p5__lhc0p1__lhr0p5__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.3507248269557268 predictive_score=0.5742973586797855
- `lh0p5__lhc0p12__lhr0p5__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.48224195338512765 predictive_score=0.6293502983643712
- `lh0p5__lhc0p08__lhr0p5__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.35588836761698545 predictive_score=0.5739609434602438
- `lh0p45__lhc0p1__lhr0p5__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.24932892656990951 predictive_score=0.5167396405490532
- `lh0p5__lhc0p15__lhr0p5__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.3908611526324531 predictive_score=0.5267332736115642
- `lh0p55__lhc0p1__lhr0p5__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.39014614792923363 predictive_score=0.5924967821561519
- `lh0p45__lhc0p12__lhr0p5__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.3891397690257625 predictive_score=0.5945279832512567
- `lh0p45__lhc0p08__lhr0p5__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.3490799716914367 predictive_score=0.5747076098758516
- `lh0p55__lhc0p12__lhr0p5__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.4493434742053527 predictive_score=0.5464420861826326
- `lh0p55__lhc0p08__lhr0p5__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.4127373489597529 predictive_score=0.5245793708996623
- `lh0p45__lhc0p15__lhr0p5__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.37222994816594124 predictive_score=0.5108777997605894
- `lh0p5__lhc0p1__lhr0p5__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p5__fc0p0` status=ok hard_macro_f1_tuned=0.32643514626312276 predictive_score=0.5025405618772281
- `lh0p5__lhc0p1__lhr0p5__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p7__fc0p0` status=ok hard_macro_f1_tuned=0.4059159337873879 predictive_score=0.5520912812398284
- `lh0p6__lhc0p1__lhr0p5__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.41137700019278967 predictive_score=0.6008899884189555
- `lh0p55__lhc0p15__lhr0p5__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.44950731561660195 predictive_score=0.60704603356545
- `lh0p5__lhc0p12__lhr0p5__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p5__fc0p0` status=ok hard_macro_f1_tuned=0.275593949269701 predictive_score=0.45838994630364316
