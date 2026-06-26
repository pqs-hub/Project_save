# Predictive Auto-Research Summary

generated_at: 2026-06-25T21:14:18
objective: `hard_f1` (`hard_macro_f1_tuned`)

## Best Variant

- variant_id: `lh0p5__lhc0p12__lhr0p5__lhrk0p05__encmean__sumglobal__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0`
- objective_value: `0.4119663070499361`
- predictive_score: `0.5255169820851682`
- best_epoch: `4`
- hard_macro_f1_tuned: `0.4119663070499361`
- hard_recall_at_top_10pct: `0.659830729166668`
- hard_reduction_score: `0.9394549622666091`

## Compared With Base Defaults

- lambda_hard: base=`0.7` best=`0.5`
- lambda_hard_count: base=`0.1` best=`0.12`
- lambda_hard_reduction: base=`0.5` best=`0.5`
- lambda_hard_rank: base=`` best=`0.05`
- encoder_type: base=`` best=`mean`
- summary_mode: base=`` best=`global`
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

- Center the next grid around the best encoder, summary, hard rank weight, hard loss, and hard-negative mining mode.
- Keep `lambda_fc=0.0` until hard-fault predictive metrics plateau.
- Add ranking or pairwise calibration loss only after ASL/focal and top-k mining plateau.

## Completed Variants

- `lh0p5__lhc0p12__lhr0p5__lhrk0p05__encgate_dir__sumcone__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.3720184204055172 predictive_score=0.5907531769117108
- `lh0p5__lhc0p12__lhr0p5__lhrk0p0__encgate_dir__sumcone__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.2617743806209952 predictive_score=0.5210266724160861
- `lh0p5__lhc0p12__lhr0p5__lhrk0p1__encgate_dir__sumcone__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.1889013880414887 predictive_score=0.4974553683253265
- `lh0p5__lhc0p12__lhr0p5__lhrk0p05__encgate_dir__sumglobal__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.18939857597333737 predictive_score=0.4959514595257251
- `lh0p5__lhc0p12__lhr0p5__lhrk0p0__encgate_dir__sumglobal__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.22498494612635003 predictive_score=0.5147531027916978
- `lh0p5__lhc0p12__lhr0p5__lhrk0p1__encgate_dir__sumglobal__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.24137809077927647 predictive_score=0.43716477340428456
- `lh0p5__lhc0p12__lhr0p5__lhrk0p05__encmean__sumcone__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.36671230973785 predictive_score=0.5208412071049866
- `lh0p5__lhc0p12__lhr0p5__lhrk0p0__encmean__sumcone__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.24271612057889408 predictive_score=0.4543137309662519
- `lh0p5__lhc0p12__lhr0p5__lhrk0p1__encmean__sumcone__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.2456916305028112 predictive_score=0.46897028869105156
- `lh0p5__lhc0p12__lhr0p5__lhrk0p05__encmean__sumglobal__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.4119663070499361 predictive_score=0.5255169820851682
- `lh0p5__lhc0p12__lhr0p5__lhrk0p0__encmean__sumglobal__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.3673149766899767 predictive_score=0.5365890201776337
- `lh0p5__lhc0p12__lhr0p5__lhrk0p1__encmean__sumglobal__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.3589840193400521 predictive_score=0.5002228053072634
