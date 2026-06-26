# Predictive Auto-Research Summary

generated_at: 2026-06-26T05:41:28
objective: `hard_f1` (`hard_macro_f1_tuned`)

## Best Variant

- variant_id: `seed2026__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p0__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0`
- objective_value: `0.473413688418648`
- predictive_score: `0.6368367663057976`
- best_epoch: `3`
- hard_macro_f1_tuned: `0.473413688418648`
- hard_macro_f1_at_0p5: `0.16951675436167268`
- hard_recall_at_top_10pct: `0.9869791666666669`
- ece_sa0: `0.04057633584601374`
- ece_sa1: `0.2280683774714007`
- temperature_scaled_ece_sa0: `0.003800363403545933`
- temperature_scaled_ece_sa1: `0.13795673542257936`
- hard_reduction_score: `0.9537415770055304`

## Compared With Base Defaults

- lambda_hard: base=`0.7` best=`0.5`
- lambda_hard_count: base=`0.1` best=`0.1`
- lambda_hard_reduction: base=`0.5` best=`0.5`
- lambda_hard_rank: base=`` best=`0.0`
- lambda_hard_brier: base=`` best=`0.0`
- lambda_hard_soft_f1: base=`` best=`0.0`
- encoder_type: base=`` best=`mean`
- summary_mode: base=`` best=`global`
- hard_pos_weight_max: base=`20.0` best=`20`
- hard_negative_sample_ratio: base=`5` best=`5`
- hard_loss: base=`asl` best=`asl`
- hard_asl_gamma_neg: base=`4.0` best=`2.0`
- hard_asl_clip: base=`0.05` best=`0.05`
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

- `seed2026__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p02__lhsf0p0__encmean__sumglobal__hlasl__agn2p0__ac0p02__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.43656015037593987 predictive_score=0.6184760887898723
- `seed2026__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p02__lhsf0p0__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.35967479674796754 predictive_score=0.5763987123254927
- `seed2026__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p02__lhsf0p0__encmean__sumglobal__hlasl__agn3p0__ac0p02__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.45543271063667556 predictive_score=0.6268736096140776
- `seed2026__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p02__lhsf0p0__encmean__sumglobal__hlasl__agn3p0__ac0p05__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.34415622170724214 predictive_score=0.5549181298171691
- `seed2026__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p02__lhsf0p0__encmean__sumglobal__hlasl__agn4p0__ac0p02__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.32702052566200684 predictive_score=0.4947548690300024
- `seed2026__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p02__lhsf0p0__encmean__sumglobal__hlasl__agn4p0__ac0p05__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.40807421935256394 predictive_score=0.5901483587149988
- `seed2026__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p0__encmean__sumglobal__hlasl__agn2p0__ac0p02__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.30372545632521447 predictive_score=0.4838677258153444
- `seed2026__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p0__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.473413688418648 predictive_score=0.6368367663057976
- `seed2026__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p0__encmean__sumglobal__hlasl__agn3p0__ac0p02__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.3573403394076405 predictive_score=0.50209004482362
- `seed2026__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p0__encmean__sumglobal__hlasl__agn3p0__ac0p05__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.3103078344309682 predictive_score=0.554521085525463
- `seed2026__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p0__encmean__sumglobal__hlasl__agn4p0__ac0p02__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.26373886474157593 predictive_score=0.3988495882905294
- `seed2026__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p0__encmean__sumglobal__hlasl__agn4p0__ac0p05__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.2837301587301587 predictive_score=0.5576914703910629
