# Predictive Auto-Research Summary

generated_at: 2026-06-26T10:17:38
objective: `hard_f1` (`hard_macro_f1_tuned`)

## Best Variant

- variant_id: `seed2026__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0`
- objective_value: `0.5519422451961875`
- predictive_score: `0.6704069536126113`
- best_epoch: `3`
- hard_macro_f1_tuned: `0.5519422451961875`
- hard_macro_f1_at_0p5: `0.0990835296313595`
- hard_recall_at_top_10pct: `0.976236979166667`
- ece_sa0: `0.020064549634747977`
- ece_sa1: `0.2499074042225402`
- temperature_scaled_ece_sa0: `0.002885854497368663`
- temperature_scaled_ece_sa1: `0.22067677696361854`
- hard_reduction_score: `0.9389457758552453`

## Compared With Base Defaults

- lambda_hard: base=`0.7` best=`0.5`
- lambda_hard_count: base=`0.1` best=`0.1`
- lambda_hard_reduction: base=`0.5` best=`0.5`
- lambda_hard_rank: base=`` best=`0.0`
- lambda_hard_brier: base=`` best=`0.0`
- lambda_hard_soft_f1: base=`` best=`0.02`
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

- `seed2026__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p02__lhsf0p0__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.26940227782858095 predictive_score=0.5409532100510481
- `seed2026__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p0__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.44712132947427063 predictive_score=0.6244435463694646
- `seed2026__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p02__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.26954452753229097 predictive_score=0.5376607459265882
- `seed2026__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p05__lhsf0p0__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.2993984679265477 predictive_score=0.5012626399742863
- `seed2026__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.5519422451961875 predictive_score=0.6704069536126113
- `seed2026__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p05__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.36336904011762233 predictive_score=0.5919672755066603
- `seed2026__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p1__lhsf0p0__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.4105050505050505 predictive_score=0.5737172626185831
- `seed2026__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p1__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.403740779180198 predictive_score=0.6071707588981684
