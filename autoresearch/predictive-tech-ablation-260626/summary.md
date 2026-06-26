# Predictive Auto-Research Summary

generated_at: 2026-06-26T03:04:55
objective: `hard_f1` (`hard_macro_f1_tuned`)

## Best Variant

- variant_id: `seed2026__lh0p5__lhc0p1__lhr0p5__lhrk0p05__encmean__sumcone__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0`
- objective_value: `0.39183542269366767`
- predictive_score: `0.5344592527785641`
- best_epoch: `2`
- hard_macro_f1_tuned: `0.39183542269366767`
- hard_recall_at_top_10pct: `0.7402343750000022`
- hard_reduction_score: `0.924074083501182`

## Compared With Base Defaults

- lambda_hard: base=`0.7` best=`0.5`
- lambda_hard_count: base=`0.1` best=`0.1`
- lambda_hard_reduction: base=`0.5` best=`0.5`
- lambda_hard_rank: base=`` best=`0.05`
- encoder_type: base=`` best=`mean`
- summary_mode: base=`` best=`cone`
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

- `seed2026__lh0p5__lhc0p1__lhr0p5__lhrk0p03__encgate_dir__sumcone__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.24385177315200068 predictive_score=0.5329963203653529
- `seed2026__lh0p5__lhc0p1__lhr0p5__lhrk0p05__encgate_dir__sumcone__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.22802413977060548 predictive_score=0.5198390669416554
- `seed2026__lh0p5__lhc0p1__lhr0p5__lhrk0p0__encgate_dir__sumcone__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.18197143110637917 predictive_score=0.4544041360197871
- `seed2026__lh0p5__lhc0p1__lhr0p5__lhrk0p03__encgate_dir__sumglobal__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.2154848566883114 predictive_score=0.4365213897259847
- `seed2026__lh0p5__lhc0p1__lhr0p5__lhrk0p05__encgate_dir__sumglobal__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.19404078531632138 predictive_score=0.5006302342349708
- `seed2026__lh0p5__lhc0p1__lhr0p5__lhrk0p0__encgate_dir__sumglobal__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.30132849619178165 predictive_score=0.5419242243237993
- `seed2026__lh0p5__lhc0p1__lhr0p5__lhrk0p03__encmean__sumcone__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.30812630812630815 predictive_score=0.49682819434907355
- `seed2026__lh0p5__lhc0p1__lhr0p5__lhrk0p05__encmean__sumcone__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.39183542269366767 predictive_score=0.5344592527785641
- `seed2026__lh0p5__lhc0p1__lhr0p5__lhrk0p0__encmean__sumcone__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.3899215336700893 predictive_score=0.5748017626306585
- `seed2026__lh0p5__lhc0p1__lhr0p5__lhrk0p03__encmean__sumglobal__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.3210755843573598 predictive_score=0.5613614667238507
- `seed2026__lh0p5__lhc0p1__lhr0p5__lhrk0p05__encmean__sumglobal__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.37481623521891977 predictive_score=0.5169836998920474
- `seed2026__lh0p5__lhc0p1__lhr0p5__lhrk0p0__encmean__sumglobal__hlasl__hhresidual_context__pw20__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0` status=ok hard_macro_f1_tuned=0.38721344100424815 predictive_score=0.5502042086241932
