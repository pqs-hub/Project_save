# Oracle Probe Evals Report

source: `autoresearch/oracle-action-probe-260629-smoke-real/`

## Data Coverage

- oracle action rows: `1`
- prediction metric rows: `2`
- rank metric rows: `2`
- state summary rows: `2`

This is a micro-smoke, not a statistically meaningful evaluation. Correlation metrics are `nan` because there is only one evaluated candidate action.

## Findings

- Found positive model score on a real negative-TC action.
- action: `__buf_split_1::control0`
- benchmark/state: `b15_C` / `initial`
- oracle_delta_tc: `-0.000290000000000068`
- reward_pred: `0.2648739218711853`
- guarded_reward: `0.2239551991224289`

## Metric Interpretation

- `oracle_action_recall@1=1.0` in this run is not meaningful because the evaluated set contains exactly one action.
- `top1_regret=0.0` is also degenerate for the same reason.
- `negative_top1=1` is meaningful as a smoke-level red flag: the only predicted top action had negative real TC delta.
- Spearman/Kendall/Pearson are `nan` until at least two finite oracle actions exist; useful correlation needs dozens of actions per benchmark/state.

## Verdict

The probe implementation is functional, but current evidence is insufficient for model-quality conclusions. The next evaluation must run the planned b15_C+i2c_aig smoke with `max_nets=16`, all 3 action types, and real backend evaluation.

## Next Command

```bash
TPI_BENCH_ROOT=/data4/pengqingsong/DFT/Dataset/deeptpi_official_aig_bench_standard \
python scripts/oracle_action_value_probe.py \
  --checkpoint autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt \
  --benchmarks b15_C,i2c_aig \
  --candidate-cache-dir autoresearch/tp-candidates-260626-2047 \
  --candidate-strategies cached_stride,cached_hard_cone,hard_fault_recall_union \
  --score-fields reward_pred,guarded_reward,hard_reduction_total_pred,hybrid_pred \
  --states initial \
  --max-nets 16 \
  --action-types CP0,CP1,OP \
  --patterns 10000 \
  --seed 2026 \
  --backend atalanta-bist \
  --plan-device cuda \
  --timeout-sec 14400 \
  --out-dir autoresearch/oracle-action-probe-260629-smoke
```

## Small Expanded Probe

source: `autoresearch/oracle-action-probe-260629-evals-small/`

- oracle action rows: `4`
- prediction metric rows: `4`
- rank metric rows: `12`

| score_field | spearman | kendall | pearson | top1_real_delta_tc | top1_regret | negative_top1 | sign_accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| reward_pred | 0.9486832980505138 | 1.0 | 0.8276071244285853 | -0.0028000000000000247 | 0.0 | 1 | 0.0 |
| guarded_reward | 0.9486832980505138 | 1.0 | 0.8586333834633046 | -0.0028000000000000247 | 0.0 | 1 | 0.0 |
| hard_reduction_total_pred | 0.9486832980505138 | 1.0 | 0.7820371318923349 | -0.0028000000000000247 | 0.0 | 1 | 1.0 |
| hybrid_pred | 0.9486832980505138 | 1.0 | 0.7833908618696958 | -0.0028000000000000247 | 0.0 | 1 | 1.0 |

Oracle action values:

| action | delta_tc | reward_pred | guarded_reward | hard_reduction_total_pred | hybrid_pred |
|---|---:|---:|---:|---:|---:|
| `N9361::control0` | -0.0028000000000000247 | 0.30438345670700073 | 0.24945855140686035 | -0.03621556982398033 | -3.067714974284172 |
| `N9361::control1` | -0.0028000000000000247 | 0.3047252297401428 | 0.2479255646467209 | -0.037057217210531235 | -3.1530709266662598 |
| `__buf_split_1::control1` | -0.002840000000000009 | 0.2781575322151184 | 0.23377731442451477 | -0.05738551542162895 | -5.226616695523262 |
| `__buf_split_1::control0` | -0.003269999999999995 | 0.2648739218711853 | 0.2239551991224289 | -0.06434640288352966 | -5.945811167359352 |

Best score field by this tiny probe: `reward_pred` with Spearman `0.9486832980505138` and top1 real delta `-0.0028000000000000247`.

Caution: this is still a tiny 4-action probe on one circuit. It can identify obvious sign/ranking failures, but it is not enough for final planner selection.

## Full Planned Smoke

source: `autoresearch/oracle-action-probe-260629-smoke/`

- oracle action rows: `288`
- finite oracle rows: `288`
- prediction metric rows: `24`
- rank metric rows: `72`
- positive / zero / negative oracle delta counts: `190` / `0` / `98`

### Score Field Summary

| score_field | groups | mean Spearman | mean Kendall | mean top1 delta | mean top1 regret | negative top1 rate | mean sign accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| hybrid_pred | 6 | 0.327398 | 0.294605 | 0.010638 | 0.012552 | 0.166667 | 0.197917 |
| hard_reduction_total_pred | 6 | 0.324443 | 0.291166 | 0.010638 | 0.012552 | 0.166667 | 0.197917 |
| guarded_reward | 6 | 0.310535 | 0.297954 | 0.002967 | 0.020223 | 0.500000 | 0.659722 |
| reward_pred | 6 | 0.294742 | 0.283848 | 0.002967 | 0.020223 | 0.500000 | 0.659722 |

### Best Oracle Actions

| benchmark | strategy | action | delta_tc | reward_pred | guarded_reward | hard_reduction_total_pred | hybrid_pred |
|---|---|---|---:|---:|---:|---:|---:|
| b15_C | hard_fault_recall_union | `N6666::observe` | 0.06855000000000011 | 0.23240526020526886 | 0.2100413292646408 | -0.06991252303123474 | -6.5488057136535645 |
| b15_C | hard_fault_recall_union | `N6538::observe` | 0.06854000000000005 | 0.3106147050857544 | 0.27014851570129395 | -0.03383130952715874 | -2.8023677319288254 |
| b15_C | hard_fault_recall_union | `N15715::observe` | 0.060020000000000184 | 0.26371830701828003 | 0.22037114202976227 | -0.056784216314554214 | -5.194332182407379 |
| b15_C | hard_fault_recall_union | `N10901::observe` | 0.051490000000000036 | 0.23229840397834778 | 0.21005623042583466 | -0.0698917806148529 | -6.546823427081108 |
| b15_C | hard_fault_recall_union | `N10902::observe` | 0.050550000000000095 | 0.26142311096191406 | 0.21680273115634918 | -0.056901659816503525 | -5.211940139532089 |
| b15_C | cached_stride | `__buf_split_2536::control1` | 0.03266000000000013 | 0.24084042012691498 | 0.20072822272777557 | -0.06942112743854523 | -6.500544100999832 |
| b15_C | hard_fault_recall_union | `N10901::control0` | 0.030810000000000115 | 0.29878801107406616 | 0.26617372035980225 | -0.0446603037416935 | -3.9010686427354813 |
| b15_C | hard_fault_recall_union | `N10902::control0` | 0.030810000000000115 | 0.2665197253227234 | 0.22667251527309418 | -0.057759229093790054 | -5.282730668783188 |
| b15_C | hard_fault_recall_union | `N6666::control0` | 0.030810000000000115 | 0.2989082932472229 | 0.26627153158187866 | -0.04469836875796318 | -3.9046570509672165 |
| b15_C | hard_fault_recall_union | `N7402::control1` | 0.03068000000000004 | 0.32712072134017944 | 0.27945980429649353 | -0.02205653116106987 | -1.599072590470314 |

### Worst Oracle Actions

| benchmark | strategy | action | delta_tc | reward_pred | guarded_reward | hard_reduction_total_pred | hybrid_pred |
|---|---|---|---:|---:|---:|---:|---:|
| i2c_aig | cached_stride | `N136::control0` | -0.043749999999999956 | 0.24632011353969574 | 0.21925735473632812 | -0.039351385086774826 | -3.4695610404014587 |
| i2c_aig | cached_hard_cone | `N136::control0` | -0.043749999999999956 | 0.24632011353969574 | 0.21925735473632812 | -0.039351385086774826 | -3.4695610404014587 |
| i2c_aig | cached_hard_cone | `N158::control0` | -0.03687000000000018 | 0.3193073272705078 | 0.3193073272705078 | 0.0657663345336914 | 7.269899278879166 |
| i2c_aig | hard_fault_recall_union | `N332::control0` | -0.02968000000000004 | 0.33673036098480225 | 0.33673036098480225 | 0.07395058870315552 | 8.119664579629898 |
| i2c_aig | hard_fault_recall_union | `__buf_split_1101::control1` | -0.02968000000000004 | 0.30317991971969604 | 0.30317991971969604 | 0.03351360186934471 | 3.9911526292562485 |
| i2c_aig | cached_stride | `__buf_split_515::control0` | -0.02905000000000002 | 0.2639951705932617 | 0.2639951705932617 | -0.023750636726617813 | -1.838966116309166 |
| i2c_aig | cached_stride | `N295::control0` | -0.02842 | 0.3327004909515381 | 0.3327004909515381 | 0.07397004961967468 | 8.11621817946434 |
| i2c_aig | hard_fault_recall_union | `N287::control0` | -0.02842 | 0.3323855996131897 | 0.3323855996131897 | 0.07341958582401276 | 8.062572538852692 |
| i2c_aig | hard_fault_recall_union | `__buf_split_1101::control0` | -0.02842 | 0.32212358713150024 | 0.32212358713150024 | 0.06037476286292076 | 6.730563119053841 |
| i2c_aig | hard_fault_recall_union | `N289::control1` | -0.02842 | 0.32988685369491577 | 0.32988685369491577 | 0.07373397052288055 | 8.0854711830616 |

### Evals Verdict

Best average rank signal in this smoke: `hybrid_pred` with mean Spearman `0.3274`.

The important failure mode is sign/calibration: many predicted top-1 choices are still negative real-TC actions even when rank correlation is positive. Treat the world model as a relative ranking signal only after adding a safety/calibration layer.
