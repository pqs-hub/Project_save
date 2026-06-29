# Small-Circuit Oracle Probe Evals

Question: is the model weak on eval circuits because training subcircuits are much smaller?

## Data

- small subckt probe: `288` finite oracle actions, positive/zero/negative = `268`/`0`/`20`
- eval big probe: `288` finite oracle actions, positive/zero/negative = `190`/`0`/`98`

## Score Summary

| dataset | score_field | mean Spearman | mean Kendall | mean top1 delta | mean top1 regret | negative top1 rate | sign accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| small_subckt | hard_reduction_total_pred | 0.074829 | 0.083498 | 0.036942 | 0.081529 | 0.083333 | 0.930556 |
| small_subckt | reward_pred | 0.058119 | 0.037029 | 0.049432 | 0.069040 | 0.083333 | 0.930556 |
| small_subckt | hybrid_pred | 0.048524 | 0.057185 | 0.036942 | 0.081529 | 0.083333 | 0.930556 |
| small_subckt | guarded_reward | -0.109608 | -0.088592 | 0.025165 | 0.093307 | 0.000000 | 0.930556 |
| eval_big | hybrid_pred | 0.327398 | 0.294605 | 0.010638 | 0.012552 | 0.166667 | 0.197917 |
| eval_big | hard_reduction_total_pred | 0.324443 | 0.291166 | 0.010638 | 0.012552 | 0.166667 | 0.197917 |
| eval_big | guarded_reward | 0.310535 | 0.297954 | 0.002967 | 0.020223 | 0.500000 | 0.659722 |
| eval_big | reward_pred | 0.294742 | 0.283848 | 0.002967 | 0.020223 | 0.500000 | 0.659722 |

## Benchmark Summary

| dataset | benchmark | actions | positive | negative | mean delta | max delta | min delta |
|---|---|---:|---:|---:|---:|---:|---:|
| small_subckt | subckt_0072 | 72 | 72 | 0 | 0.044583 | 0.242650 | 0.007350 |
| small_subckt | subckt_0304 | 72 | 72 | 0 | 0.014112 | 0.036360 | 0.000650 |
| small_subckt | subckt_0335 | 72 | 72 | 0 | 0.082473 | 0.337500 | 0.015460 |
| small_subckt | subckt_0350 | 72 | 52 | 20 | 0.001656 | 0.019230 | -0.018510 |
| eval_big | b15_C | 144 | 144 | 0 | 0.023183 | 0.068550 | 0.000030 |
| eval_big | i2c_aig | 144 | 46 | 98 | -0.018459 | 0.004750 | -0.043750 |

## Interpretation

- Best small-circuit score field: `hard_reduction_total_pred`, mean Spearman `0.0748`, negative top1 rate `0.0833`.
- Best eval-circuit score field: `hybrid_pred`, mean Spearman `0.3274`, negative top1 rate `0.1667`.
- This does not support the simple size-mismatch hypothesis; alignment is not better on small subcircuits in this probe.
- Treat this as a first probe: subcircuits are from training distribution, so strong results here would indicate distribution/scale shift, but weak results here would indicate a more basic score-target mismatch.
