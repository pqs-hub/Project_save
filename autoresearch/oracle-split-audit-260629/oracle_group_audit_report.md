# Oracle Action Group Audit

generated_at: `2026-06-29T18:47:01`

## Split Summary

| split | rows | groups | negative rate | all-positive groups | mean delta | min delta | max delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 5184 | 288 | 0.1381 | 168 | 0.002984 | -0.072110 | 0.247380 |
| expanded_val | 864 | 48 | 0.4155 | 11 | -0.004885 | -0.257850 | 0.160490 |
| transfer | 288 | 6 | 0.3403 | 3 | 0.002362 | -0.043750 | 0.068550 |

## Action Type Summary

| split | type | rows | negative rate | mean delta |
|---|---|---:|---:|---:|
| train | `control0` | 1728 | 0.2089 | 0.002446 |
| train | `control1` | 1728 | 0.1944 | 0.002478 |
| train | `observe` | 1728 | 0.0110 | 0.004027 |
| expanded_val | `control0` | 288 | 0.6389 | -0.011869 |
| expanded_val | `control1` | 288 | 0.5799 | -0.007731 |
| expanded_val | `observe` | 288 | 0.0278 | 0.004946 |
| transfer | `control0` | 96 | 0.5000 | 0.000821 |
| transfer | `control1` | 96 | 0.5000 | 0.001797 |
| transfer | `observe` | 96 | 0.0208 | 0.004470 |

## Group Negative Count Histogram

| split | negative actions in group | groups |
|---|---:|---:|
| train | 0 | 168 |
| train | 1 | 22 |
| train | 2 | 27 |
| train | 3 | 7 |
| train | 4 | 5 |
| train | 5 | 3 |
| train | 6 | 4 |
| train | 7 | 3 |
| train | 8 | 3 |
| train | 9 | 6 |
| train | 10 | 6 |
| train | 11 | 11 |
| train | 12 | 19 |
| train | 13 | 4 |
| expanded_val | 0 | 11 |
| expanded_val | 1 | 2 |
| expanded_val | 2 | 1 |
| expanded_val | 3 | 1 |
| expanded_val | 4 | 1 |
| expanded_val | 6 | 1 |
| expanded_val | 7 | 1 |
| expanded_val | 8 | 2 |
| expanded_val | 9 | 4 |
| expanded_val | 10 | 2 |
| expanded_val | 11 | 5 |
| expanded_val | 12 | 13 |
| expanded_val | 13 | 4 |
| transfer | 0 | 3 |
| transfer | 32 | 2 |
| transfer | 34 | 1 |

## Ranker-Worsened Groups

| split | benchmark | strategy | baseline top1 | ranker top1 | delta vs baseline |
|---|---|---|---:|---:|---:|
| train | `subckt_0081` | `cached_random` | 0.048030 | 0.001460 | -0.046570 |
| train | `subckt_0297` | `cached_random` | 0.058910 | 0.015240 | -0.043670 |
| train | `subckt_0072` | `cached_stride` | 0.041040 | 0.007350 | -0.033690 |
| train | `subckt_0327` | `cached_random` | 0.038920 | 0.006760 | -0.032160 |
| train | `subckt_0063` | `cached_hard_cone` | 0.032080 | 0.002490 | -0.029590 |
| train | `subckt_0063` | `cached_random` | 0.032080 | 0.002490 | -0.029590 |
| expanded_val | `subckt_0045` | `cached_random` | 0.032080 | 0.002490 | -0.029590 |
| expanded_val | `subckt_0217` | `cached_hard_cone` | 0.029840 | 0.001410 | -0.028430 |
| transfer | `i2c_aig` | `cached_hard_cone` | 0.000060 | -0.027800 | -0.027860 |
| transfer | `i2c_aig` | `cached_stride` | 0.000060 | -0.027800 | -0.027860 |
| train | `subckt_0260` | `cached_random` | 0.028880 | 0.003660 | -0.025220 |
| train | `subckt_0243` | `cached_hard_cone` | 0.024820 | 0.000860 | -0.023960 |
| expanded_val | `subckt_0217` | `cached_random` | 0.029840 | 0.006220 | -0.023620 |
| train | `subckt_0384` | `cached_stride` | -0.003020 | -0.024860 | -0.021840 |
| train | `subckt_0261` | `cached_stride` | 0.020000 | 0.000520 | -0.019480 |
| train | `subckt_0016` | `cached_random` | 0.016360 | 0.000320 | -0.016040 |
| train | `subckt_0230` | `cached_random` | 0.000040 | -0.015740 | -0.015780 |
| train | `subckt_0230` | `cached_hard_cone` | 0.000040 | -0.015260 | -0.015300 |
| train | `subckt_0230` | `cached_stride` | 0.000040 | -0.015260 | -0.015300 |
| train | `subckt_0011` | `cached_stride` | 0.017030 | 0.002190 | -0.014840 |

## Recommendation

- Balance by group, not only by rows.
- Require both positive and negative actions inside a group for rank training.
- Prefer negative `control0` / `control1` examples because transfer failures concentrate there.
- Keep transfer evaluation-only; do not train on `b15_C` or `i2c_aig` transfer rows.
