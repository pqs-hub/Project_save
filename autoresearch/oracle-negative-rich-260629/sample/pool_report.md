# Negative-Rich Oracle Subckt Sample

generated_at: `2026-06-29T19:01:23`

## Summary

| item | count |
|---|---:|
| all subckt bench files | 400 |
| excluded subckts | 16 |
| existing train subckts | 96 |
| eligible train-pool subckts | 384 |
| fresh eligible subckts | 288 |
| pilot subckts | 96 |
| topup subckts | 192 |
| remaining after pilot/topup | 96 |

## Policy

- Keep expanded validation subckts excluded from training collection.
- Prefer subckts that do not already have train oracle labels.
- Keep `all_eligible_subckts.txt` so collection can be expanded later instead of stopping at a small fixed target.

## Files

- `pilot_subckts.txt`: first backend batch.
- `topup_subckts.txt`: second backend batch if pilot is insufficient.
- `remaining_subckts.txt`: additional eligible subckts after pilot/topup.
- `all_eligible_subckts.txt`: all non-validation subckts available for train oracle collection.
