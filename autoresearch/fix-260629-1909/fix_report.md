# Fix Report: candidate cache missing in negative-rich oracle runners

## Error

User command:

```bash
./autoresearch/oracle-negative-rich-260629/run_topup.sh
```

failed with:

```text
ValueError: cached candidate strategies require --candidate-cache-dir
```

## Root Cause

The runner used:

```text
cached_hard_cone,cached_random,cached_stride
```

but did not pass:

```text
--candidate-cache-dir
```

Also, existing candidate caches only covered old labeled subckts. The new pilot/topup samples are mostly fresh subckts, so a cache directory for the new negative-rich pool must be built first.

## Fix

Added:

```text
scripts/list_missing_candidate_cache.py
```

Updated:

```text
autoresearch/oracle-negative-rich-260629/run_pilot.sh
autoresearch/oracle-negative-rich-260629/run_topup.sh
```

Both runners now:

```text
1. Use autoresearch/tp-candidates-negative-rich-260629 as candidate cache dir.
2. Check which subckts are missing cache JSONs.
3. Build only missing cache files with scripts/build_hard_tp_candidate_cache.py.
4. Pass --candidate-cache-dir into scripts/oracle_action_value_probe.py.
```

## Verification

Passed:

```bash
python -m py_compile scripts/list_missing_candidate_cache.py scripts/build_hard_tp_candidate_cache.py scripts/oracle_action_value_probe.py
bash -n autoresearch/oracle-negative-rich-260629/run_pilot.sh autoresearch/oracle-negative-rich-260629/run_topup.sh
```

Smoke test:

```bash
python scripts/build_hard_tp_candidate_cache.py --benchmarks subckt_0057 --out-dir autoresearch/tp-candidates-negative-rich-260629
python scripts/oracle_action_value_probe.py ... --benchmarks subckt_0057 --candidate-cache-dir autoresearch/tp-candidates-negative-rich-260629 --candidate-strategies cached_hard_cone,cached_random,cached_stride --dry-run
```

Result:

```text
subckt_0057 cache built
dry-run oracle probe completed
evaluated_oracle_actions = 3
status = progressed
```

## Current Cache State

```text
pilot cached: 0 / 96
topup cached: 1 / 192
```

The next topup run will automatically build the remaining 191 topup caches before backend oracle evaluation.

## Next Command

Retry the same command:

```bash
./autoresearch/oracle-negative-rich-260629/run_topup.sh
```

