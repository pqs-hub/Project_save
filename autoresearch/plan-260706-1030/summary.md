# Plan Summary

This plan converts the current research direction into a bounded autoresearch config.

The loop metric is deliberately non-target: `non_target_dev_best_delta_tc_pct`. The final 8-circuit eval is an acceptance check only and must not be used to choose between variants.

Recommended command:

```text
$autoresearch
Goal: Improve world-model rerank using NDCG/listwise, conservative/uncertainty, candidate-context, or DAG-aware methods without using the 8 target circuits for training, calibration, or model selection.
Scope: tpi_jepa/train.py,tpi_jepa/model.py,tpi_jepa/plan.py,tpi_jepa/protocol.py,configs/planner_aligned_*safe*.json,scripts/*.py,autoresearch/improve-260706-0959/*.sh,autoresearch/improve-260706-0959/*.py
Metric: non_target_dev_best_delta_tc_pct
Direction: higher_is_better
Verify: cd /data4/pengqingsong/DFT/TPI-my.3 && GPUS_CSV=0,1,2,3,4,5,6,7 bash autoresearch/improve-260706-0959/run_dev_non_target_rerank_parallel.sh >/tmp/tpi_dev_non_target_rerank.log 2>&1 && python - <<'PY'
import csv, math
from pathlib import Path
path = Path("autoresearch/improve-260706-0959/dev_non_target_rerank/summary.tsv")
best = None
if path.exists():
    with path.open(newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("status") != "ok":
                continue
            try:
                value = float(row.get("delta_test_coverage", "nan")) * 100.0
            except ValueError:
                continue
            if math.isfinite(value):
                best = value if best is None else max(best, value)
print(f"{best if best is not None else -999.0:.6f}")
PY
Guard: cd /data4/pengqingsong/DFT/TPI-my.3 && python -m py_compile tpi_jepa/protocol.py tpi_jepa/train.py tpi_jepa/model.py tpi_jepa/plan.py autoresearch/improve-260706-0959/summarize_vs_old.py && python -m pytest tests/test_eval_protocol_contract.py
Iterations: 20
```

Full guard and final held-out acceptance commands are in `config.md`.
