# Autoresearch Plan

## Goal

Improve the world-model rerank capability using paper-inspired methods while keeping the 8 target circuits fully held out from training, calibration, and model selection.

## Scope

- `tpi_jepa/train.py`
- `tpi_jepa/model.py`
- `tpi_jepa/plan.py`
- `tpi_jepa/protocol.py`
- `configs/planner_aligned_*safe*.json`
- `scripts/*.py`
- `autoresearch/improve-260706-0959/*.sh`
- `autoresearch/improve-260706-0959/*.py`

## Forbidden Data

Do not use these for training, calibration, dev gating, or model selection:

- `autoresearch/exact-rank-table2-hybrid-k96-realfault-300k-260703-142737/*`
- `autoresearch/oracle-action-probe-260629-smoke/oracle_actions.tsv`
- Any oracle/action data whose `benchmark_id` is one of the 8 eval targets or aliases:
  `epfl__arithmetic__max__max`, `epfl__random_control__i2c__i2c`, `iscas99__b15_1`,
  `iscas99__b17`, `iscas99__b20`, `iscas99__b21`, `iscas99__b22`,
  `openabcd__mem_ctrl_orig`, `max`, `max_aig`, `i2c`, `i2c_aig`, `b15_C`,
  `b17_C`, `b20_C`, `b21_C`, `b22_C`, `mem_ctrl`, `mem_ctrl_aig`.

## Metric

Name: `non_target_dev_best_delta_tc_pct`

Direction: `higher_is_better`

Description: Best rerank delta test coverage on the local non-target development circuit `subckt_0001`. This is a proxy metric for iteration only. It must not be replaced by eval8 during model selection.

## Verify

```bash
cd /data4/pengqingsong/DFT/TPI-my.3 && GPUS_CSV=0,1,2,3,4,5,6,7 bash autoresearch/improve-260706-0959/run_dev_non_target_rerank_parallel.sh >/tmp/tpi_dev_non_target_rerank.log 2>&1 && python - <<'PY'
import csv
import math
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
```

## Guard

```bash
cd /data4/pengqingsong/DFT/TPI-my.3 && python -m py_compile tpi_jepa/protocol.py tpi_jepa/train.py tpi_jepa/model.py tpi_jepa/plan.py autoresearch/improve-260706-0959/summarize_vs_old.py && python -m pytest tests/test_eval_protocol_contract.py && python - <<'PY'
import csv
import json
from pathlib import Path

configs = sorted(Path("configs").glob("planner_aligned_*_safe.json"))
target_ids = {
    "epfl__arithmetic__max__max", "epfl__random_control__i2c__i2c",
    "iscas99__b15_1", "iscas99__b17", "iscas99__b20", "iscas99__b21",
    "iscas99__b22", "openabcd__mem_ctrl_orig", "max", "max_aig",
    "i2c", "i2c_aig", "b15_C", "b17_C", "b20_C", "b21_C", "b22_C",
    "mem_ctrl", "mem_ctrl_aig",
}
blocked_path_parts = {"exact-rank-table2", "oracle-action-probe", "deeptpi_table2"}

def action_paths(value):
    if isinstance(value, str):
        return [value]
    return [item["path"] if isinstance(item, dict) else item for item in value]

for cfg_path in configs:
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("exclude_eval_protocol") != "configs/eval_protocol_coverage_only.json":
        raise SystemExit(f"{cfg_path}: missing eval8 protocol exclusion")
    missing = target_ids - set(cfg.get("oracle_forbidden_benchmarks", []))
    if missing:
        raise SystemExit(f"{cfg_path}: missing forbidden ids {sorted(missing)}")
    for oracle_path in action_paths(cfg.get("oracle_actions", [])):
        if any(part in oracle_path for part in blocked_path_parts):
            raise SystemExit(f"{cfg_path}: blocked oracle path {oracle_path}")
        with Path(oracle_path).open(newline="") as f:
            bad = sorted({(r.get("benchmark_id") or "").strip() for r in csv.DictReader(f, delimiter="\t")} & target_ids)
        if bad:
            raise SystemExit(f"{cfg_path}: oracle contains target ids {bad}")
    with Path(cfg["labels"]).open(newline="") as f:
        bad = sorted({(r.get("benchmark_id") or "").strip() for r in csv.DictReader(f)} & target_ids)
    if bad:
        raise SystemExit(f"{cfg_path}: labels contain target ids {bad}")
print("guard passed")
PY
```

## Iterations

Recommended: `20`

Reason: this is a moderate-to-large multi-file research loop. First 8 iterations should try low-risk loss/score changes; later iterations can attempt architecture changes if proxy results plateau.

## Prioritized Experiment Units

1. `listndcg_reward_safe`: add top-heavy NDCG@8/16 ranking loss to reward route.
2. `listndcg_q_safe`: add top-heavy NDCG@8/16 ranking loss to q route with conservative weight.
3. `lcb_ensemble_safe`: combine independently trained safe seeds with lower-confidence score `mean - alpha * std`.
4. `conservative_q_safe`: add CQL-style penalty against high-scoring non-oracle-top actions.
5. `candidate_context_rerank_safe`: train a 96-candidate context reranker on non-target oracle groups.
6. `dag_encoder_reward_safe`: add DAG/topological positional encoding or DAG-aware global attention.

## Final Held-Out Acceptance

This is not the loop Verify. Run only after selecting a candidate using non-target data:

```bash
cd /data4/pengqingsong/DFT/TPI-my.3 && CONFIRM_EVAL8_HELDOUT=1 PRIOR_MODE=old GPUS_CSV=0,1,2,3,4,5,6,7 bash autoresearch/improve-260706-0959/run_eval8_final_heldout_parallel.sh
```

Accept only if `autoresearch/improve-260706-0959/eval8_final_heldout_oldbudget/comparison_vs_old.tsv` has `beats_old=True` for all 8 rows of the selected model.
