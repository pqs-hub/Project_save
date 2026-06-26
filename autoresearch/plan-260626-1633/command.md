# Commands

## Static checks

```bash
python -m py_compile tpi_jepa/plan.py scripts/run_gmean_sweep.py
python -m tpi_jepa.plan --help
python scripts/run_gmean_sweep.py --help
```

## Planner-only large-graph smoke

```bash
CUDA_VISIBLE_DEVICES=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m tpi_jepa.plan \
  --checkpoint autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt \
  --benchmark-id b22_C \
  --budget 5 \
  --max-candidates 32 \
  --device cuda \
  --planner beam \
  --beam-width 2 \
  --lookahead-depth 2 \
  --score-field reward_pred \
  --beam-objective cumulative \
  --candidate-strategy hard_fault_cone \
  --out autoresearch/plan-260626-1633/smoke_b22_plan.csv
```

## Low-cost rollout gate

```bash
OUT_DIR=autoresearch/safe-rollout-smoke-260626-1633 \
BENCHMARKS=b15_C,b17_C,i2c_aig,max_aig \
PATTERNS=50000 \
MAX_CANDIDATES=32 \
LOOKAHEAD_DEPTHS=2 \
BEAM_WIDTHS=2 \
CUDA_VISIBLE_DEVICES=4 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash autoresearch/plan-260626-posweight30-world-rollout/run_posweight30_world_rollout.sh
```
