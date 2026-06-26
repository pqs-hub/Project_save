# Commands

## Static checks

```bash
python -m py_compile tpi_jepa/model.py tpi_jepa/plan.py tpi_jepa/scoap.py scripts/run_gmean_sweep.py
python -m tpi_jepa.plan --help
python scripts/run_gmean_sweep.py --help
```

## b17 targeted planner probe

```bash
TPI_BENCH_ROOT=/data4/pengqingsong/DFT/Dataset/deeptpi_official_aig_bench_standard \
CUDA_VISIBLE_DEVICES=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m tpi_jepa.plan \
  --checkpoint autoresearch/highseed-improvement-260626-run-posweight-30/runs/seed2030__lh0p5__lhc0p1__lhr0p5__lhrk0p0__lhb0p0__lhsf0p02__encmean__sumglobal__hlasl__agn2p0__ac0p05__hhresidual_context__pw30__ns5__nmtopk__tshard_weighted__fmtestability__ewfault_path__ek0p6__fc0p0/epoch_002.pt \
  --benchmark-id b17_C \
  --budget 5 \
  --max-candidates 32 \
  --device cuda \
  --planner beam \
  --beam-width 2 \
  --lookahead-depth 2 \
  --score-field guarded_reward \
  --beam-objective cumulative \
  --candidate-strategy hard_fault_cone \
  --candidate-diversity-penalty 0.10 \
  --candidate-diversity-depth 8 \
  --out autoresearch/plan-260626-2002/probe_b17_guarded_div0p10_d8.csv
```

## Targeted b17 50k sweep

```bash
OUT_DIR=autoresearch/safe-rollout-b17-260626-2002 \
BENCHMARKS=b17_C \
PATTERNS=50000 \
MAX_CANDIDATES=32 \
LOOKAHEAD_DEPTHS=1,2 \
BEAM_WIDTHS=1,2 \
SCORE_FIELDS=guarded_reward,return_pred \
BEAM_OBJECTIVES=mean,terminal,cumulative \
CANDIDATE_DIVERSITY_PENALTIES=0.05,0.10,0.20 \
CANDIDATE_DIVERSITY_DEPTHS=6,8,12 \
CUDA_VISIBLE_DEVICES=4 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash autoresearch/plan-260626-posweight30-world-rollout/run_posweight30_world_rollout.sh
```

## Four-benchmark promotion gate

Fill in the winning parameters from `autoresearch/safe-rollout-b17-260626-2002/grouped_results.tsv`.

```bash
OUT_DIR=autoresearch/safe-rollout-smoke-260626-2002 \
BENCHMARKS=b15_C,b17_C,i2c_aig,max_aig \
PATTERNS=50000 \
MAX_CANDIDATES=32 \
LOOKAHEAD_DEPTHS=<winning_depth> \
BEAM_WIDTHS=<winning_width> \
SCORE_FIELDS=<winning_score_field> \
BEAM_OBJECTIVES=<winning_objective> \
CANDIDATE_DIVERSITY_PENALTIES=<winning_diversity_penalty> \
CANDIDATE_DIVERSITY_DEPTHS=<winning_diversity_depth> \
CUDA_VISIBLE_DEVICES=4 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash autoresearch/plan-260626-posweight30-world-rollout/run_posweight30_world_rollout.sh
```
