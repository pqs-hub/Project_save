# TPI-my.3 Operator Runbook

This runbook is for running, checking, and extending the current TPI-JEPA
pipeline. For architecture details, start with `docs/codebase_guide.md`; for
research conclusions, start with `docs/exploration_knowledge_base.md`.

## Ground Rules

- Run commands from the project root.
- Use GPU for training, model inference, planning, and checkpoint evaluation
  unless the task is explicitly CPU-only.
- Do not use CPU fallback as a silent substitute for expected CUDA execution.
- Treat `runs/` and `autoresearch/` as generated artifact roots.
- Use `configs/eval_protocol_coverage_only.json` for fixed final planner
  comparisons.
- Prefer `--stream-logs` for long sweeps so terminal output is live while logs
  are still written to disk.

## Fast Local Checks

Use these before or after documentation-only or script-only changes:

```bash
python -m py_compile tpi_jepa/*.py scripts/*.py
pytest tests
```

Use these when touching data parsing, graph construction, features, model
forward paths, or training configs:

```bash
python -m tpi_jepa.inspect_data
python -m tpi_jepa.labels
python -m tpi_jepa.smoke_test
python -m tpi_jepa.train --config configs/aig_lowtc_100k_world_model_smoke.json
```

## Main Workflows

### Inspect data

```bash
python -m tpi_jepa.inspect_data
python -m tpi_jepa.labels
python -m tpi_jepa.dataset /data4/pengqingsong/DFT/Dataset/atalanta_bist_lowtc_subckt_100k_labels/labels.csv
```

Use this path when debugging missing samples, benchmark resolution, or label
distribution drift. `tpi_jepa/labels.py` owns label loading and BENCH lookup;
`tpi_jepa/dataset.py` owns tensorized transition and rollout samples.

### Train a smoke model

```bash
CUDA_VISIBLE_DEVICES=0 python -m tpi_jepa.train \
  --config configs/aig_lowtc_100k_world_model_smoke.json
```

Expected outputs are a run directory with `latest.pt`, `best.pt`,
`history.csv`, and optional epoch checkpoints depending on config.

### Train a full model

```bash
CUDA_VISIBLE_DEVICES=0 python -m tpi_jepa.train \
  --config configs/aig_lowtc_100k_world_model_full.json
```

Important config fields include:

- `labels`
- `run_dir`
- `device`
- `feature_mode`
- `relation_mode`
- `rollout_training`
- `rollout_max_horizon`
- `lambda_hard`
- `lambda_hard_count`
- `lambda_hard_reduction`
- `oracle_actions`
- `lambda_oracle_rank`
- `exclude_eval_protocol`
- `exclude_protocol_auxiliary`

When `exclude_eval_protocol` is set, `tpi_jepa/protocol.py` removes fixed
evaluation benchmarks and optional auxiliary benchmarks before train/validation
splitting.

### Generate a plan

```bash
CUDA_VISIBLE_DEVICES=0 python -m tpi_jepa.plan \
  --checkpoint runs/aig_lowtc_100k_smoke/best.pt \
  --benchmark-id iscas89__s838 \
  --budget 5 \
  --planner beam \
  --beam-width 4 \
  --lookahead-depth 3 \
  --score-field hard_reduction_total_pred \
  --candidate-strategy hard_fault_cone \
  --out autoresearch/manual-plan/iscas89__s838.csv
```

Common planner choices:

- `greedy`: iteratively select the best next action.
- `beam`: rollout from the current latent state with bounded candidate sets.
- `beam_full`: score full action sequences.

Common score fields:

- `q_pred`
- `reward_pred`
- `return_pred`
- `hard_reduction_total_pred`
- `hybrid_pred`
- `bounded_residual_hybrid_pred`
- `derived_hard_reduction_total_pred`
- `derived_hard_reduction_hybrid_pred`

Common candidate strategies:

- `testability`
- `hard_fault`
- `hard_fault_cone`
- `hard_fault_cluster`
- `hard_fault_recall_union`
- `reconvergence`
- `ffr`
- `mixed`
- `recall_pool`
- `cached_netlist`
- `cached_hard_cone`
- `cached_stride`
- `cached_random`

Use `--candidate-cache-dir` with cached strategies. Use
`--candidate-real-fault-priors` and `--candidate-activation-priors` when the
candidate generator should use priors different from the model feature priors.

### Evaluate a plan with backend TC

```bash
python -m tpi_jepa.evaluate_plan_tmax \
  --benchmark-id iscas89__s838 \
  --plan-csv autoresearch/manual-plan/iscas89__s838.csv \
  --out-dir autoresearch/manual-plan/iscas89__s838_eval \
  --backend atalanta-bist \
  --patterns 300000 \
  --seed 2026 \
  --eval-step-mode final \
  --cleanup-workdir
```

Use `--eval-step-mode all` only when step-level records are needed, because it
is more expensive than final-only evaluation.

### Run a fixed-protocol planner sweep

For any claim compared with restored DeepTPI Table-II or the historical
8-circuit best result, use `configs/eval_protocol_coverage_only.json`. That
protocol locks the benchmark list, BENCH root, 300k-pattern setup, and Table-II
`#TPs` budgets. Do not compare against those baselines with budgets recomputed
from the current parser.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/run_gmean_sweep.py \
  --eval-protocol configs/eval_protocol_coverage_only.json \
  --checkpoint runs/tmax50k_coverage_only_full5/best.pt \
  --score-fields hard_reduction_total_pred,hybrid_pred,bounded_residual_hybrid_pred \
  --planners beam \
  --beam-objectives cumulative,terminal \
  --beam-widths 4,8 \
  --lookahead-depths 3,5 \
  --max-candidates 64,96 \
  --candidate-strategies hard_fault_cone,hard_fault_recall_union \
  --plan-device cuda \
  --eval-backend atalanta-bist \
  --time-limit-hours 12 \
  --stream-logs \
  --out-dir autoresearch/gmean-fixed-protocol
```

The sweep writes:

- `results.tsv`: per benchmark and variant.
- `grouped_results.tsv`: aggregate metrics per variant.
- `budgets.json`: resolved benchmark budgets.
- `logs/*.plan.log` and `logs/*.eval.log`: subprocess logs.
- `best.json` and best artifacts when a variant is selected.

Validate the protocol and any finished results before reporting:

```bash
python scripts/validate_eval_protocol.py \
  --protocol configs/eval_protocol_coverage_only.json \
  --results autoresearch/gmean-fixed-protocol
```

For parallel single-benchmark launchers, pass both `--eval-protocol` and
`--protocol-keep-cli-benchmarks`; otherwise the protocol intentionally expands
to the full eight-benchmark suite.

### Render the eight-benchmark report

```bash
python scripts/report_table8.py \
  --results autoresearch/gmean-fixed-protocol/results.tsv \
  --protocol configs/eval_protocol_coverage_only.json \
  --method-name TPI-JEPA \
  --out-md autoresearch/gmean-fixed-protocol/table8_report.md
```

### Evaluate hard-fault checkpoints

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_hard_checkpoints.py \
  --config configs/aig_lowtc_100k_hard_pretrain.json \
  --run-dir runs/aig_lowtc_100k_full \
  --max-val-samples 1024 \
  --max-steps 256 \
  --device cuda \
  --diagnostics-dir autoresearch/hard-eval-diagnostics \
  --write-action-ranking-diagnostics
```

This is the right validation path for hard-fault labels, hard-count heads,
calibration reports, and action-ranking diagnostics tied to checkpoint
selection.

### Finetune or score oracle action-value data

For checkpoint finetuning:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/finetune_oracle_action_values.py \
  --checkpoint runs/current/best.pt \
  --oracle-actions autoresearch/oracle-train/oracle_actions.tsv \
  --val-oracle-actions autoresearch/oracle-val/oracle_actions.tsv \
  --out-dir autoresearch/oracle-finetune \
  --train-scope bounded_residual \
  --ranking-score-field bounded_residual_hybrid_pred \
  --value-score-field bounded_residual_hybrid_pred \
  --epochs 3 \
  --plan-device cuda
```

For offline rescoring without changing backend labels, use
`scripts/evaluate_oracle_action_values.py` on an existing `oracle_actions.tsv`.
This avoids backend and candidate-sampling noise during checkpoint comparison.

## Debugging Map

- Missing BENCH path: inspect `tpi_jepa/labels.py` and environment variables
  `DFT_ROOT` and `TPI_BENCH_ROOT`.
- Sample disappears during filtering: inspect `TPIDataset._filter_specs()` in
  `tpi_jepa/dataset.py`.
- Feature dimension mismatch: inspect `tpi_jepa/features.py`, config
  `feature_mode`, and checkpoint `feature_dim`.
- Relation dimension mismatch: inspect `relation_mode`, `relation_depth`, and
  `make_action_relation_features()`.
- Planner candidate quality issue: inspect `enumerate_candidates()` and the
  strategy-specific candidate functions in `tpi_jepa/plan.py`.
- Multi-circuit planner cache issue: ensure `clear_planner_caches()` is called
  by batch scripts before switching benchmarks.
- Backend evaluation failure: inspect the per-run evaluator log first, then
  `tpi_jepa/evaluate_plan_tmax.py`.

## Artifact Hygiene

- Put one-off experiment results under a named `autoresearch/<topic>/`
  directory.
- Keep generated logs out of source docs unless they are summarized.
- Store selection evidence in TSV/JSON files, not only terminal output.
- Write `handoff.json` for experiment directories that should feed future
  automation.
