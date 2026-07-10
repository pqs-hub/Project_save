# TPI-my.3 Script Inventory

This inventory groups the project scripts by operational role. It is based on
the current CLI entry points in `tpi_jepa/` and `scripts/`.

## Package Entry Points

| Entry point | Role | Primary outputs |
| --- | --- | --- |
| `python -m tpi_jepa.inspect_data` | Inspect default labels and dataset shape. | Terminal summary |
| `python -m tpi_jepa.labels` | Load labels and verify BENCH resolution. | Terminal summary |
| `python -m tpi_jepa.dataset` | Build dataset samples from labels. | Terminal summary |
| `python -m tpi_jepa.smoke_test` | End-to-end parse, graph, feature, model, loss check. | Terminal summary |
| `python -m tpi_jepa.train` | Train the TPI-JEPA world model. | Checkpoints and `history.csv` |
| `python -m tpi_jepa.plan` | Generate test-point plans from a checkpoint. | Plan CSV |
| `python -m tpi_jepa.evaluate_plan_tmax` | Evaluate a plan with TMAX or Atalanta-BIST. | `labels.csv`, reports, backend logs |

## Data and Label Tools

| Script | Purpose |
| --- | --- |
| `scripts/relabel_sequences_with_backend.py` | Recompute sequence labels by running a backend over existing insertion sequences. |
| `scripts/run_relabel_batches.py` | Split relabeling work into batches and run backend jobs in parallel. |
| `scripts/merge_label_sets.py` | Merge multiple label CSVs into a combined label set. |
| `scripts/build_sequence_label_subset.py` | Build a smaller sequence label subset for controlled experiments. |
| `scripts/build_distill_label_mix.py` | Combine label sources for distillation-style training. |
| `scripts/restore_deeptpi_npz_bench.py` | Restore BENCH files from a DeepTPI NPZ archive. |
| `scripts/collect_real_fault_logs.py` | Run backend fault simulation and collect hard-fault sidecar artifacts. |
| `scripts/build_real_fault_priors.py` | Convert real-fault artifacts into node/action priors. |
| `scripts/build_activation_priors.py` | Estimate activation priors by random simulation. |

## Candidate and Planner Tools

| Script | Purpose |
| --- | --- |
| `scripts/build_tp_candidate_cache.py` | Build cached test-point candidate pools. |
| `scripts/build_hard_tp_candidate_cache.py` | Build hard-fault-aware candidate caches. |
| `scripts/profile_candidate_generation.py` | Time and inspect candidate generation strategies. |
| `scripts/candidate_recall_diagnostics.py` | Measure whether candidate pools recall oracle or target actions. |
| `scripts/plan_candidate_baseline.py` | Produce candidate-baseline plans for comparison. |
| `scripts/evaluate_existing_plans.py` | Run backend evaluation on plan CSVs that already exist. |
| `scripts/run_gmean_sweep.py` | Run the main fixed-protocol planner sweep and aggregate metrics. |
| `scripts/validate_eval_protocol.py` | Validate fixed Table-II protocol budgets and result files before reporting. |
| `scripts/run_hfc_ablation.py` | Compare hard-fault-cone planner variants. |
| `scripts/run_eval8_hard_fault_cone_budget5.sh` | Shell wrapper for an eight-benchmark hard-fault-cone evaluation. |
| `scripts/run_eval8_restored_table2_parallel.sh` | Shell wrapper for restored Table-2-style evaluation. |
| `scripts/run_eval8_restored_table2_ablation_parallel.sh` | Shell wrapper for restored Table-2 ablations. |
| `scripts/run_mainline_ablation_eval8.sh` | Shell wrapper for mainline eight-benchmark ablations. |
| `scripts/run_exact_rank_probe_table2_parallel.sh` | Shell wrapper for exact-rank probes on Table-2-style benchmarks. |

## Training and Sweep Tools

| Script | Purpose |
| --- | --- |
| `scripts/overnight_framework_search.py` | Sweep architecture, feature, candidate, and planner settings over a time budget. |
| `scripts/overnight_rollout_search.py` | Sweep rollout-training variants. |
| `scripts/run_predictive_autoresearch.py` | Generate and run predictive hard-fault pretraining variants. |
| `scripts/run_q_oracle_experiment.py` | Coordinate Q-oracle training and gate evaluation. |
| `scripts/run_q_v1_train_eval.py` | Train/evaluate one Q-v1 variant. |
| `scripts/run_q_v1_parallel.sh` | Shell wrapper for Q-v1 parallel experiments. |
| `scripts/run_q_v2_parallel.sh` | Shell wrapper for Q-v2 parallel experiments. |
| `scripts/run_rollout_loss_ablation_parallel.sh` | Shell wrapper for rollout-loss ablations. |
| `scripts/run_ab_oracle_rank_experiment.py` | Run A/B oracle-ranking experiments. |

## Oracle Action-Value Tools

| Script | Purpose |
| --- | --- |
| `scripts/oracle_action_value_probe.py` | Generate backend-labeled oracle action groups. |
| `scripts/evaluate_oracle_action_values.py` | Rescore fixed oracle actions with a checkpoint. |
| `scripts/finetune_oracle_action_values.py` | Finetune checkpoint heads or bounded residuals on oracle actions. |
| `scripts/build_balanced_oracle_action_subset.py` | Build balanced oracle train/validation subsets. |
| `scripts/sample_negative_rich_oracle_subckts.py` | Select subcircuits likely to provide negative-rich oracle groups. |
| `scripts/audit_oracle_action_groups.py` | Summarize oracle group quality and risk buckets. |
| `scripts/merge_oracle_action_tsv.py` | Merge oracle action TSVs while preserving group keys. |
| `scripts/train_action_value_ranker.py` | Train a lightweight ranker over rescored oracle action features. |
| `scripts/evaluate_q_calibration.py` | Evaluate Q-score calibration and promotion gates. |

## Diagnostics and Reporting

| Script | Purpose |
| --- | --- |
| `scripts/evaluate_hard_checkpoints.py` | Evaluate hard-fault checkpoint metrics, calibration, and action-ranking diagnostics. |
| `scripts/evaluate_q_calibration.py` | Evaluate Q-score calibration against oracle and transfer gates. |
| `scripts/measure_world_model_precision.py` | Probe exact candidate ranking and reward precision. |
| `scripts/proxy_diagnostics.py` | Compute proxy metric diagnostics over rescored TSVs. |
| `scripts/return_target_diagnostics.py` | Inspect return-target construction. |
| `scripts/state_update_diagnostics.py` | Validate proxy state updates produce expected nonzero deltas. |
| `scripts/planner_encoder_matrix.py` | Smoke matrix over encoder, relation, and planner combinations. |
| `scripts/summarize_tac_sweep.py` | Summarize testability-aware candidate sweep results. |
| `scripts/report_table8.py` | Render final eight-benchmark TC improvement tables. |
| `scripts/collect_q_v1_results.py` | Collect Q-v1 experiment result summaries. |
| `scripts/collect_q_v2_results.py` | Collect Q-v2 experiment result summaries. |
| `scripts/collect_rollout_loss_results.py` | Collect rollout-loss ablation summaries. |

## Repeated Patterns

- Long sweeps write `config.json`, `results.tsv`, `grouped_results.tsv`,
  per-run logs, and best-artifact summaries under `autoresearch/`.
- Backend evaluation scripts usually expose `--patterns`, `--seed`,
  `--timeout-sec`, `--backend`, and `--cleanup-workdir`.
- Planner wrappers usually call `python -m tpi_jepa.plan` first, then
  `python -m tpi_jepa.evaluate_plan_tmax`.
- Oracle comparison should reuse fixed `oracle_actions.tsv` files when possible
  so model comparison is not confounded by new candidate sampling or backend
  variance.
- GPU-oriented scripts commonly have `--device`, `--plan-device`,
  `--parallel-devices`, or CUDA environment variables. Keep these explicit in
  run commands.

## Known Documentation Gaps

- Several scripts have useful CLI flags but short or missing module docstrings.
- Shell wrappers encode current experiment recipes but not all assumptions in
  comments.
- The config JSON schema is implicit in `tpi_jepa/train.py`; there is no
  generated config-field reference yet.
- Candidate strategies are implemented in one large planner module, so future
  documentation should add strategy-specific examples when that code stabilizes.
