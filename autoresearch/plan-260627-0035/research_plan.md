# No-Oracle Coverage Research Plan

## Objective

Maximize no-oracle generalization coverage under the frozen evaluation protocol:

```text
maximize macro_mean_delta_tc
```

The final claim is valid only when measured on `configs/eval_protocol_coverage_only.json`:

- fixed 8 final circuits
- `patterns=300000`
- `seed=2026`
- budget `max(1, floor(logic_gate_count * 0.01))`
- no final-circuit TMAX fault logs or `real_fault_priors` in model inputs or planner ranking
- selection rule: safe variants first, then maximize `macro_mean_delta_tc`, then `min_delta_tc`, `router_delta_tc`, lower `negative_count`

Hard-F1, JEPA loss, latent loss, return loss, and gate_dir behavior are diagnostics, not success criteria.

## Scope

Primary scope:

- candidate generation and candidate recall diagnostics
- fixed-candidate ranking diagnostics
- final protocol evaluation and baseline comparisons
- leakage guards preventing final-circuit fault priors
- seed-stability reports

Secondary ablation scope:

- `mean` vs `gate_dir`
- greedy vs beam
- basic vs cone relation
- proxy state update vs static
- discounted return vs one-step reward

Out of scope until candidate recall is healthy:

- deeper JEPA architectures
- additional hard-F1 losses
- larger beam searches
- adopting `gate_dir` as default without final coverage evidence

## Stage 1: Candidate Recall Gate

Goal:

```text
Candidate Recall@K up, before changing the world model.
```

Working metric:

```text
recall_at_32 >= 0.50
recall_at_64 >= 0.70
recall_at_128 >= 0.85
```

Current measured anchor:

```text
mixed recall_at_32 = 0.140625 on 256 samples
```

Allowed changes:

- broaden or diversify `enumerate_candidates`
- add structural candidate strategies not using final-circuit real fault logs
- use SCOAP, FFR, reconvergence, topology, cone size, centrality, and graph-only priors
- report per-circuit and per-action-type recall

Forbidden:

- using final eval circuit `real_fault_priors`
- using final TMAX undetected-node CSVs as ranking features
- tuning candidate rules directly on final 8 TMAX outcomes

Verify:

```bash
python scripts/candidate_recall_diagnostics.py \
  --candidate-strategy <strategy> \
  --top-k 32,64,128 \
  --max-sequences 4096
```

Required diagnostic extension:

```text
candidate_recall_diagnostics should emit:
- candidate_strategy
- K
- recall_at_K
- checked
- per_benchmark_recall_at_K
- per_action_type_recall_at_K
```

Keep/discard rule:

- keep a candidate strategy only if it improves Recall@64 without collapsing Recall@32 or exploding duplicate local-cone candidates
- discard if recall gains come from oracle fault priors or final eval leakage

## Stage 2: Fixed-Candidate Ranking Gate

Goal:

```text
Given a candidate set that contains useful actions, prove the ranker selects better actions than structural heuristics.
```

Freeze:

- same candidate files or same candidate strategy
- same candidate budget
- same training split
- no final eval fault priors

Compare:

- candidate order baseline
- SCOAP heuristic ranking
- traditional structural heuristics
- `mean` encoder
- `gate_dir` encoder
- greedy
- beam
- oracle top-1 within candidate set

Metrics:

```text
planner_regret = oracle_candidate_delta_tc - selected_delta_tc
top1_action_accuracy
topk_action_accuracy
spearman(pred_reward, true_delta_tc)
pearson(pred_reward, true_delta_tc)
calibration_by_reward_bin
negative_predicted_gain_rate
```

Acceptance:

```text
model planner_regret < heuristic planner_regret
model top1/topK accuracy > heuristic top1/topK accuracy
reward correlation is positive and stable across seeds
```

Gate_dir rule:

Adopt `gate_dir` only if it improves fixed-candidate ranking or final protocol metrics under identical candidate sets and seeds. Otherwise keep `mean` as default.

## Stage 3: Final Coverage Gate

Goal:

```text
TC(plan_model) - TC(plan_baseline) > 0
```

Primary command template:

```bash
python scripts/run_gmean_sweep.py \
  --eval-protocol configs/eval_protocol_coverage_only.json \
  --checkpoint <checkpoint.pt> \
  --candidate-strategies <no_oracle_candidate_strategy> \
  --max-candidates <fixed_budget> \
  --candidate-sample-seeds 0,1,2 \
  --planners greedy,beam \
  --score-fields reward_pred,return_pred \
  --beam-objectives cumulative,discounted \
  --real-fault-priors "" \
  --activation-priors "" \
  --plan-device cpu \
  --out-dir autoresearch/no-oracle-final-<stamp>
```

Primary metric:

```text
macro_mean_delta_tc
```

Required safety metrics:

```text
min_delta_tc
negative_count
router_delta_tc
positive_count
per_benchmark_delta_test_coverage
```

Acceptance:

```text
macro_mean_delta_tc > best non-oracle baseline
negative_count <= baseline negative_count
min_delta_tc >= baseline min_delta_tc
result holds for multiple candidate_sample_seeds
```

Baseline set:

- random candidate baseline with same budget and seeds
- SCOAP/testability candidate baseline
- reconvergence baseline
- FFR baseline
- current `mixed` baseline
- candidate-only plan from `scripts/plan_candidate_baseline.py`

## Leakage Rules

Allowed:

- training rows excluding eval protocol circuits
- graph structure from final BENCH files
- SCOAP and structural proxies computed from final BENCH files
- fixed protocol metadata: circuit IDs, pattern count, budget rule

Forbidden for final claims:

- final-circuit TMAX fault logs as input features
- final-circuit `real_fault_priors`
- final-circuit undetected-node CSVs in candidate ranking
- selecting variants based on final 8 results and reporting them as unbiased

Configs should keep:

```json
"exclude_eval_protocol": "configs/eval_protocol_coverage_only.json",
"exclude_protocol_auxiliary": true
```

Final sweep command should omit `--real-fault-priors` or pass an empty value.

## Success Predicate

A run is considered successful only if a frozen-protocol result file exists and the selected no-oracle model beats the no-oracle baseline by the protocol selection rule:

```text
grouped_results.tsv contains a safe no-oracle model row with:
macro_mean_delta_tc > best_no_oracle_baseline_macro_mean_delta_tc
negative_count <= best_no_oracle_baseline_negative_count
min_delta_tc >= best_no_oracle_baseline_min_delta_tc
```

Before Stage 3, the intermediate success predicate is:

```text
candidate_recall_at_64 >= 0.70 on a non-final held-out label sample
```

