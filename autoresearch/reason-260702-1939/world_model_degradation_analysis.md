# World Model Degradation Analysis

## Compared Runs

- Hardcone baseline: `autoresearch/eval8-restored-table2-ablation-hardcone96-300k-parallel-260702-005921`
- Cached cluster run: `autoresearch/eval8-restored-table2-hardcluster-cached-k48-300k-260702-161149`

The cached cluster run used `candidate_real_fault_priors`, while `real_fault_priors` stayed null for the GNN input.

## Main Result

The cached hard-cluster candidate generator improved the heuristic-only baseline, but reduced the world model's marginal value.

| Run | A heuristic | B rerank | C depth2 |
|---|---:|---:|---:|
| hardcone96 | -1.272% | 4.193% | 4.368% |
| clusterK48 | 3.013% | 3.378% | 3.419% |

For hardcone96, world rerank turned a bad heuristic into a good method. For clusterK48, the heuristic was already good and the world model only added a small gain, or sometimes hurt.

## Per-circuit Degradation

Largest C-depth2 drops versus hardcone96:

| Circuit | hardcone96 C | clusterK48 C | Gap |
|---|---:|---:|---:|
| max_aig | 12.898% | 8.873% | -4.025% |
| b17_C | 2.949% | 0.328% | -2.621% |
| b21_C | 4.181% | 2.045% | -2.136% |

ClusterK48 improved:

| Circuit | hardcone96 C | clusterK48 C | Gap |
|---|---:|---:|---:|
| i2c_aig | 0.205% | 1.161% | +0.956% |
| mem_ctrl_aig | 1.827% | 2.772% | +0.945% |
| b20_C | 2.133% | 2.411% | +0.278% |

## Evidence

### 1. World model marginal gain collapsed

Hardcone96:

| Circuit | A | B | C | C-A |
|---|---:|---:|---:|---:|
| max_aig | -1.417% | 12.957% | 12.898% | +14.315% |
| b21_C | -1.271% | 2.114% | 4.181% | +5.452% |
| b17_C | -2.202% | 2.227% | 2.949% | +5.151% |

ClusterK48:

| Circuit | A | B | C | C-A |
|---|---:|---:|---:|---:|
| max_aig | 7.823% | 8.819% | 8.873% | +1.050% |
| b21_C | 2.104% | 2.051% | 2.045% | -0.059% |
| b17_C | 1.348% | 1.346% | 0.328% | -1.020% |

This means the world model did not fail uniformly. It lost its ability to improve over the new candidate generator.

### 2. Candidate distribution shifted strongly

ClusterK48 candidate plans overlap much more with heuristic-only than hardcone96.

Hardcone96 B versus A action overlap was typically 0-6%, except mem_ctrl. ClusterK48 B versus A overlap was around 26-64%.

This indicates the world model is no longer making a strong independent selection; it mostly follows the candidate manager's high-ranked regions.

### 3. Candidate strategy changed too many factors at once

The comparison is not only hardcone versus cluster. It also changed:

- `max_candidates`: 96 -> 48
- real-fault prior entered candidate generation
- candidate pool became cluster-balanced and type-balanced
- cached manager filters each node/type using a 0.9 threshold
- local near-mask and dynamic type history changed the pool shape

So the observed drop cannot be attributed to one factor without additional ablations.

### 4. Model score scale is unstable

`plan_reward_sum` has extreme ranges, especially on large graphs, and does not correlate well with final TC.

For clusterK48 C-depth2, prediction sum correlation with final delta across 8 circuits was about -0.07. This confirms the reward head is useful mostly as a local ranker inside a familiar candidate distribution, not as a calibrated global utility.

### 5. Action type shift explains specific failures

Examples:

- `max_aig`: hardcone C used 56 observe, 9 control0, 29 control1. Cluster C used 39 observe, 1 control0, 54 control1. It lost many observe and control0 choices and dropped 4.0% TC.
- `b17_C`: hardcone C was heavily control1 and observe with little control0. Cluster C became much more balanced, but dropped 2.6% TC.
- `b21_C`: hardcone C used many observe actions. Cluster C shifted toward control1, dropping 2.1% TC.

The cluster manager's balancing is good for heuristic robustness, but it can suppress the action-type bias that the trained world model exploits.

## Most Likely Causes

1. Distribution shift: the world model was trained/evaluated mostly on hardcone-like candidate pools, so cluster-balanced candidates are out-of-distribution.
2. K reduction: K=48 likely removed some actions that hardcone96 made available to the model, especially on `max_aig`, `b21_C`, and `b17_C`.
3. Heuristic prior became too strong: clusterK48 already solves much of the selection problem, leaving less room for the world model and sometimes constraining it.
4. Action-type balancing conflicts with learned model preferences on some circuits.
5. Depth2 rollout is not robust when reward predictions are uncalibrated and candidate distribution shifts.

## Recommended Next Experiments

Run small, isolated ablations:

1. `hard_fault_cluster cached K96` with real prior.
2. `hard_fault_cluster cached K48` without real prior.
3. `hard_fault_cone K48` with real prior.
4. Hybrid recall pool: `hardcone 50% + cluster 30% + ffr/reconv/testability 20%`.

Success criterion:

- Keep A heuristic improvement from cluster.
- Recover B/C world model performance close to hardcone96.

Best next default:

Use hybrid recall instead of replacing hardcone:

```text
hard_fault_cone 50%
hard_fault_cluster 30%
FFR/reconvergence/testability/diversity 20%
```

