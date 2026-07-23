# Research Findings

## Outcome

The current bottleneck is policy supervision, not raw GNN capacity. Round4 learned only from actions selected by its teacher and then changed the state distribution at deployment. It improved b17/b20 but over-selected CP0 and changed b22 from step 2, converting one existing win into a loss. A larger encoder would still be trained against this biased action distribution.

DeepTPI also resolves the legality question explicitly: it converts each original gate into an equivalent AIG fragment without logic optimization and masks fragment-only nodes that have no original-gate position. The recovered exact allowlists in this repository therefore match the paper's intended candidate semantics.

## High-confidence mechanism

1. Roll out the current learner on non-target training subcircuits.
2. At the same visited prefix, evaluate a balanced list of CP0, CP1, and OP alternatives with real 300k-pattern ATPG.
3. Train listwise/pairwise candidate ranking, typed marginal/return heads, and separate SA0/SA1 reduction heads.
4. Apply the typed model as a conservative residual over the proven Round13/base score. A candidate may overturn the base action only when enough independent typed heads agree relative to the base action; CP0 requires unanimous support.
5. Select checkpoint and every trust parameter on b15 only, then freeze and evaluate all five exact-legal circuits.

This combines DAgger-style state-distribution correction, listwise candidate supervision, and an offline-RL support constraint. It directly targets the observed failure rather than introducing per-circuit rules.

## Why the architecture upgrade is second

DeepGate2/3 and GraphGPS support a later encoder upgrade: functionality-aware pair supervision and sparse/global subcircuit context can address long-range logic semantics and GNN over-squashing. They are not the first experiment because the current train/deploy mismatch can make a stronger model confidently learn the wrong ranking. First establish trustworthy counterfactual labels and a safe residual; then compare an encoder change under the same labels and b15-only selection.

## Evaluation rule

The score is lexicographic, not macro-only:

1. exact candidate legality and one uniform configuration must pass;
2. maximize the number of circuits above DeepTPI;
3. maximize the worst circuit gap;
4. minimize summed deficits;
5. use macro TC only as the final tie-break.

Round13 remains the engineering incumbent until a candidate improves this objective without turning a previously winning circuit into a loser.

## Primary sources

- DeepTPI: https://arxiv.org/abs/2206.06975
- DAgger: https://proceedings.mlr.press/v15/ross11a/ross11a.pdf
- Conservative Q-Learning: https://arxiv.org/abs/2006.04779
- Supported Trust Region: https://proceedings.mlr.press/v202/mao23c.html
- DeepGate2: https://arxiv.org/abs/2305.16373
- DeepGate3: https://arxiv.org/abs/2407.11095
- TD-MPC2: https://arxiv.org/abs/2310.16828
- Decision Transformer: https://proceedings.neurips.cc/paper_files/paper/2021/hash/7f489f642a0ddb10272b5c31057f0663-Abstract.html
- EDALearn: https://arxiv.org/abs/2312.01674

