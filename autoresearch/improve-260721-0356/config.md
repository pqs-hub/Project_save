# Improve Configuration

- Mode: `improve`
- Goal: improve the exact-legal TPI pipeline until one uniform method exceeds DeepTPI final TC on b15, b20, b21, b22, and b17.
- ICP: a DFT/EDA researcher who needs physically insertable candidates, reproducible ATPG evaluation, and cross-circuit generalization rather than a circuit-specific leaderboard result.
- Competitors/reference points: DeepTPI, commercial DFT TPI, DeepGate2/3 circuit encoders, conservative offline RL, and latent world-model planning.
- Selection boundary: checkpoint, score head, and all planner hyperparameters are selected on b15 only; b20/b21/b22/b17 are frozen validation circuits.
- Fair protocol: exact original gate-level-netlist allowlists; budgets 278/616/628/915/994; Atalanta-BIST 300,000 patterns; seed 2026; one checkpoint/head/planner/config for all five circuits.
- Forbidden: per-circuit head selection, suffix ATPG enumeration, plan splicing, action rewriting, or validation-circuit calibration.
- Primary objective: lexicographically maximize (1) exact/uniform audit, (2) number of circuits above DeepTPI, (3) worst DeepTPI gap, (4) summed negative gap, and only then (5) macro TC.
- Engineering incumbent: Round13 remains the non-regression reference because Round4 merely swaps a b22 win for a b20 win.
- Depth: standard, with 15 research iterations across the five required categories.

