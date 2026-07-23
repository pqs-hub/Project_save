# Exact-legal ITC99 framework optimization

- Goal: make final TC exceed DeepTPI Table IV on each of b15, b20, b21, b22, and b17.
- Scope: `tpi_jepa/plan.py`, exact-candidate recovery inputs, planner/evaluation scripts, and planner training when needed.
- Metric: `-sum(remaining per-circuit TC deficits) + macro_TC / 1000`; higher is better.
- Verify: `python scripts/score_exact_itc99_vs_deeptpi.py autoresearch/exact-itc99-current-best-q-lcb-260712/summary.json`
- Success predicate: the same command with `--check` exits zero.
- Guard: exact-plan legality plus the mapping/scoring test suite.
- Working circuits: b15 and b17. Regression circuits: b20, b21, and b22.
- Iterations: 68 bounded, extended after the user requested continuation, with the fixed Atalanta-BIST coverage protocol checked before keeping a change.

The starting final TC was 89.987/96.538/95.752/96.615/90.390. The correct
DeepTPI final TC is 93.20/95.02/94.51/95.59/91.67.
