# Exact-legal ITC99 best-of-run bundle

This directory pins the best exact-legal plan and fixed-protocol evaluation for
each of the five Table-IV circuits. Each `results.tsv` is a compact manifest
consumed by `scripts/summarize_exact_itc99_eval.py`; the plans and ATPG labels
remain in their original experiment directories so the evidence is not copied
or altered.

Protocol: Atalanta-BIST, 300,000 random patterns, seed 2026. Every selected AIG
node must appear in the structurally recovered original gate-level netlist.
