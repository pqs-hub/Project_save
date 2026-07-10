# Autoresearch Learn Summary

- Mode: `update`
- Scope: whole codebase, excluding generated experiment trees
- Depth: standard
- Output directory: `autoresearch/learn-260704-0956`

## Baseline

Existing documentation:

- `README.md`
- `docs/codebase_guide.md`
- `docs/exploration_knowledge_base.md`
- `docs/hard_f1_autoresearch_report.md`

The existing docs explain the model/data pipeline and research conclusions.
The largest gap was operator-facing documentation: which scripts to run, how
they connect, where artifacts land, and which validation commands are cheap.

## Generated Documentation

- `docs/operator_runbook.md`
- `docs/script_inventory.md`

## Remaining Gaps

- A config-field reference for `tpi_jepa/train.py` is still missing.
- Candidate strategy examples should be expanded after the planner code settles.
- Some scripts have CLI flags but minimal module-level narrative.

## Validation Plan

- Compile all package and script Python files.
- Run existing pytest tests.
- Check links from the new Markdown docs to local files.
- Scan generated docs for common secret patterns.
